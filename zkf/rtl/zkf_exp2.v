// Streamed base-2 exponential for the Zubax Kulibin float format: y = 2**x.
// Zero-bubble, throughput-1, no backpressure.
// Behavior:
//
//   exp2(-inf)   = +0
//   exp2(+0)     = 1.0
//   exp2(finite) = 2**x, faithfully rounded
//   exp2(+inf)   = +inf
//   tiny finite results follow the zero/MIN_NORMAL boundary rule; overflow maps to +inf
//
// Algorithm:
//
//  1. Split x = i + f with i = floor(x) and f in [0,1) by shifting the significand by the exponent into a fixed-point.
//
//  2. Then 2**x = 2**f * 2**i, where 2**f in [1,2) is a normalized significand produced by the pipelined per-WMAN
//     table+polynomial core selected by the generate-if below.
//
//  3. The result is packed with exponent i via _zkf_pack, which applies overflow->inf and tiny/MIN_NORMAL boundary.
//
// The reduction is split across register stages (shift-amount computation, barrel shift, negate) and the evaluator's
// ROM read is followed by a mandatory fabric register, so no single stage carries both a wide carry chain and a
// multiply.
//
// STAGE_PRODUCT selects product computation staging; see _zkf_pmul.
// WMULTIPLIER optionally hints the native DSP tile argument width; see _zkf_pmul.
// STAGE_PACK={0,1} forwards to _zkf_pack.STAGE_INPUT, registering the packer's input cone (+1 cycle).
// STAGE_OUTPUT={0,1} registers the output.

`default_nettype none

module zkf_exp2 #(
    parameter WEXP          = 6,
    parameter WMAN          = 18,   // significand precision including the hidden bit
    parameter WMULTIPLIER   = 0,    // see _zkf_pmul
    parameter STAGE_INPUT   = 0,    // number of input register stages (>=0); +STAGE_INPUT cycles
    parameter STAGE_REDUCE  = 0,    // 0: direct fixed->ROM input; 1: register reduced i/f/flags, +1 stage
    parameter STAGE_PRODUCT = 0,    // see _zkf_pmul
    parameter STAGE_PACK    = 0,    // 0: comb pack input; 1: register pack input (+1 stage)
    parameter STAGE_OUTPUT  = 0,    // 0: combinational outputs;  1: registered outputs, +1 stage
    parameter LATENCY       = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] x,

    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    localparam DEGREE = (((WMAN+18)/11)-1);
    localparam LATENCY_REF = STAGE_INPUT + STAGE_REDUCE + 4 + DEGREE*(2+STAGE_PRODUCT) + STAGE_PACK + STAGE_OUTPUT;
    generate
        if ((WEXP < 2) || (WMAN < 4)) begin : g_invalid_wman
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        // The reduction's constants use unsized integer shifts on WEXP; WEXP >= 31 would overflow Verilog's 32-bit
        // integer constant arithmetic.
        if (WEXP >= 31) begin : g_invalid_wexp_too_wide
            _zkf_invalid_exp2_wexp_too_wide_unportable u_invalid();
        end
        if ((STAGE_REDUCE != 0) && (STAGE_REDUCE != 1)) begin : g_invalid_stage_reduce
            _zkf_invalid_stage_reduce u_invalid();
        end
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    localparam WFRAC = WMAN - 1;
    // FF: fraction bits kept for the reduced argument f. MUST equal the generator's GUARD_FF (zkf_transcendental.py).
    localparam FF   = WMAN + 12;
    localparam WEU  = WEXP;                // signed unbiased exponent fed to _zkf_pack
    localparam SBW  = WEU + 4;             // evaluator sideband: {i, force_inf, force_zero, is_zero, lost_sticky}

    wire              rb_valid;
    wire [WEU+FF-1:0] rb_mag;
    wire              rb_lost_sticky;
    wire              rb_sign;
    wire              rb_is_zero;
    wire              rb_oor;
    _zkf_exp2_reduce #(.WEXP(WEXP), .WMAN(WMAN), .FF(FF), .STAGE_INPUT(STAGE_INPUT)) u_reduce (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .x(x),
        .out_valid(rb_valid),
        .mag(rb_mag),
        .lost_sticky(rb_lost_sticky),
        .sign(rb_sign),
        .is_zero(rb_is_zero),
        .oor(rb_oor)
    );

    // -- RB2 combinational: form the signed two's-complement value, then split into the signed integer part i and
    // the unsigned fraction f. Negating an unsigned magnitude gives correct signed floor semantics for negative x:
    // floor(-3.25) maps to -4 because the slice picks up the sign-extended integer bits. The OOR cases are routed
    // via rb_oor downstream (force_inf for positive overflow, force_zero for negative), so the magnitude and split
    // for those inputs are don't-care.
    wire signed [WEU+FF:0]   v_signed = rb_sign ? (~{1'b0, rb_mag} + {{(WEU+FF){1'b0}}, 1'b1}) : {1'b0, rb_mag};
    wire signed [WEU:0]      i_full   = v_signed[WEU+FF:FF];
    wire [FF-1:0]            f_bits   = v_signed[FF-1:0];
    wire signed [WEU-1:0]    i_clamped     = i_full[WEU-1:0];   // i fits in WEXP signed bits when oor=0
    wire                     force_inf_in  = rb_oor & ~rb_sign; // +inf / positive overflow
    wire                     force_zero_in = rb_oor &  rb_sign; // -inf / negative underflow

    wire                     eval_in_valid;
    wire signed [WEU-1:0]    eval_i;
    wire [FF-1:0]            eval_f;
    wire                     eval_force_inf;
    wire                     eval_force_zero;
    wire                     eval_is_zero;
    wire                     eval_lost_sticky;

    generate
        if (STAGE_REDUCE != 0) begin : g_reduce_stage
            reg                    r0_valid;
            reg signed [WEU-1:0]   r0_i;
            reg [FF-1:0]           r0_f;
            reg                    r0_force_inf;
            reg                    r0_force_zero;
            reg                    r0_is_zero;
            // Registered lost-sticky. At the per-PR STAGE_REDUCE=1 coverage format (w2m16, WEXP<=4) lost_sticky is
            // structurally 0, so this store constant-folds to a line Verilator 5.048 cannot mark covered.
            // verilator coverage_off
            reg                    r0_lost_sticky;
            // verilator coverage_on

            always @(posedge clk) begin
                if (rst) begin
                    r0_valid <= 1'b0;
                end else begin
                    r0_valid <= rb_valid;
                end

                r0_i           <= i_clamped;
                r0_f           <= f_bits;
                r0_force_inf   <= force_inf_in;
                r0_force_zero  <= force_zero_in;
                r0_is_zero     <= rb_is_zero;
                r0_lost_sticky <= rb_lost_sticky;
            end

            assign eval_in_valid    = r0_valid;
            assign eval_i           = r0_i;
            assign eval_f           = r0_f;
            assign eval_force_inf   = r0_force_inf;
            assign eval_force_zero  = r0_force_zero;
            assign eval_is_zero     = r0_is_zero;
            assign eval_lost_sticky = r0_lost_sticky;
        end else begin : g_reduce_direct
            assign eval_in_valid    = rb_valid;
            assign eval_i           = i_clamped;
            assign eval_f           = f_bits;
            assign eval_force_inf   = force_inf_in;
            assign eval_force_zero  = force_zero_in;
            assign eval_is_zero     = rb_is_zero;
            assign eval_lost_sticky = rb_lost_sticky;
        end
    endgenerate

    // -- Pipelined evaluator: 2**f significand + GRS. The sideband {i, force_inf, force_zero, is_zero, lost} is delayed
    // inside the generated evaluator by a plain pipe, aligned to the evaluator output.
    wire [SBW-1:0]  sb_in_e = {
        eval_i, eval_force_inf, eval_force_zero, eval_is_zero, eval_lost_sticky
    };
    wire            ev_valid;
    wire [SBW-1:0]  sb_out_e;   // bit 0 = lost; high bits sliced into e_i/e_finf/e_fzero/e_is_zero below
    wire [WMAN-1:0] eval_sig;
    wire            eval_guard;
    wire            eval_round;
    wire            eval_sticky;
    // The table+polynomial core is pre-generated per WMAN by zkf_transcendental.py as _zkf_exp2_m<WMAN>. We pass the
    // closed-form degree D below; the core asserts it equals the degree its ROM was fitted for (mirrors the LATENCY
    // parameter), so the Horner depth / latency cannot drift.
    // Intentional: unsupported in-range WMAN names missing _zkf_exp2_m<WMAN>, prompting table generation.
    `define ZKF_EXP2_TABLE(W) end else if (WMAN == W) begin \
        _zkf_exp2_m``W #( \
            .D(DEGREE), .WSB(SBW), \
            .WMULTIPLIER(WMULTIPLIER), .STAGE_PRODUCT(STAGE_PRODUCT) \
        ) u_eval ( \
            .clk(clk), .rst(rst), .in_valid(eval_in_valid), .sb_in(sb_in_e), .f(eval_f), \
            .out_valid(ev_valid), .sb_out(sb_out_e), .significand(eval_sig), \
            .guard(eval_guard), .round(eval_round), .sticky(eval_sticky));
    // verilog_lint: waive-start generate-label  (macro-expanded selector blocks are intentionally unlabeled)
    generate
        if (1'b0) begin  // seed: the macro opens with "end else if", so every table line is uniform
        `ZKF_EXP2_TABLE(4)
        `ZKF_EXP2_TABLE(5)
        `ZKF_EXP2_TABLE(6)
        `ZKF_EXP2_TABLE(7)
        `ZKF_EXP2_TABLE(8)
        `ZKF_EXP2_TABLE(9)
        `ZKF_EXP2_TABLE(10)
        `ZKF_EXP2_TABLE(11)
        `ZKF_EXP2_TABLE(12)
        `ZKF_EXP2_TABLE(13)
        `ZKF_EXP2_TABLE(14)
        `ZKF_EXP2_TABLE(15)
        `ZKF_EXP2_TABLE(16)
        `ZKF_EXP2_TABLE(17)
        `ZKF_EXP2_TABLE(18)
        `ZKF_EXP2_TABLE(19)
        `ZKF_EXP2_TABLE(20)
        `ZKF_EXP2_TABLE(21)
        `ZKF_EXP2_TABLE(22)
        `ZKF_EXP2_TABLE(23)
        `ZKF_EXP2_TABLE(24)
        `ZKF_EXP2_TABLE(25)
        `ZKF_EXP2_TABLE(26)
        `ZKF_EXP2_TABLE(27)
        `ZKF_EXP2_TABLE(28)
        `ZKF_EXP2_TABLE(29)
        `ZKF_EXP2_TABLE(30)
        `ZKF_EXP2_TABLE(31)
        `ZKF_EXP2_TABLE(32)
        `ZKF_EXP2_TABLE(33)
        `ZKF_EXP2_TABLE(34)
        `ZKF_EXP2_TABLE(35)
        `ZKF_EXP2_TABLE(36)
        `ZKF_EXP2_TABLE(37)
        `ZKF_EXP2_TABLE(38)
        `ZKF_EXP2_TABLE(39)
        `ZKF_EXP2_TABLE(40)
        `ZKF_EXP2_TABLE(41)
        `ZKF_EXP2_TABLE(42)
        `ZKF_EXP2_TABLE(43)
        `ZKF_EXP2_TABLE(44)
        `ZKF_EXP2_TABLE(45)
        `ZKF_EXP2_TABLE(46)
        `ZKF_EXP2_TABLE(47)
        `ZKF_EXP2_TABLE(48)
        `ZKF_EXP2_TABLE(49)
        `ZKF_EXP2_TABLE(50)
        `ZKF_EXP2_TABLE(51)
        `ZKF_EXP2_TABLE(52)
        `ZKF_EXP2_TABLE(53)
        end else begin
            _zkf_invalid_unsupported_table_wman u_invalid();
        end
    endgenerate
    `undef ZKF_EXP2_TABLE
    // verilog_lint: waive-stop generate-label
    wire signed [WEU-1:0] e_i        = sb_out_e[SBW-1 -: WEU];
    wire                  e_finf     = sb_out_e[3];
    wire                  e_fzero    = sb_out_e[2];
    wire                  e_is_zero  = sb_out_e[1];
    wire                  e_lost     = sb_out_e[0];   // lost-sticky; see rb_lost_sticky above

    // For x == +0, 2**0 = 1.0 (exp_unbiased 0, significand 1.0, no GRS); otherwise 2**f * 2**i.
    wire signed [WEU-1:0] pack_exp = e_is_zero ? {WEU{1'b0}} : e_i;
    wire [WMAN-1:0]       pack_sig = e_is_zero ? {1'b1, {WFRAC{1'b0}}} : eval_sig;
    wire                  pack_g   = e_is_zero ? 1'b0 : eval_guard;
    wire                  pack_r   = e_is_zero ? 1'b0 : eval_round;
    wire                  pack_s   = e_is_zero ? 1'b0 : (eval_sticky | e_lost);

    _zkf_pack #(
        .WEXP(WEXP), .WMAN(WMAN), .WEXP_UNBIASED(WEU),
        .STAGE_INPUT(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT)
    ) u_pack (
        .clk(clk),
        .rst(rst),
        .in_valid(ev_valid),
        .sign(1'b0),
        .force_zero(e_fzero),
        .force_inf(e_finf),
        .exp_unbiased(pack_exp),
        .significand(pack_sig),
        .guard(pack_g),
        .round(pack_r),
        .sticky(pack_s),
        .out_valid(out_valid),
        .y(y)
    );
endmodule

// The magnitude of x as a fixed-point number with FF fraction bits, two register stages after STAGE_INPUT.
// The caller forms the signed value from {sign, mag}. lost_sticky is the OR of the bits dropped below mag's LSB, which
// needs an exponent below -13. oor: |x| >= 2^(WEXP-1), inf included; the result exponent cannot represent it.
// mag and lost_sticky are don't-care when oor or is_zero.
module _zkf_exp2_reduce #(
    parameter WEXP        = 6,
    parameter WMAN        = 18,
    parameter FF          = WMAN + 12,
    parameter STAGE_INPUT = 0
) (
    input  wire clk,
    input  wire rst,

    input  wire                 in_valid,
    input  wire [WEXP+WMAN-1:0] x,

    output wire                 out_valid,
    output wire   [WEXP+FF-1:0] mag,
    output wire                 lost_sticky,
    output wire                 sign,
    output wire                 is_zero,
    output wire                 oor
);
    localparam WFRAC   = WMAN - 1;
    localparam WFULL   = WEXP + WMAN;
    localparam WMAG    = WEXP + FF;
    // One padding bit below the significand: _zkf_rshift_sticky's output bit 0 OR's the data bit at position 0
    // together with the dropped sticky, and those must stay separate.
    localparam RSH_MAX = WMAN + 1;                      // beyond this everything is sticky
    localparam WRSH    = $clog2(RSH_MAX + 1);
    localparam LSH_MAX = WMAG - WMAN;                   // beyond this the magnitude leaves the container
    localparam WLSH    = $clog2(LSH_MAX + 1);

    // LEFT_SHIFT_BASE is the exponent at which the significand already sits at the binary point, the boundary between
    // right and left shift; it is negative at small WEXP, where every shift is a left one. Below RIGHT_OVER_BASE the
    // right shift exceeds RSH_MAX. At OOR_THRESHOLD and above, which covers inf, |x| >= 2^(WEXP-1).
    localparam integer BIAS            = (1 << (WEXP - 1)) - 1;
    localparam integer LEFT_SHIFT_BASE = BIAS + WFRAC - FF;
    localparam integer RIGHT_OVER_BASE = LEFT_SHIFT_BASE - RSH_MAX;
    localparam integer OOR_THRESHOLD   = BIAS + WEXP - 1;
    // WD sizes the two shift-amount subtractions.
    localparam integer MAX_EXP_IN      = (1 << WEXP) - 1;
    localparam integer ABS_LSB         = (LEFT_SHIFT_BASE >= 0) ? LEFT_SHIFT_BASE : -LEFT_SHIFT_BASE;
    localparam integer MAX_POS_DELTA   = MAX_EXP_IN - LEFT_SHIFT_BASE;
    localparam integer WD              = $clog2(((ABS_LSB > MAX_POS_DELTA) ? ABS_LSB : MAX_POS_DELTA) + 1) + 1;
    localparam signed [WD-1:0] LEFT_SHIFT_BASE_EXT = LEFT_SHIFT_BASE[WD-1:0];

    wire             in_valid_q;
    wire [WFULL-1:0] x_q;
    zkf_pipe #(.W(WFULL), .N(STAGE_INPUT)) u_input_pipe (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in(x),
        .out_valid(in_valid_q), .out(x_q)
    );

    // -- Stage 1: the shift amounts and the predicates. The barrel shifters sit in the next stage so that neither
    // carries both a subtraction and a wide variable shift. The two amounts are separate folded-constant subtractions
    // rather than one negated, which would put both on the same carry chain; the predicates are unsigned comparisons
    // against constants for the same reason.
    wire      [WEXP-1:0] exp_in = x_q[WFULL-2:WFRAC];
    wire signed [WD-1:0] exp_s  = $signed({{(WD-WEXP){1'b0}}, exp_in});
    wire signed [WD-1:0] left_shift_full  = exp_s - LEFT_SHIFT_BASE_EXT;
    wire signed [WD-1:0] right_shift_full = LEFT_SHIFT_BASE_EXT - exp_s;
    wire                 oor_in = exp_in >= OOR_THRESHOLD[WEXP-1:0];
    wire is_left_shift;
    wire right_too_big;
    generate
        if (LEFT_SHIFT_BASE <= 0) begin : g_lshift_always
            assign is_left_shift = 1'b1;
        end else begin : g_lshift_cmp
            assign is_left_shift = exp_in >= LEFT_SHIFT_BASE[WEXP-1:0];
        end
        if (RIGHT_OVER_BASE <= 0) begin : g_rover_never
            assign right_too_big = 1'b0;
        end else begin : g_rover_cmp
            assign right_too_big = exp_in < RIGHT_OVER_BASE[WEXP-1:0];
        end
    endgenerate

    reg             s1_valid;
    reg             s1_sign;
    reg             s1_is_zero;
    reg             s1_is_left_shift;
    reg             s1_oor;
    reg  [WMAN-1:0] s1_sig;
    reg  [WRSH-1:0] s1_rshamt;
    reg  [WLSH-1:0] s1_lshamt;

    // -- Stage 2: the right shift folds its discarded tail into a sticky bit; the left shift is exact.
    wire [WMAN:0] rsh;
    _zkf_rshift_sticky #(.W(WMAN + 1), .WSHIFT(WRSH), .STAGE_SPLIT(0)) u_rshift (
        .clk(clk), .x({s1_sig, 1'b0}), .shamt(s1_rshamt), .y(rsh)
    );
    wire [WMAG-1:0] lsh = {{LSH_MAX{1'b0}}, s1_sig} << s1_lshamt;

    reg            s2_valid;
    reg            s2_sign;
    reg            s2_is_zero;
    reg            s2_oor;
    reg [WMAG-1:0] s2_mag;
    reg            s2_lost_sticky;

    always @(posedge clk) begin
        if (rst) begin
            s1_valid <= 1'b0;
            s2_valid <= 1'b0;
        end else begin
            s1_valid <= in_valid_q;
            s2_valid <= s1_valid;
        end
        s1_sign          <= x_q[WFULL-1];
        s1_is_zero       <= ~|exp_in;
        s1_is_left_shift <= is_left_shift;
        s1_oor           <= oor_in;
        s1_sig           <= {1'b1, x_q[WFRAC-1:0]};
        // An out-of-range magnitude is not shifted, which keeps it inside the container.
        s1_lshamt        <= (is_left_shift && !oor_in) ? left_shift_full[WLSH-1:0] : {WLSH{1'b0}};
        s1_rshamt        <= right_too_big ? RSH_MAX[WRSH-1:0] : right_shift_full[WRSH-1:0];

        s2_sign        <= s1_sign;
        s2_is_zero     <= s1_is_zero;
        s2_oor         <= s1_oor;
        s2_mag         <= s1_is_left_shift ? lsh : {{LSH_MAX{1'b0}}, rsh[WMAN:1]};
        s2_lost_sticky <= ~s1_is_left_shift & rsh[0];
    end

    assign out_valid   = s2_valid;
    assign mag         = s2_mag;
    assign lost_sticky = s2_lost_sticky;
    assign sign        = s2_sign;
    assign is_zero     = s2_is_zero;
    assign oor         = s2_oor;
endmodule

`default_nettype wire
