/// Streamed float square root: y = sqrt(x), correctly rounded (RNTE). Zero-bubble, throughput-1.
/// Behavior:
///
///   sqrt(finite>0) = sqrt(x), round-to-nearest ties-to-even
///   sqrt(+/-0)     = +0                       (sign of zero is ignored)
///   sqrt(+inf)     = +inf
///   sqrt(x<0)      = -inf, domain_error=1     (incl. -inf; log2-style poison value)
///
/// The output sign bit equals domain_error for every input. domain_error is aligned with y/out_valid.
///
/// The root is produced by an unrolled radix-4 restoring recurrence (2 root bits per stage); the first digit is
/// folded into the decode stage because its selection thresholds are constants.
/// The result exponent never overflows and the round-up carry into it is provably unreachable
/// (an all-1 significand always rounds down at even WMAN; at odd WMAN an all-1 raw root forces rem<=raw, guard 0),
/// so _zkf_pack runs with ASSUME_NO_OVERFLOW=1 while the carry path still rides its combined {exp,significand}
/// rounding adder.
///
/// STAGE_INPUT=0: input combinational paths are exposed.
/// STAGE_INPUT=1: inputs are latched, the external module sees registers at the input (one extra cycle).
/// STAGE_INPUT>1: add extra dummy stages; helps in routing-congested designs (+STAGE_INPUT cycles).
///
/// STAGE_PACK=0: pack inputs are combinational (default).
/// STAGE_PACK=1: register pack inputs (forwarded to _zkf_pack.STAGE_INPUT) (+1 cycle).
///
/// STAGE_OUTPUT=0: y and domain_error are combinational (default).
/// STAGE_OUTPUT=1: registered (one extra cycle).

`default_nettype none

module zkf_sqrt #(
    parameter WEXP         = 6,
    parameter WMAN         = 18,   // significand precision including the hidden bit
    parameter STAGE_INPUT  = 0,    // number of input register stages (>=0); +STAGE_INPUT cycles
    parameter STAGE_PACK   = 0,    // 0 = comb pack inputs; 1 = register pack inputs (+1 cycle)
    parameter STAGE_OUTPUT = 0,    // 0 = combinational outputs; 1 = registered outputs (+1 cycle)
    parameter LATENCY      = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] x,

    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y,
    output wire                 domain_error
);
    localparam WFULL         = WEXP + WMAN;
    localparam WEXP_UNBIASED = WEXP + 2;

    localparam LATENCY_REF = 1 + STAGE_INPUT + (WMAN / 2) + STAGE_PACK + STAGE_OUTPUT;
    generate
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    // Optional input register stage(s): latch the operand before any combinational logic (+STAGE_INPUT cycles).
    wire             in_valid_q;
    wire [WFULL-1:0] x_q;
    zkf_pipe #(.W(WFULL), .N(STAGE_INPUT)) u_input_pipe (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in(x), .out_valid(in_valid_q), .out(x_q)
    );

    wire                            core_valid;
    wire                            core_force_zero;
    wire                            core_force_inf;
    wire                            core_domain_error;
    wire signed [WEXP_UNBIASED-1:0] core_exp_biased;
    wire                 [WMAN-1:0] core_significand;
    wire                            core_guard;
    wire                            core_sticky;

    _zkf_sqrt_core #(.WEXP(WEXP), .WMAN(WMAN)) u_core (
        .clk(clk),
        .rst(rst),
        .in_valid(in_valid_q),
        .x(x_q),
        .out_valid(core_valid),
        .force_zero(core_force_zero),
        .force_inf(core_force_inf),
        .domain_error(core_domain_error),
        .exp_biased(core_exp_biased),
        .significand(core_significand),
        .guard(core_guard),
        .sticky(core_sticky),
        .raw(),
        .partial_rem()  // Byproducts are not used in this module.
    );

    // The packer masks the sign to 0 on the forced-zero path, so wiring sign = domain_error keeps +0 canonical.
    // The packer drives the external y/out_valid directly; STAGE_OUTPUT selects registered vs combinational output.
    _zkf_pack #(
        .WEXP(WEXP), .WMAN(WMAN), .EXP_IS_BIASED(1), .ASSUME_NO_OVERFLOW(1),
        .STAGE_INPUT(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT)
    ) u_pack (
        .clk(clk),
        .rst(rst),
        .in_valid(core_valid),
        .sign(core_domain_error),
        .force_zero(core_force_zero),
        .force_inf(core_force_inf),
        .exp_unbiased(core_exp_biased),
        .significand(core_significand),
        .guard(core_guard),
        .round(1'b0),
        .sticky(core_sticky),
        .out_valid(out_valid),
        .y(y)
    );

    // The delay line is a pure free-running datapath (no reset port). See reset policy.
    // STAGE_OUTPUT matches the packer so domain_error stays aligned with y.
    _zkf_pack_delay #(
        .W(1), .STAGE_INPUT(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT)
    ) u_pack_delay (.clk(clk), .x(core_domain_error), .y(domain_error));
endmodule


/// Internal root generator for Zubax Kulibin float square root.
///
/// The root bits are produced by an unrolled radix-4 restoring recurrence. With x = m*2^e and r = e mod 2
/// (= ~exp[0], BIAS odd), the radicand is m' = m*2^r in [1,4) and sqrt(x) = sqrt(m') * 2^floor(e/2).
/// The biased result exponent is (exp + BIAS) >> 1: BIAS is odd, so the shifted-away bit is exactly r and the
/// unsigned shift floors correctly for every field value -- a single adder, no -r correction.
///
/// State after digit j: root prefix Q (1+2j bits, normalized), remainder rem = X - Q^2 in [0, 2Q] where X is the
/// integer value of the radicand bits consumed so far, and M = 3Q+1 maintained incrementally so the d=3 trial
/// subtrahend stays a plain two-operand carry chain. The first digit is resolved inside the decode stage (its
/// thresholds are the constants 9/20/33), so the pipeline is 1 + QFRAC/2 register stages.
///
/// The final truncated root raw carries QFRAC = WFRAC + (WFRAC % 2) fractional bits. At even WMAN the guard bit
/// is raw[0]; at odd WMAN guard = (rem > raw), exact because the root exceeds raw + 1/2 iff rem > raw (strict:
/// rem == raw is reachable and must give guard 0). sticky = (rem != 0) in both parities. Rounding ties are
/// structurally impossible (a midpoint squared is odd-numerator while the radicand is even), so no round bit.
/// The truncated root and the final remainder are exposed as byproducts that are occasionally useful.
/// There is no sign output: the result sign equals domain_error by policy, so a consumer packs with sign
/// wired to domain_error (as the zkf_sqrt top does).
///
/// Register stages: 1 + QFRAC/2 (= 1 + WMAN/2). Inputs are not latched but outputs are. Throughput is one
/// sample per cycle.

module _zkf_sqrt_core #(
    parameter WEXP          = 6,
    parameter WMAN          = 18,                       // significand precision including the hidden bit
    parameter QFRAC         = (WMAN - 1) + ((WMAN - 1) % 2),    // do not alter
    parameter WEXP_UNBIASED = WEXP + 2                          // do not alter
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] x,

    output reg                            out_valid,
    output reg                            force_zero,
    output reg                            force_inf,
    output reg                            domain_error,
    output reg signed [WEXP_UNBIASED-1:0] exp_biased,
    output reg                 [WMAN-1:0] significand,
    output reg                            guard,
    output reg                            sticky,
    output reg                 [QFRAC:0]  raw,
    output reg               [QFRAC+1:0]  partial_rem
);
    generate
        if ((WEXP < 2) || (WMAN < 4)) begin : g_invalid_wman
            _zkf_invalid_wexp_or_wman u_invalid();
        end
    endgenerate

    localparam WFRAC   = WMAN - 1;
    localparam WFULL   = WEXP + WMAN;
    localparam QSTAGES = QFRAC / 2;
    localparam QRAW    = QFRAC + 1;
    localparam WREM    = QFRAC + 2;
    localparam WTAIL   = 4 * QSTAGES;   // fractional radicand bits consumed by the recurrence (= 2*QFRAC)

    // The growing per-stage state lives in triangular chains: register bank k (0-based) holds only the bits known
    // after k+1 digits, so late stages carry no dead low-order tail and early stages no dead high-order prefix.
    //   Q prefix:      width 3+2k at offset k*(k+2), banks 0..QSTAGES-1
    //   remainder:     width 4+2k at offset k*(k+3), banks 0..QSTAGES-1
    //   M = 3Q+1:      width 5+2k at offset k*(k+4), banks 0..QSTAGES-2 (dead after the last digit)
    //   radicand tail: width 4*(QSTAGES-1-k) at offset 2k*(2*QSTAGES-k-1), banks 0..QSTAGES-2
    localparam QTRI_Q = QSTAGES * (QSTAGES + 2);
    localparam QTRI_R = QSTAGES * (QSTAGES + 3);
    localparam QTRI_M = (QSTAGES - 1) * (QSTAGES + 3);
    localparam QTRI_T = 2 * QSTAGES * (QSTAGES - 1);

    localparam [WEXP-1:0] EXP_INF  = {WEXP{1'b1}};
    localparam [WEXP-1:0] EXP_BIAS = {1'b0, {WEXP-1{1'b1}}};

    wire             x_sign = x[WFULL-1];
    wire  [WEXP-1:0] x_exp  = x[WFULL-2:WFRAC];
    wire [WFRAC-1:0] x_frac = x[WFRAC-1:0];

    // Classify into force-zero/force-infinity controls for _zkf_pack. Zero beats sign (the sign of a zero operand
    // is ignored), negative beats infinity (-inf is a domain error). The root core free-runs for all encodings.
    wire x_zero   = x_exp == {WEXP{1'b0}};
    wire x_inf    = x_exp == EXP_INF;
    wire negative = x_sign && !x_zero;

    // Biased result exponent floor((exp-BIAS)/2)+BIAS == (exp+BIAS)>>1: BIAS is odd, so parity(exp+BIAS) = r and
    // the unsigned shift floors away exactly the parity bit -- correct for every finite field value.
    wire [WEXP:0] exp_sum = {1'b0, x_exp} + {1'b0, EXP_BIAS};

    // Radicand m' = m*2^r in [1,4): 2 integer bits v in {1,2,3} plus WFRAC fractional bits, formed width-safely
    // (the top bit is 0 before the shift). The fractional tail is zero-padded to the 4*QSTAGES bits the digit
    // stages consume, so late stages append constant zeros.
    wire              r_odd  = !x_exp[0];                       // r = (exp - BIAS) mod 2
    wire   [WMAN:0]   mprime = {2'b01, x_frac} << r_odd;
    wire       [1:0]  v      = mprime[WMAN -: 2];
    wire [WTAIL-1:0]  tail0  = {mprime[WFRAC-1:0], {(WTAIL-WFRAC){1'b0}}};

    // First radix-4 digit resolved inside the decode stage: with Q0 = 1 (M0 = 4, rem0 = v-1 <= 2*Q0) the step's
    // subtrahends constant-fold to 9/20/33, leaving three shallow 6-bit constant compares with no prior-stage state.
    wire [1:0] d1;
    wire [3:0] rem1_w;
    wire [4:0] m1_w;
    _zkf_sqrt_radix4_step #(.WQ(1)) u_step1 (
        .q(1'b1), .m(3'd4), .rem(v - 2'd1), .n4(tail0[WTAIL-1 -: 4]),
        .digit(d1), .rem_next(rem1_w), .m_next(m1_w)
    );

    // Sideband delay lines aligned with the radix pipeline; keeping them scalar lets tools optimize bits freely.
    reg            r_valid        [0:QSTAGES-1];
    reg            r_force_zero   [0:QSTAGES-1];
    reg            r_force_inf    [0:QSTAGES-1];
    reg            r_domain_error [0:QSTAGES-1];
    reg [WEXP-1:0] r_exp          [0:QSTAGES-1];

    wire [QTRI_Q-1:0] q_tri;
    wire [QTRI_R-1:0] rem_tri;
    wire [QTRI_M-1:0] m_tri;
    wire [QTRI_T-1:0] tail_tri;

    // Stage zero: classification, the exponent add, and the constant-threshold first digit -- three shallow
    // parallel cones.
    reg [2:0]               r_q1;
    reg [3:0]               r_rem1;
    reg [4:0]               r_m1;
    reg [4*(QSTAGES-1)-1:0] r_tail1;
    always @(posedge clk) begin
        if (rst) begin
            r_valid[0] <= 1'b0;
        end else begin
            r_valid[0] <= in_valid;
        end
        r_force_zero[0]   <= x_zero;
        r_force_inf[0]    <= x_inf || negative;
        r_domain_error[0] <= negative;
        r_exp[0]          <= exp_sum[WEXP:1];
        r_q1              <= {1'b1, d1};
        r_rem1            <= rem1_w;
        r_m1              <= m1_w;
        r_tail1           <= tail0[WTAIL-5:0];
    end
    assign q_tri[2:0]                    = r_q1;
    assign rem_tri[3:0]                  = r_rem1;
    assign m_tri[4:0]                    = r_m1;
    assign tail_tri[4*(QSTAGES-1)-1:0]   = r_tail1;

    genvar i_stage;
    generate
        for (i_stage = 2; i_stage <= QSTAGES; i_stage = i_stage + 1) begin : g_stage
            localparam KP    = i_stage - 2;    // bank holding the state after i_stage-1 digits
            localparam K     = i_stage - 1;
            localparam WQ    = 3 + 2 * KP;     // input root prefix width
            localparam QOFFP = KP * (KP + 2);
            localparam ROFFP = KP * (KP + 3);
            localparam MOFFP = KP * (KP + 4);
            localparam TOFFP = 2 * KP * (2 * QSTAGES - KP - 1);
            localparam TWP   = 4 * (QSTAGES - 1 - KP);
            localparam QOFF  = K * (K + 2);
            localparam ROFF  = K * (K + 3);

            wire [WQ-1:0] q_prev   = q_tri[QOFFP +: WQ];
            wire [WQ:0]   rem_prev = rem_tri[ROFFP +: WQ+1];
            wire [WQ+1:0] m_prev   = m_tri[MOFFP +: WQ+2];
            wire [3:0]    n4       = tail_tri[TOFFP+TWP-4 +: 4];

            wire [1:0]    digit;
            wire [WQ+2:0] rem_next;
            wire [WQ+3:0] m_next;
            _zkf_sqrt_radix4_step #(.WQ(WQ)) u_step (
                .q(q_prev), .m(m_prev), .rem(rem_prev), .n4(n4),
                .digit(digit), .rem_next(rem_next), .m_next(m_next)
            );

            // Reset only validity; payload registers intentionally free-run.
            reg [WQ+1:0] q_r;
            reg [WQ+2:0] rem_r;
            always @(posedge clk) begin
                if (rst) begin
                    r_valid[K] <= 1'b0;
                end else begin
                    r_valid[K] <= r_valid[KP];
                end
                r_force_zero[K]   <= r_force_zero[KP];
                r_force_inf[K]    <= r_force_inf[KP];
                r_domain_error[K] <= r_domain_error[KP];
                r_exp[K]          <= r_exp[KP];
                q_r               <= {q_prev, digit};
                rem_r             <= rem_next;
            end
            assign q_tri[QOFF +: WQ+2]   = q_r;
            assign rem_tri[ROFF +: WQ+3] = rem_r;

            // M and the narrowing radicand tail are dead after the last digit.
            if (i_stage < QSTAGES) begin : g_carry
                localparam MOFF = K * (K + 4);
                localparam TOFF = 2 * K * (2 * QSTAGES - K - 1);
                localparam TW   = TWP - 4;
                reg [WQ+3:0] m_r;
                reg [TW-1:0] tail_r;
                always @(posedge clk) begin
                    m_r    <= m_next;
                    tail_r <= tail_tri[TOFFP +: TW];   // drop the 4 bits consumed this stage
                end
                assign m_tri[MOFF +: WQ+4]  = m_r;
                assign tail_tri[TOFF +: TW] = tail_r;
            end
        end
    endgenerate

    localparam KF = QSTAGES - 1;
    wire [QRAW-1:0] final_raw = q_tri[KF*(KF+2) +: QRAW];
    wire [WREM-1:0] final_rem = rem_tri[KF*(KF+3) +: WREM];

    wire            final_guard;
    wire [WMAN-1:0] final_significand;
    generate
        if ((WMAN % 2) == 0) begin : g_grs_even
            // QFRAC = WFRAC+1: the recurrence already produced the bit below the WMAN-bit root.
            assign final_significand = final_raw[QRAW-1:1];
            assign final_guard       = final_raw[0];
        end else begin : g_grs_odd
            // QFRAC = WFRAC: guard = (rem > raw), strict -- borrow of the widened raw - rem subtract; rem == raw
            // is reachable (all-ones raw at r=1) and must yield guard 0.
            wire [WREM:0] guard_diff = {2'b00, final_raw} - {1'b0, final_rem};
            assign final_significand = final_raw;
            assign final_guard       = guard_diff[WREM];
        end
    endgenerate

    // Final output stage closes the root-prefix/sticky/exponent combinational paths at the module boundary.
    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0;
        end else begin
            out_valid <= r_valid[KF];
        end
        force_zero   <= r_force_zero[KF];
        force_inf    <= r_force_inf[KF];
        domain_error <= r_domain_error[KF];
        exp_biased   <= {2'b00, r_exp[KF]};   // nonnegative by construction; zero-extend into the signed pack port
        significand  <= final_significand;
        guard        <= final_guard;
        sticky       <= |final_rem;
        raw          <= final_raw;
        partial_rem  <= final_rem;
    end
endmodule


// Resolve one radix-4 root digit using parallel candidate subtracts, and maintain M = 3Q+1 incrementally.
// Contract (assumed by the caller, proven by the step proof): q normalized (q[WQ-1] set), m = 3q+1,
// 0 <= rem <= 2q. Then rem_next = 16*rem + n4 - digit*(8q + digit) for the maximal digit keeping it
// nonnegative, 0 <= rem_next <= 2*{q, digit}, and m_next = 3*{q, digit} + 1.
module _zkf_sqrt_radix4_step #(parameter WQ = 3) (
    input wire [WQ-1:0] q,
    input wire [WQ+1:0] m,
    input wire [WQ:0]   rem,
    input wire [3:0]    n4,

    output wire [1:0]    digit,
    output wire [WQ+2:0] rem_next,
    output reg  [WQ+3:0] m_next
);
    localparam WT = WQ + 5;   // trial minuend {rem, n4} and widest subtrahend 24q+9
    localparam WD = WQ + 6;   // subtract width incl. the borrow MSB

    // Plain two-operand subtrahends: 8q+1, 16q+4, and 24q+9 = 8m+1 from the maintained M register.
    wire [WT-1:0] minuend = {rem, n4};
    wire [WT-1:0] t1 = {2'b00, q, 3'b001};
    wire [WT-1:0] t2 = {1'b0, q, 4'b0100};
    wire [WT-1:0] t3 = {m, 3'b001};

    wire [WD-1:0] diff1 = {1'b0, minuend} - {1'b0, t1};
    wire [WD-1:0] diff2 = {1'b0, minuend} - {1'b0, t2};
    wire [WD-1:0] diff3 = {1'b0, minuend} - {1'b0, t3};
    wire          ge1   = !diff1[WT];
    wire          ge2   = !diff2[WT];
    wire          ge3   = !diff3[WT];

    assign digit[1] = ge2;
    assign digit[0] = ge3 || (ge1 && !ge2);
    assign rem_next = ge3 ? diff3[WQ+2:0] :
                      ge2 ? diff2[WQ+2:0] :
                      ge1 ? diff1[WQ+2:0] :
                            minuend[WQ+2:0];

    // m_next = 4m + (3*digit - 3): the four candidates computed in parallel and muxed by the resolved digit.
    wire [WQ+3:0] m4 = {m, 2'b00};
    always @(*) begin
        case (digit)
            2'd0:    m_next = m4 - {{(WQ+2){1'b0}}, 2'd3};
            2'd1:    m_next = m4;
            2'd2:    m_next = m4 + {{(WQ+2){1'b0}}, 2'd3};
            default: m_next = m4 + {{(WQ+1){1'b0}}, 3'd6};
        endcase
    end
endmodule

`default_nettype wire
