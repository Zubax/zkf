// Streamed division and square root on one radix-4 digit pipeline, throughput 1:
//
//   op_sqrt = 0:  y = a / b
//   op_sqrt = 1:  y = sqrt(a)
//
// Both are correctly rounded (RNTE) and take the same latency, ceil(WMAN/2) - 1 cycles plus the knobs below, so one
// result port serves them.
//
// A fixed MODE drops the other operation's logic and ignores op_sqrt.
//
// Both recurrences run as w' = 4w - F_d in one fixed-point frame, one digit per register stage. Division: the
// quotient is prenormalized into [1, 2) (the dividend doubled if it is below the divisor), D = den and D3 = 3*den are
// held, and F = {D, 2D, D3}. Square root: D = 2Q and D3 = 2(3Q + u) with u = ulp(Q), and F = {D | u/4, 2D | u,
// D3 | u/4} -- the divider's subtrahends with one bit set where they are zero, so both share the subtractors.
// Stage 0 resolves the root's first digit from the radicand's top bits and, at even WMAN, the divider's by
// three-operand subtracts of the prenormalized dividend; at odd WMAN the divider instead resolves its last two bits
// together with the guard. Neither operation can produce an exact tie and the rounded significand never carries out,
// so the last digit is decided by compares alone and rounding selects the truncated prefix P or P + 1, formed beside
// the last digit stage.
//
// STAGE_INPUT: input registers ahead of the decode; values >1 add dummy stages for routing-congested designs.
// STAGE_DECODE={0,1}: at even WMAN, 1 gives the divider's first digit a register stage of its own instead of resolving
//     it in stage 0, in every MODE (the root then decides its last digit in the added stage); no effect at odd WMAN.
// STAGE_PACK={0,1}: register the last digit decision ahead of the packer.
// STAGE_OUTPUT={0,1}: register the results.
// Each knob that takes effect costs as many cycles as its value.

`default_nettype none

module zkf_divsqrt #(
    parameter WEXP         = 6,
    parameter WMAN         = 18,   // significand precision including the hidden bit
    parameter STAGE_INPUT  = 0,
    parameter STAGE_DECODE = 0,
    parameter STAGE_PACK   = 0,
    parameter STAGE_OUTPUT = 0,
    parameter MODE         = 2,     // 0 -- division, 1 -- sqrt, 2 -- selected at runtime via op_sqrt.
    parameter LATENCY      = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire                 op_sqrt,
    input wire [WEXP+WMAN-1:0] a,
    input wire [WEXP+WMAN-1:0] b,           // ignored during sqrt operation.

    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y,
    output wire                 error   // a / b: b encodes +0 (also for 0/0, where y = +0); sqrt(a): a < 0, y = -inf
);
    localparam WFRAC    = WMAN - 1;
    localparam WFULL    = WEXP + WMAN;
    localparam WEU      = WEXP + 2;
    localparam EVEN     = (WMAN % 2) == 0;
    localparam K        = WMAN / 2;
    localparam FOLD     = EVEN && (STAGE_DECODE == 0);
    localparam NREG     = EVEN ? (K - 1 + STAGE_DECODE) : K;   // register stages of the digit pipeline
    localparam G        = NREG - 1;                            // the last of them
    localparam HAS_DIV  = MODE != 1;
    localparam HAS_ROOT = MODE != 0;
    localparam WFRAME   = (HAS_ROOT && EVEN) ? WMAN : WFRAC;   // fractional bits of the frame
    localparam WI       = HAS_ROOT ? 2 : 1;                    // integer bits of w and D
    localparam WW       = WI + WFRAME;
    localparam WD3      = WW + 2;
    localparam SH       = WFRAME - WFRAC;                      // the divider operands' offset in the frame
    localparam WP       = 2 * NREG + 1;                        // digit prefix after the last stage, integer bit incl.

    localparam LATENCY_REF = NREG + STAGE_INPUT + STAGE_PACK + STAGE_OUTPUT;
    generate
        if ((WEXP < 2) || (WMAN < 4)) begin : g_invalid_format
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        if ((STAGE_DECODE != 0) && (STAGE_DECODE != 1)) begin : g_invalid_stage_decode
            _zkf_invalid_stage_decode u_invalid();
        end
        if ((MODE < 0) || (MODE > 2)) begin : g_invalid_mode
            _zkf_invalid_divsqrt_mode u_invalid();
        end
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    localparam [WEXP-1:0] EXP_INF  = {WEXP{1'b1}};
    localparam [WEXP-1:0] EXP_BIAS = {1'b0, {WEXP-1{1'b1}}};

    wire             in_q_valid;
    wire [2*WFULL:0] in_q;
    zkf_pipe #(.W(2*WFULL + 1), .N(STAGE_INPUT)) u_input_pipe (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in({op_sqrt, b, a}),
        .out_valid(in_q_valid), .out(in_q)
    );
    wire op = (MODE == 2) ? in_q[2*WFULL] : (MODE == 1);

    wire             a_sign = in_q[WFULL-1];
    wire  [WEXP-1:0] a_exp  = in_q[WFULL-2:WFRAC];
    wire [WFRAC-1:0] a_frac = in_q[WFRAC-1:0];
    wire             b_sign = in_q[2*WFULL-1];
    wire  [WEXP-1:0] b_exp  = in_q[2*WFULL-2:WFULL+WFRAC];
    wire [WFRAC-1:0] b_frac = in_q[WFULL+WFRAC-1:WFULL];
    wire             a_zero = a_exp == {WEXP{1'b0}};
    wire             a_inf  = a_exp == EXP_INF;
    wire             b_zero = (b_exp == {WEXP{1'b0}}) && !op;   // b reaches nothing in a root transaction
    wire             b_inf  = b_exp == EXP_INF;
    wire             neg    = a_sign && !a_zero && op;

    // Special operands, per the README's semantics.
    wire s0_force_zero = op ? a_zero : (a_zero || b_inf);
    wire s0_force_inf  = op ? (a_inf || neg) : (!a_zero && !b_inf && (b_zero || a_inf));
    wire s0_sign       = op ? neg : (a_sign ^ (b_sign && !b_zero));

    // The biased exponent, before the divider's normalization correction.
    wire signed [WEU-1:0] e_div  = $signed({2'b00, a_exp}) - $signed({2'b00, b_exp}) + $signed({2'b00, EXP_BIAS});
    wire         [WEXP:0] e_root = {1'b0, a_exp} + {1'b0, EXP_BIAS};   // BIAS is odd: >> 1 floors (exp - BIAS)/2
    wire signed [WEU-1:0] s0_exp = op ? $signed({2'b00, e_root[WEXP:1]}) : e_div;

    // Stage 0 of each operation: the state after its first digit (the divider's after none when unfolded).
    wire  [WW-1:0] dv_w, rt_w;
    wire  [WW-1:0] dv_d, rt_d;
    wire [WD3-1:0] dv_d3, rt_d3;
    wire     [1:0] dv_digit, rt_digit;
    wire           dv_h;
    generate
        if (HAS_DIV) begin : g_div0
            wire [WMAN-1:0] sb = {1'b1, b_frac};
            wire [WMAN-1:0] rem;
            _zkf_divsqrt_div0 #(.WMAN(WMAN), .FOLD(FOLD)) u_div0 (
                .sa({1'b1, a_frac}), .sb(sb), .h(dv_h), .digit(dv_digit), .rem(rem)
            );
            wire [WD3-1:0] fw = {{(WD3-WMAN){1'b0}}, rem} << SH;
            wire [WD3-1:0] fd = {{(WD3-WMAN){1'b0}}, sb} << SH;
            assign dv_w  = fw[WW-1:0];
            assign dv_d  = fd[WW-1:0];
            assign dv_d3 = fd + {fd[WD3-2:0], 1'b0};
        end else begin : g_no_div0
            assign dv_h     = 1'b0;
            assign dv_digit = 2'd0;
            assign dv_w     = {WW{1'b0}};
            assign dv_d     = {WW{1'b0}};
            assign dv_d3    = {WD3{1'b0}};
        end
        if (HAS_ROOT) begin : g_root0
            _zkf_divsqrt_root0 #(.WMAN(WMAN), .WFRAME(WFRAME)) u_root0 (
                .frac(a_frac), .r(!a_exp[0]), .digit(rt_digit), .w(rt_w), .d(rt_d), .d3(rt_d3)
            );
        end else begin : g_no_root0
            assign rt_digit = 2'd0;
            assign rt_w     = {WW{1'b0}};
            assign rt_d     = {WW{1'b0}};
            assign rt_d3    = {WD3{1'b0}};
        end
    endgenerate

    reg                   r_valid [0:G];
    reg                   r_op    [0:G];
    reg          [WW-1:0] r_w     [0:G];
    reg          [WW-1:0] r_d     [0:G];
    reg         [WD3-1:0] r_d3    [0:G];
    reg signed  [WEU-1:0] r_exp   [0:G];
    reg                   r_h;            // the divider's normalization correction to the exponent
    reg                   r_sign  [0:G];
    reg                   r_zero  [0:G];
    reg                   r_inf   [0:G];
    reg                   r_err   [0:G];
    reg          [WP-1:0] r_pp1;          // prefix + 1, formed beside the last digit so that rounding only selects
    wire [NREG*(NREG+2)-1:0] prefix_tri;   // the digit prefix of stage s (3 + 2s bits) at offset s(s+2)
    // At MODE=2, the subtrahend bits the root injects into the consumer of stage s (the next stage, or the folded last
    // decision), at 3s: registered, off the compares' paths. The other builds derive them.
    wire [3*NREG-1:0] inj_pipe;

    // The integer bit and digit 1 (when unfolded, the divider has no digit yet: two leading zeros keep it aligned).
    wire [1:0] digit0  = op ? rt_digit : dv_digit;
    wire [2:0] prefix0 = (op || FOLD) ? {1'b1, digit0} : 3'b001;
    always @(posedge clk) begin
        if (rst) begin
            r_valid[0] <= 1'b0;
        end else begin
            r_valid[0] <= in_q_valid;
        end
        r_op[0]   <= op;
        r_w[0]    <= op ? rt_w : dv_w;
        r_d[0]    <= op ? rt_d : dv_d;
        r_d3[0]   <= op ? rt_d3 : dv_d3;
        r_exp[0]  <= s0_exp;
        r_h       <= dv_h && !op;
        r_sign[0] <= s0_sign;
        r_zero[0] <= s0_force_zero;
        r_inf[0]  <= s0_force_inf;
        r_err[0]  <= b_zero || neg;   // each is gated by its operation
    end

    genvar i_stage;
    generate
        if (HAS_DIV) begin : g_prefix0
            reg [2:0] r_prefix0;
            always @(posedge clk) r_prefix0 <= prefix0;
            assign prefix_tri[2:0] = r_prefix0;
        end else begin : g_no_prefix0
            assign prefix_tri[2:0] = 3'b000;
        end
        if (G == 0) begin : g_pp1_0
            always @(posedge clk) r_pp1 <= prefix0 + 3'd1;
        end
        if (MODE == 2) begin : g_inj0
            reg [2:0] r_inj;
            always @(posedge clk) r_inj <= {3{op}} | {dv_d3[WFRAME-4], dv_d[WFRAME-3], dv_d[WFRAME-4]};
            assign inj_pipe[2:0] = r_inj;
        end else begin : g_no_inj0
            assign inj_pipe[2:0] = 3'b000;
        end

        for (i_stage = 1; i_stage <= G; i_stage = i_stage + 1) begin : g_stage
            localparam S  = i_stage;
            localparam IQ = HAS_ROOT ? (WFRAME - 2 * S - 2) : 0;   // the root's u/4 after S digits
            wire     [1:0] digit;
            wire  [WW-1:0] w_next;
            wire  [WW-1:0] d_next;
            wire [WD3-1:0] d3_next;
            _zkf_divsqrt_step #(.WW(WW), .IQ(IQ), .MODE(MODE)) u_step (
                .sqrt(r_op[S-1]),
                .w(r_w[S-1]), .d(r_d[S-1]), .d3(r_d3[S-1]), .inj(inj_pipe[3*(S-1) +: 3]),
                .digit(digit), .w_next(w_next), .d_next(d_next), .d3_next(d3_next)
            );
            always @(posedge clk) begin
                if (rst) begin
                    r_valid[S] <= 1'b0;
                end else begin
                    r_valid[S] <= r_valid[S-1];
                end
                r_op[S]   <= r_op[S-1];
                r_w[S]    <= w_next;
                r_d[S]    <= d_next;
                r_d3[S]   <= d3_next;
                r_exp[S]  <= (S == 1) ? (r_exp[S-1] - $signed({{(WEU-1){1'b0}}, r_h})) : r_exp[S-1];
                r_sign[S] <= r_sign[S-1];
                r_zero[S] <= r_zero[S-1];
                r_inf[S]  <= r_inf[S-1];
                r_err[S]  <= r_err[S-1];
            end
            wire [2*S:0] pin;   // the prefix this stage extends
            if (HAS_DIV) begin : g_prefix
                reg [2*S+2:0] r_prefix;
                always @(posedge clk) r_prefix <= {prefix_tri[(S-1)*(S+1) +: 2*S+1], digit};
                assign prefix_tri[S*(S+2) +: 2*S+3] = r_prefix;
                assign pin = prefix_tri[(S-1)*(S+1) +: 2*S+1];
            end else begin : g_no_prefix
                assign prefix_tri[S*(S+2) +: 2*S+3] = {(2*S+3){1'b0}};
                assign pin = r_d[S-1][WFRAME+1 -: 2*S+1];   // D = 2Q
            end
            // D and D3 hold where the next consumer injects; the root's injections make the stage's own state moot.
            if ((MODE == 2) && (IQ >= 2)) begin : g_inj
                reg [2:0] r_inj;
                always @(posedge clk) r_inj <= {3{r_op[S-1]}} | {r_d3[S-1][IQ-2], r_d[S-1][IQ-1], r_d[S-1][IQ-2]};
                assign inj_pipe[3*S +: 3] = r_inj;
            end else begin : g_no_inj
                assign inj_pipe[3*S +: 3] = 3'b000;
            end
            if (S == G) begin : g_last
                wire [2*S:0] pin1 = pin + {{(2*S){1'b0}}, 1'b1};
                always @(posedge clk) r_pp1 <= (digit == 2'd3) ? {pin1, 2'b00} : {pin, digit + 2'd1};
            end
        end
    endgenerate

    // 5*den and 7*den for the divider's radix-8 last step, added in the last digit stage beside its subtracts.
    wire [WW+2:0] f_d5, f_d7;
    generate
        if (!EVEN && HAS_DIV) begin : g_d57
            reg [WW+2:0] r_d5, r_d7;
            always @(posedge clk) begin
                r_d5 <= {3'b000, r_d[G-1]} + {1'b0, r_d[G-1], 2'b00};
                r_d7 <= {r_d[G-1], 3'b000} - {3'b000, r_d[G-1]};
            end
            assign f_d5 = r_d5;
            assign f_d7 = r_d7;
        end else begin : g_no_d57
            assign f_d5 = {(WW+3){1'b0}};
            assign f_d7 = {(WW+3){1'b0}};
        end
    endgenerate

    wire [WMAN-1:0] significand;
    _zkf_divsqrt_last #(.WMAN(WMAN), .MODE(MODE), .FOLD(FOLD), .WW(WW), .WP(WP)) u_last (
        .sqrt(r_op[G]), .w(r_w[G]), .d(r_d[G]), .d3(r_d3[G]), .d5(f_d5), .d7(f_d7),
        .inj({inj_pipe[3*G+2], inj_pipe[3*G]}),
        .prefix(prefix_tri[G*(G+2) +: WP]), .prefix1(r_pp1), .significand(significand)
    );
    wire signed [WEU-1:0] f_exp = (G == 0) ? (r_exp[0] - $signed({{(WEU-1){1'b0}}, r_h})) : r_exp[G];

    _zkf_pack #(
        .WEXP(WEXP), .WMAN(WMAN), .EXP_IS_BIASED(1), .ASSUME_NO_OVERFLOW(MODE == 1),
        .STAGE_INPUT(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT)
    ) u_pack (
        .clk(clk), .rst(rst),
        .in_valid(r_valid[G]),
        .sign(r_sign[G]), .force_zero(r_zero[G]), .force_inf(r_inf[G]),
        .exp_unbiased(f_exp), .significand(significand),
        .guard(1'b0), .round(1'b0), .sticky(1'b0),
        .out_valid(out_valid), .y(y)
    );
    _zkf_pack_delay #(.W(1), .STAGE_INPUT(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT)) u_error_delay (
        .clk(clk), .x(r_err[G]), .y(error)
    );
endmodule


// The divider's stage 0: h = sa < sb, the prenormalized dividend n = sa << h, and the remainder n - sb (unfolded)
// or, folded, the first quotient digit -- the largest k in 0..3 with 4n >= (4 + k)sb -- and 4n - (4 + k)sb. Both
// h outcomes are resolved in parallel, so that h, itself a carry chain, only selects.
module _zkf_divsqrt_div0 #(parameter WMAN = 18, parameter FOLD = 1) (
    input  wire [WMAN-1:0] sa,
    input  wire [WMAN-1:0] sb,
    output wire            h,
    output wire      [1:0] digit,
    output wire [WMAN-1:0] rem
);
    wire [WMAN:0] r0 = {1'b0, sa} - {1'b0, sb};
    wire [WMAN:0] r1 = {sa, 1'b0} - {1'b0, sb};
    assign h = r0[WMAN];
    generate
        if (FOLD) begin : g_fold
            localparam WC = WMAN + 4;
            wire [WC-1:0] y1 = {4'b0000, sb};
            wire [WC-1:0] y2 = {3'b000, sb, 1'b0};
            wire [WC-1:0] y4 = {2'b00, sb, 2'b00};
            wire [WC-1:0] y8 = {1'b0, sb, 3'b000};
            wire [2*WMAN-1:0] rems;     // for h at h*WMAN
            wire        [3:0] digits;   // for h at 2h
            genvar i_h;
            for (i_h = 0; i_h < 2; i_h = i_h + 1) begin : g_h
                wire [WC-1:0] x = (i_h == 1) ? {1'b0, sa, 3'b000} : {2'b00, sa, 2'b00};
                wire [WC-1:0] n1, n2, c3;
                _zkf_divsqrt_add3 #(.W(WC), .CI(0)) u_c1 (.p(~x), .q(y4), .r(y1), .s(n1));   // ~(x - 5sb)
                _zkf_divsqrt_add3 #(.W(WC), .CI(0)) u_c2 (.p(~x), .q(y4), .r(y2), .s(n2));   // ~(x - 6sb)
                _zkf_divsqrt_add3 #(.W(WC), .CI(1)) u_c3 (.p(x), .q(y1), .r(~y8), .s(c3));   // x - 7sb
                wire [WMAN:0] r   = (i_h == 1) ? r1 : r0;
                wire          ge1 = n1[WC-1];
                wire          ge2 = n2[WC-1];
                wire          ge3 = !c3[WC-1];
                // As in the digit step: selected by the monotonic compares, not the digit.
                wire [WMAN-1:0] lo = ge1 ? ~n1[WMAN-1:0] : {r[WMAN-3:0], 2'b00};
                wire [WMAN-1:0] hi = ge3 ? c3[WMAN-1:0] : ~n2[WMAN-1:0];
                assign rems[i_h*WMAN +: WMAN] = ge2 ? hi : lo;
                assign digits[2*i_h +: 2]     = {ge2, ge3 || (ge1 && !ge2)};
            end
            assign digit = h ? digits[3:2] : digits[1:0];
            assign rem   = h ? rems[2*WMAN-1:WMAN] : rems[WMAN-1:0];
        end else begin : g_plain
            assign digit = 2'd0;
            assign rem   = h ? r1[WMAN-1:0] : r0[WMAN-1:0];
        end
    endgenerate
endmodule


// The root's stage 0. The radicand m' = (1 + frac) * 2^r is in [1, 4), r = 1 at an even exponent; Q0 = 1, so
// w0 = m' - 1, D = 2, D3 = 8 and u = 1. The first digit counts the thresholds 1.5625, 2.25 and 3.0625 that m' reaches,
// read off the top five fraction bits rather than from the subtracts; 4*w0 - F for F = 2.25, 5 and 8.25, which have
// no bits below u/4, is formed beside it. The state is in a frame of WFRAME fraction bits.
module _zkf_divsqrt_root0 #(parameter WMAN = 18, parameter WFRAME = 18) (
    input  wire     [WMAN-2:0] frac,
    input  wire                r,
    output wire          [1:0] digit,
    output wire   [WFRAME+1:0] w,
    output wire   [WFRAME+1:0] d,
    output wire   [WFRAME+3:0] d3
);
    localparam WFRAC = WMAN - 1;
    wire [WFRAC+4:0] padded = {frac, 5'b00000};
    wire       [4:0] t      = padded[WFRAC+4 -: 5];
    wire             ge1    = r || (t[4] && (t[3:1] != 3'd0));
    wire             ge2    = r && (t[4:2] != 3'd0);
    wire             ge3    = r && t[4] && (t[3:0] != 4'd0);
    assign digit = {ge2, ge3 || (ge1 && !ge2)};

    wire     [WMAN:0] mprime = {2'b01, frac} << r;
    wire     [WMAN:0] m1     = mprime - {2'b01, {WFRAC{1'b0}}};
    wire [WFRAME+1:0] w0;
    generate
        if (WFRAME > WFRAC) begin : g_even
            assign w0 = {m1, 1'b0};
        end else begin : g_odd
            assign w0 = m1;
        end
    endgenerate
    wire [WFRAME+3:0] m4 = {w0, 2'b00};
    wire        [5:0] h  = m4[WFRAME+3:WFRAME-2];   // 4*w0 in units of u/4
    wire        [5:0] h1 = h - 6'd9;
    wire        [5:0] h2 = h - 6'd20;
    wire        [5:0] h3 = h - 6'd33;
    reg         [5:0] h_next;
    reg         [4:0] d3_half;                      // D3 in units of u/2: 13 + 3*digit
    always @* begin
        case (digit)
            2'd0:    begin h_next = h;  d3_half = 5'd13; end
            2'd1:    begin h_next = h1; d3_half = 5'd16; end
            2'd2:    begin h_next = h2; d3_half = 5'd19; end
            default: begin h_next = h3; d3_half = 5'd22; end
        endcase
    end
    assign w  = {h_next[3:0], m4[WFRAME-3:0]};
    assign d  = {1'b1, digit, {(WFRAME-1){1'b0}}};
    assign d3 = {d3_half, {(WFRAME-1){1'b0}}};
endmodule


// The last digit decision of zkf_divsqrt, by compares alone, and the rounding it implies as a select between the
// prefix and prefix + 1 (prefix1). A root-only build takes the prefix from d = 2Q.
//   Even WMAN, folded: digit K of either operation, from 4w >= F1 and 4w >= F3 (the root's u/4 at frame bit 0).
//   Even WMAN, unfolded: digit K of the divider as above; the root has all its digits and rounds half up.
//   Odd WMAN: the divider's last two bits and guard as one radix-8 step, from 8w >= k*den for k = 1, 3, 5, 7; the
//   root's guard is the next digit's 4w >= 2D | u, u at frame bit 0.
module _zkf_divsqrt_last #(
    parameter WMAN = 18,
    parameter MODE = 2,
    parameter FOLD = 1,
    parameter WW   = 20,
    parameter WP   = 17
) (
    input  wire            sqrt,
    input  wire   [WW-1:0] w,
    input  wire   [WW-1:0] d,
    input  wire   [WW+1:0] d3,
    input  wire   [WW+2:0] d5,
    input  wire   [WW+2:0] d7,
    input  wire      [1:0] inj,   // at MODE=2, bit 0 of the folded root's F3 and F1
    input  wire   [WP-1:0] prefix,
    input  wire   [WP-1:0] prefix1,
    output wire [WMAN-1:0] significand
);
    localparam EVEN     = (WMAN % 2) == 0;
    localparam HAS_DIV  = MODE != 1;
    localparam HAS_ROOT = MODE != 0;
    localparam WFRAME   = WW - (HAS_ROOT ? 2 : 1);
    localparam WT       = WW + 2;

    // Compares as subtracts, so every flow maps them onto carry chains.
    function ge(input [WT:0] l, input [WT:0] r);
        reg [WT+1:0] t;
        begin
            t  = {1'b0, l} - {1'b0, r};
            ge = !t[WT+1];
        end
    endfunction

    wire root = HAS_ROOT && sqrt;
    wire [WT:0] m4 = {1'b0, w, 2'b00};
    wire  [WP-1:0] p;
    generate
        if (HAS_DIV) begin : g_prefix
            assign p = prefix;
        end else begin : g_from_d
            assign p = d[WFRAME+1 -: WP];   // D = 2Q
        end
        if (EVEN) begin : g_even
            wire  [1:0] rb  = (MODE == 1) ? 2'b11 : inj;
            wire  [1:0] b0  = (FOLD && HAS_ROOT) ? rb : {d3[0], d[0]};
            wire        ge1 = ge(m4, {3'b000, d[WW-1:1], b0[0]});
            wire        ge3 = ge(m4, {1'b0, d3[WW+1:1], b0[1]});
            wire [WMAN-1:0] sig_step = ge3 ? {prefix1[WMAN-2:0], 1'b0} : {p[WMAN-2:0], ge1};
            if (FOLD) begin : g_folded
                assign significand = sig_step;
            end else begin : g_unfolded
                assign significand = root ? prefix1[WMAN:1] : sig_step;   // round half up: (q + 1) >> 1
            end
        end else begin : g_odd
            wire [WMAN-1:0] sig_div, sig_root;
            if (HAS_DIV) begin : g_div
                wire [WT:0] m8  = {w, 3'b000};
                wire        ge1 = ge(m8, {3'b000, d});
                wire        ge3 = ge(m8, {1'b0, d3});
                wire        ge5 = ge(m8, d5);
                wire        ge7 = ge(m8, d7);
                reg   [1:0] tail;
                always @* begin
                    case ({ge5, ge3, ge1})
                        3'b111:  tail = 2'b11;
                        3'b011:  tail = 2'b10;
                        3'b001:  tail = 2'b01;
                        default: tail = 2'b00;
                    endcase
                end
                assign sig_div = ge7 ? {prefix1[WMAN-3:0], 2'b00} : {p[WMAN-3:0], tail};
            end else begin : g_no_div
                assign sig_div = {WMAN{1'b0}};
            end
            if (HAS_ROOT) begin : g_root
                wire g = ge(m4, {2'b00, d, 1'b1});
                assign sig_root = g ? prefix1[WMAN-1:0] : p[WMAN-1:0];
            end else begin : g_no_root
                assign sig_root = {WMAN{1'b0}};
            end
            assign significand = root ? sig_root : sig_div;
        end
    endgenerate
endmodule


// p + q + r + CI with one carry chain: a carry-save layer, then a single addition.
module _zkf_divsqrt_add3 #(parameter W = 8, parameter CI = 0) (
    input  wire [W-1:0] p,
    input  wire [W-1:0] q,
    input  wire [W-1:0] r,
    output wire [W-1:0] s
);
    wire [W-1:0] sum   = p ^ q ^ r;
    wire [W-1:0] carry = (p & q) | (p & r) | (q & r);
    assign s = sum + {carry[W-2:0], CI != 0};
endmodule

`default_nettype wire
