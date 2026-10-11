// Streamed base-2 logarithm for the Zubax Kulibin float format: y = log2(x).
// Zero-bubble, throughput-1, no backpressure.
// Behavior:
//
//   log2(finite>0) = log2(x), faithfully rounded
//   log2(+inf)     = +inf
//   log2(+0)       = -inf, pole=1
//   log2(x<0)      = -inf, domain_error=1
//
// Algorithm:
//
//  1. With x = m * 2^e, m in [1,2), log2(x) = e + log2(m). The significand is re-centered around 1: from the table
//     segment next to sqrt(2) up, m is halved and e incremented, so the reduced fraction f = m'-1 is exact
//     (Sterbenz), |log2(m')| stays near 1/2 at most, and x -> 1 maps to e = 0, f -> 0 with no cancellation.
//     _zkf_log2_decode forms e, |f| and the position the result will lead at.
//
//  2. The per-WMAN table+polynomial core evaluates the kernel C(f) = log2(1+f)/f; log2(m') = f * C(f).
//
//  3. The final multiply is aligned so that the leading one of |e + log2(m')| lands on a fixed bit of the sum or one
//     below it: |f| is normalized when e = 0; otherwise C is shifted down against the leading one of |e|, dropping
//     bits far below the result's ULP, so the multiplier stays as wide as the fraction.
//
//  4. _zkf_log2_pack rounds and assembles the float from |e| and that product.
//
// A tie rounds away from zero, not to even as in the other operators. The only results that can be a true tie are
// log2(2**k) with |k| wider than the significand, which takes WEXP >= WMAN+2; such a format is refused here rather
// than given a tie rule nobody needs. Whoever needs one may accept this rounding there, or build the tie rule under
// that condition alone. Elsewhere a tie of the computed value is an artifact of the kernel's own error, with both
// neighbors faithful.
//
// STAGE_PRODUCT and its _SQUARE, _ODD and _CHAIN are the stages of the core's products, as in zkf_exp2;
// STAGE_PRODUCT_FINAL is the depth of the final multiply, a _zkf_pmul. WMULTIPLIER optionally hints the native DSP tile
// argument width; see _zkf_pmul.
//
// DEGREE  = (((WMAN+18)/11)-1)
// ODD     = (DEGREE < 3) ? 0 : STAGE_PRODUCT_ODD
// FEED    = (STAGE_PRODUCT_SQUARE > ODD) ? STAGE_PRODUCT_SQUARE : ODD
// CHAIN   = FEED + (DEGREE / 2) * (1 + STAGE_PRODUCT_CHAIN)
// KERNEL  = (1 + STAGE_PRODUCT > CHAIN) ? (1 + STAGE_PRODUCT) : CHAIN
// LATENCY = STAGE_INPUT + 2 + KERNEL + STAGE_ALIGN + STAGE_PRODUCT_FINAL + STAGE_PACK + STAGE_OUTPUT

`default_nettype none

module zkf_log2 #(
    parameter WEXP                 = 6,
    parameter WMAN                 = 18,   // significand precision including the hidden bit
    parameter WMULTIPLIER          = 0,    // see _zkf_pmul
    parameter STAGE_INPUT          = 0,    // number of input register stages (>=0)
    parameter STAGE_PRODUCT        = 0,
    parameter STAGE_PRODUCT_SQUARE = 0,
    parameter STAGE_PRODUCT_ODD    = 0,
    parameter STAGE_PRODUCT_CHAIN  = 0,
    parameter STAGE_ALIGN          = 0,    // 1: registers the kernel in front of its alignment shift
    parameter STAGE_PRODUCT_FINAL  = 0,    // see _zkf_pmul
    parameter STAGE_PACK           = 0,    // 1: registers the rounded candidates in front of their select
    parameter STAGE_OUTPUT         = 0,    // 0: combinational outputs;  1: registered outputs
    parameter LATENCY              = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] x,

    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y,
    output wire                 domain_error,
    output wire                 pole
);
    localparam DEGREE = (((WMAN+18)/11)-1);                  // must match the generator's degree()
    localparam ODD    = (DEGREE < 3) ? 0 : STAGE_PRODUCT_ODD;
    localparam FEED   = (STAGE_PRODUCT_SQUARE > ODD) ? STAGE_PRODUCT_SQUARE : ODD;
    localparam CHAIN  = FEED + (DEGREE / 2) * (1 + STAGE_PRODUCT_CHAIN);
    localparam KERNEL = (1 + STAGE_PRODUCT > CHAIN) ? (1 + STAGE_PRODUCT) : CHAIN;  // core stages after the ROM's
    localparam LATENCY_REF = STAGE_INPUT + 2 + KERNEL + STAGE_ALIGN + STAGE_PRODUCT_FINAL + STAGE_PACK + STAGE_OUTPUT;
    generate
        // The decode and the result assembly use unsized integer shifts on WEXP; WEXP >= 31 would overflow 32-bit
        // integer constants.
        if ((WEXP < 2) || (WMAN < 4) || (WEXP >= 31)) begin : g_invalid_wman
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        if (WEXP >= WMAN + 2) begin : g_invalid_tie
            _zkf_invalid_log2_wexp_admits_ties u_invalid();
        end
        if ((STAGE_ALIGN != 0) && (STAGE_ALIGN != 1)) begin : g_invalid_stage_align
            _zkf_invalid_stage_align u_invalid();
        end
        if ((STAGE_PACK != 0) && (STAGE_PACK != 1)) begin : g_invalid_stage_pack
            _zkf_invalid_stage_pack u_invalid();
        end
        if ((STAGE_OUTPUT != 0) && (STAGE_OUTPUT != 1)) begin : g_invalid_stage_output
            _zkf_invalid_stage_output u_invalid();
        end
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;
    // WCF: fraction bits of the kernel. WG2: bits below the kernel's LSB that the shifted kernel keeps. WGB: bits the
    // final sum keeps below its lower rounding position. The core checks them against the generator's.
    localparam WCF   = WMAN + 12;
    localparam WG2   = 2;
    localparam WGB   = 12;
    localparam WP    = $clog2(WEXP + 1);        // p+1 in [0, WEXP-1], p the predicted leading-one position
    localparam WLZ   = $clog2(WFRAC);
    localparam WCS   = WCF + WG2 + 2;           // the kernel, shifted
    localparam LOW   = WCF + WG2 - 2 - WGB;     // product bits below the final sum

    wire             in_valid_q;
    wire [WFULL-1:0] x_q;
    zkf_pipe #(.W(WFULL), .N(STAGE_INPUT)) u_input_pipe (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in(x),
        .out_valid(in_valid_q), .out(x_q)
    );

    wire        rc;
    wire        k_valid;
    wire [WCF:0] c;
    // Intentional: unsupported in-range WMAN names missing _zkf_log2_m<WMAN>, prompting table generation.
    `define ZKF_LOG2_TABLE(W) end else if (WMAN == W) begin \
        _zkf_log2_m``W #( \
            .STAGES(KERNEL), .WCF(WCF), .WG2(WG2), .WGB(WGB), .WMULTIPLIER(WMULTIPLIER), \
            .STAGE_PRODUCT(STAGE_PRODUCT), .STAGE_PRODUCT_SQUARE(STAGE_PRODUCT_SQUARE), \
            .STAGE_PRODUCT_ODD(STAGE_PRODUCT_ODD), .STAGE_PRODUCT_CHAIN(STAGE_PRODUCT_CHAIN) \
        ) u_core ( \
            .clk(clk), .rst(rst), .in_valid(in_valid_q), .frac(x_q[WFRAC-1:0]), .rc(rc), .out_valid(k_valid), .c(c));
    // verilog_lint: waive-start generate-label  (macro-expanded selector blocks are intentionally unlabeled)
    generate
        if (1'b0) begin  // seed: the macro opens with "end else if", so every table line is uniform
        `ZKF_LOG2_TABLE(4)
        `ZKF_LOG2_TABLE(5)
        `ZKF_LOG2_TABLE(6)
        `ZKF_LOG2_TABLE(7)
        `ZKF_LOG2_TABLE(8)
        `ZKF_LOG2_TABLE(9)
        `ZKF_LOG2_TABLE(10)
        `ZKF_LOG2_TABLE(11)
        `ZKF_LOG2_TABLE(12)
        `ZKF_LOG2_TABLE(13)
        `ZKF_LOG2_TABLE(14)
        `ZKF_LOG2_TABLE(15)
        `ZKF_LOG2_TABLE(16)
        `ZKF_LOG2_TABLE(17)
        `ZKF_LOG2_TABLE(18)
        `ZKF_LOG2_TABLE(19)
        `ZKF_LOG2_TABLE(20)
        `ZKF_LOG2_TABLE(21)
        `ZKF_LOG2_TABLE(22)
        `ZKF_LOG2_TABLE(23)
        `ZKF_LOG2_TABLE(24)
        `ZKF_LOG2_TABLE(25)
        `ZKF_LOG2_TABLE(26)
        `ZKF_LOG2_TABLE(27)
        `ZKF_LOG2_TABLE(28)
        `ZKF_LOG2_TABLE(29)
        `ZKF_LOG2_TABLE(30)
        `ZKF_LOG2_TABLE(31)
        `ZKF_LOG2_TABLE(32)
        `ZKF_LOG2_TABLE(33)
        `ZKF_LOG2_TABLE(34)
        `ZKF_LOG2_TABLE(35)
        `ZKF_LOG2_TABLE(36)
        `ZKF_LOG2_TABLE(37)
        `ZKF_LOG2_TABLE(38)
        `ZKF_LOG2_TABLE(39)
        `ZKF_LOG2_TABLE(40)
        `ZKF_LOG2_TABLE(41)
        `ZKF_LOG2_TABLE(42)
        `ZKF_LOG2_TABLE(43)
        `ZKF_LOG2_TABLE(44)
        `ZKF_LOG2_TABLE(45)
        `ZKF_LOG2_TABLE(46)
        `ZKF_LOG2_TABLE(47)
        `ZKF_LOG2_TABLE(48)
        `ZKF_LOG2_TABLE(49)
        `ZKF_LOG2_TABLE(50)
        `ZKF_LOG2_TABLE(51)
        `ZKF_LOG2_TABLE(52)
        `ZKF_LOG2_TABLE(53)
        end else begin
            _zkf_invalid_unsupported_table_wman u_invalid();
        end
    endgenerate
    `undef ZKF_LOG2_TABLE
    // verilog_lint: waive-stop generate-label

    // -- Side path, as deep as the core: 1+KERNEL stages. First the raw fields, held beside the ROM's register so that
    // only rc has logic in front of it; then _zkf_log2_decode, over as many of its own stages as the core leaves it.
    localparam SD = (KERNEL >= 3) ? 1 : 0;    // the decode's own stage
    localparam S1 = (KERNEL >= 2) ? 1 : 0;
    wire             s_valid, d_valid, m_valid;
    wire             a_rc, a_sign;
    wire [WEXP-1:0]  a_exp;
    wire [WFRAC-1:0] a_frac;
    zkf_pipe #(.W(WFULL + 1), .N(KERNEL - SD - S1)) u_side0 (
        .clk(clk), .rst(rst), .in_valid(in_valid_q), .in({rc, x_q}),
        .out_valid(s_valid), .out({a_rc, a_sign, a_exp, a_frac})
    );
    // d = {|f|, sh, p1, |e|, ezero, rc, sub, neg, one, is_inf, domain_error, pole}
    localparam MZ = 7, MR = 6, MB = 5, MN = 4, M1 = 3, MI = 2;
    localparam ME = 8;                           // offsets of |e|, p1, sh and |f|
    localparam MP = ME + WEXP;
    localparam MS = MP + WP;
    localparam WM = MS + WP;
    wire [WFRAC+WM-1:0] d;
    _zkf_log2_decode #(.WEXP(WEXP), .WMAN(WMAN), .STAGE(SD)) u_decode (
        .clk(clk), .rst(rst), .in_valid(s_valid), .rc(a_rc), .sign(a_sign), .exp(a_exp), .frac(a_frac),
        .out_valid(d_valid),
        .fmag(d[WM +: WFRAC]), .sh(d[MS +: WP]), .p1(d[MP +: WP]), .emag(d[ME +: WEXP]),
        .ezero(d[MZ]), .rc_q(d[MR]), .sub(d[MB]), .neg(d[MN]), .one(d[M1]), .is_inf(d[MI]),
        .domain_error(d[1]), .pole(d[0])
    );
    wire [WFRAC+WM-1:0] m1;
    zkf_pipe #(.W(WFRAC + WM), .N(S1)) u_side1 (
        .clk(clk), .rst(rst), .in_valid(d_valid), .in(d), .out_valid(m_valid), .out(m1)
    );
    // |f| is normalized when e = 0, split at u_norm's register, the last of the side path; |e| is shifted so that the
    // result's predicted leading one lands on bit WEXP-2.
    wire [WLZ-1:0]      n_lz;
    wire [WFRAC-1:0]    f_norm;
    wire [WFRAC+WM-1:0] n_sb;
    _zkf_normshift #(.W(WFRAC), .STAGE_SPLIT(1), .WSB(WFRAC + WM)) u_norm (
        .clk(clk), .rst(rst), .in_valid(m_valid), .x(m1[WM +: WFRAC]),
        .sb_in({m1[WFRAC+WM-1:MP], m1[ME +: WEXP] << (WEXP - 1 - m1[MP +: WP]), m1[ME-1:0]}),
        .out_valid(), .sb_out(n_sb), .zero(), .count(n_lz), .y(f_norm)
    );
    wire             a_valid;
    wire [WCF:0]      c_q;
    wire [WFRAC-1:0] b_op;
    wire [WLZ-1:0]   b_lz;
    wire [WM-1:0]    m2;
    zkf_pipe #(.W(WCF + 1 + WFRAC + WLZ + WM), .N(STAGE_ALIGN)) u_align (
        .clk(clk), .rst(rst), .in_valid(k_valid),
        .in({c, n_sb[MZ] ? f_norm : n_sb[WM +: WFRAC], n_lz, n_sb[WM-1:0]}),
        .out_valid(a_valid), .out({c_q, b_op, b_lz, m2})
    );
    wire                 f_valid;
    wire [WFRAC+WCS-1:0] prod;
    _zkf_pmul #(
        .WA(WFRAC), .WB(WCS), .SIGNED(0), .WSB(1),
        .WMULTIPLIER(WMULTIPLIER), .STAGE_PRODUCT(STAGE_PRODUCT_FINAL)
    ) u_final (
        .clk(clk), .rst(rst), .in_valid(a_valid), .sb_in(1'b0),
        .a(b_op), .b({c_q, {(WG2+1){1'b0}}} >> m2[MS +: WP]),
        .out_valid(f_valid), .sb_out(), .p(prod)
    );
    wire             p_valid;
    wire [WFULL-1:0] y_c;
    wire [1:0]       flags_c;
    _zkf_log2_pack #(
        .WEXP(WEXP), .WMAN(WMAN), .WGB(WGB), .STAGE(1 + STAGE_PRODUCT_FINAL), .STAGE_SELECT(STAGE_PACK), .WSB(2)
    ) u_pack (
        .clk(clk), .rst(rst), .lz(b_lz), .p1(m2[MP +: WP]), .eal(m2[ME +: WEXP]),
        .ezero(m2[MZ]), .rc(m2[MR]), .sub(m2[MB]), .neg(m2[MN]), .one(m2[M1]), .is_inf(m2[MI]), .sb_in(m2[1:0]),
        .in_valid(f_valid), .prod(prod[WFRAC+WCS-1:LOW]), .out_valid(p_valid), .y(y_c), .sb_out(flags_c)
    );
    zkf_pipe #(.W(WFULL + 2), .N(STAGE_OUTPUT)) u_output_pipe (
        .clk(clk), .rst(rst), .in_valid(p_valid), .in({flags_c, y_c}),
        .out_valid(out_valid), .out({domain_error, pole, y})
    );
endmodule


// Decode of zkf_log2 from the fields of x and rc, whether its significand is re-centered (m' = m/2, e + 1):
// x = m' * 2**e, f = m' - 1, and where the result |e + log2(m')| = |e| +- l will lead.
//
// |f| is frac, or 2**WFRAC - frac when re-centered (then one bit lower); rc implies a nonzero frac.
// With n the leading-one index of |e|, the result leads at n or n-1; |e| = 1 minus a nonzero l leads at -1 or -2, and
// |e| = 2**(WEXP-1) (always minus l) at WEXP-2. p is the upper of each pair, p1 = p + 1; the kernel is aligned to it
// by a right shift of sh = rc + p + 1, or 1 when e = 0, where |f| gets normalized instead.
module _zkf_log2_decode #(
    parameter WEXP  = 6,
    parameter WMAN  = 18,
    parameter STAGE = 0     // 1: a register between e, |f| and what is derived from them
) (
    input  wire                      clk,
    input  wire                      rst,
    input  wire                      in_valid,
    input  wire                      rc,
    input  wire                      sign,
    input  wire           [WEXP-1:0] exp,
    input  wire           [WMAN-2:0] frac,

    output wire                      out_valid,
    output wire           [WMAN-2:0] fmag,
    output wire [$clog2(WEXP+1)-1:0] sh,
    output reg  [$clog2(WEXP+1)-1:0] p1,
    output wire           [WEXP-1:0] emag,
    output wire                      ezero,
    output wire                      rc_q,          // rc, beside the outputs
    output wire                      sub,           // e and f differ in sign: the result is |e| - l
    output reg                       neg,           // sign of the result
    output wire                      one,           // x = 1: +0
    output wire                      is_inf,        // the result is an infinity: x is zero, negative or +inf
    output wire                      domain_error,
    output wire                      pole
);
    localparam WE = WEXP + 1;                // |e| <= 2**(WEXP-1)
    localparam WP = $clog2(WEXP + 1);
    localparam integer BIAS = (1 << (WEXP - 1)) - 1;

    wire   [WE-1:0] bias = rc ? BIAS[WE-1:0] - 1'b1 : BIAS[WE-1:0];
    wire   [WE-1:0] e_p  = {1'b0, exp} - bias;
    wire [WEXP-1:0] e_n  = bias[WEXP-1:0] - exp;   // |e| of a negative e

    wire e_neg, f_zero, x_inf, x_sign;
    zkf_pipe #(.W(WMAN - 1 + WEXP + 7), .N(STAGE)) u_split (
        .clk(clk), .rst(rst), .in_valid(in_valid),
        .in({
            rc ? (~frac + 1'b1) : frac, e_p[WE-1] ? e_n : e_p[WEXP-1:0], e_p[WE-1], ~|e_p, ~|frac, ~|exp,
            &exp, sign, rc
        }),
        .out_valid(out_valid), .out({fmag, emag, e_neg, ezero, f_zero, pole, x_inf, x_sign, rc_q})
    );

    reg [WP-1:0] msb1;                       // 1 + leading-one index of |e|
    integer i;
    always @* begin
        msb1 = {WP{1'b0}};
        for (i = 0; i < WEXP - 1; i = i + 1) if (emag[i]) msb1 = i + 1;
    end
    always @* begin
        case (1'b1)
            emag[WEXP-1]:                p1 = WEXP - 1;
            sub & (emag == 1) & ~f_zero: p1 = {WP{1'b0}};
            default:                     p1 = msb1;
        endcase
        case (1'b1)
            is_inf:  neg = pole | x_sign;
            ezero:   neg = rc_q;
            default: neg = e_neg;
        endcase
    end

    assign sh           = ezero ? {{(WP-1){1'b0}}, 1'b1} : (p1 + rc_q);
    assign sub          = ~ezero & (e_neg ^ rc_q);
    assign is_inf       = pole | x_inf | x_sign;
    assign one          = ezero & f_zero;
    assign domain_error = x_sign & ~pole;
endmodule


// Result assembly of zkf_log2: the float (-1)**neg * (|e| +- l), from |e| aligned to the result's predicted
// leading-one position and the product l = prod; equivalent to _zkf_pack of that sum, a tie rounding away from zero.
//
// The sum |e| +- prod has WS = WFRAC+WGB+4 bits and leads at bit WS-2 ("hi") or one below ("lo"), or is exactly
// 2**(WS-1); its caller guarantees that frame. Two sums, each with its own half ULP, cover both positions: the lower
// one's top bits select, and a carry out of the upper one gives the next exponent with a zero fraction. Each
// candidate's record {sign, fraction kept, exponent field}, with the zero / MIN_NORMAL / normal classes and the
// special results folded in, is formed from the side inputs, which arrive STAGE register stages ahead of prod. A sum
// whose upper candidate is MIN_NORMAL takes no half ULP: there the sum itself decides the class.
module _zkf_log2_pack #(
    parameter WEXP  = 6,
    parameter WMAN  = 18,
    parameter WGB          = 12,   // bits of the sum below the lower rounding position
    parameter STAGE        = 1,    // register stages between the side inputs and prod
    parameter STAGE_SELECT = 0,    // register stages between the rounded candidates and their select
    parameter WSB          = 1     // sideband carried beside the side inputs
) (
    input  wire clk,
    input  wire rst,

    input  wire [$clog2(WMAN-1)-1:0] lz,     // e = 0: hi's exponent is -rc - lz
    input  wire [$clog2(WEXP+1)-1:0] p1,     // e != 0: hi's exponent is p1 - 1
    input  wire           [WEXP-1:0] eal,    // |e|, the result's predicted leading one on bit WEXP-2; zero when e = 0
    input  wire                      ezero,
    input  wire                      rc,
    input  wire                      sub,    // the sum is |e| - prod
    input  wire                      neg,    // sign of the result
    input  wire                      one,    // x = 1: +0
    input  wire                      is_inf, // the result is an infinity; overrides the rest
    input  wire            [WSB-1:0] sb_in,

    input  wire                    in_valid,   // with prod
    input  wire     [WMAN+WGB+2:0] prod,
    output wire                    out_valid,
    output wire    [WEXP+WMAN-1:0] y,
    output wire          [WSB-1:0] sb_out
);
    localparam WFRAC = WMAN - 1;
    localparam WLZ   = $clog2(WFRAC);
    localparam WP    = $clog2(WEXP + 1);
    localparam WS    = WFRAC + WGB + 4;
    localparam WEB   = ((WLZ > WEXP) ? WLZ : WEXP) + 3;  // signed biased exponent of a candidate
    localparam WC    = WEXP + 2;
    localparam integer BIAS = (1 << (WEXP - 1)) - 1;

    wire signed [WEB-1:0] off = ezero ? ~$signed({{(WEB-WLZ){1'b0}}, lz}) : $signed({{(WEB-WP){1'b0}}, p1});
    wire [3*WC-1:0] rec_c;
    wire [2:0]      mn_c;
    genvar gk;
    generate
        for (gk = 0; gk < 3; gk = gk + 1) begin : g_cand   // 0: lo, 1: hi, 2: carry
            reg signed [WEB-1:0] base;
            always @* begin
                case ({ezero, rc})
                    2'b10:   base = BIAS + gk;
                    2'b11:   base = BIAS + gk - 1;
                    default: base = BIAS + gk - 2;
                endcase
            end
            wire signed [WEB-1:0] be = base + off;
            reg [WC-1:0] rec;
            always @* begin
                case (1'b1)
                    is_inf:          rec = {neg, 1'b0, {WEXP{1'b1}}};
                    one | be[WEB-1]: rec = {WC{1'b0}};                           // +0
                    ~|be:            rec = {neg, 1'b0, {(WEXP-1){1'b0}}, 1'b1};  // MIN_NORMAL
                    default:         rec = {neg, 1'b1, be[WEXP-1:0]};
                endcase
            end
            assign rec_c[gk*WC +: WC] = rec;
            assign mn_c[gk] = ~|be;
        end
    endgenerate
    wire [WC-1:0]   r_lo, r_hi, r_cy;
    wire [1:0]      exact;   // the upper candidate of the hi, lo sum is MIN_NORMAL
    wire [WEXP-1:0] eal_q;
    wire            sub_q;
    wire [WSB-1:0]  sb_q;
    zkf_pipe #(.W(3*WC + 2 + WEXP + 1 + WSB), .N(STAGE)) u_side (
        .clk(clk), .rst(1'b0), .in_valid(1'b0), .in({rec_c, mn_c[2:1], eal, sub, sb_in}),
        .out_valid(), .out({r_cy, r_hi, r_lo, exact, eal_q, sub_q, sb_q})
    );

    wire [WS-1:0] e_al = {eal_q, {(WS-WEXP){1'b0}}};
    wire [WS-1:0] h_lo = {{(WS-1){1'b0}}, ~exact[0]} << WGB;
    wire [WS-1:0] h_hi = {{(WS-1){1'b0}}, ~exact[1]} << (WGB + 1);
    wire [WS-1:0] s_lo = sub_q ? (e_al + h_lo - prod) : (e_al + h_lo + prod);
    wire [WS-1:0] s_hi = sub_q ? (e_al + h_hi - prod) : (e_al + h_hi + prod);
    wire             carry, is_hi;
    wire [WFRAC-1:0] f_hi, f_lo;
    wire [WC-1:0]    q_lo, q_hi, q_cy;
    zkf_pipe #(.W(2 + 2*WFRAC + 3*WC + WSB), .N(STAGE_SELECT)) u_select (
        .clk(clk), .rst(rst), .in_valid(in_valid),
        .in({
            s_hi[WS-1], s_lo[WS-1] | s_lo[WS-2],
            s_hi[WGB+2 +: WFRAC], s_lo[WGB+1 +: WFRAC],
            r_cy, r_hi, r_lo, sb_q
        }),
        .out_valid(out_valid), .out({carry, is_hi, f_hi, f_lo, q_cy, q_hi, q_lo, sb_out})
    );
    wire          keep_hi = is_hi & q_hi[WEXP];   // after a carry these fraction bits are zero by themselves
    wire          keep_lo = ~is_hi & q_lo[WEXP];
    reg  [WEXP:0] sign_exp;
    always @* begin
        case (1'b1)
            carry:   sign_exp = {q_cy[WEXP+1], q_cy[WEXP-1:0]};
            is_hi:   sign_exp = {q_hi[WEXP+1], q_hi[WEXP-1:0]};
            default: sign_exp = {q_lo[WEXP+1], q_lo[WEXP-1:0]};
        endcase
    end
    assign y = {sign_exp, (f_hi & {WFRAC{keep_hi}}) | (f_lo & {WFRAC{keep_lo}})};
endmodule

`default_nettype wire
