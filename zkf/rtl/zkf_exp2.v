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
//  1. _zkf_exp2_reduce splits x = i + f, i = floor(x), f in [0,1) at WCF fraction bits, to within an LSB that the
//     table certificate accounts for.
//
//  2. 2**f in [1,2) comes from the per-WMAN table+polynomial core with half an ULP already added, so its truncation
//     is the rounded significand. Rounding up to 2 selects the exponent i+1.
//
//  3. _zkf_exp2_pack assembles the float: both exponents are classified beside the core's last sum, whose top bit
//     selects between them.
//
// The half ULP rounds a tie of the computed value up, not to even as the other operators do. That costs no accuracy:
// 2**x is irrational unless x is an integer, and then it is a power of two, so no normal result lies midway between
// two floats (2**-BIAS, midway between zero and MIN_NORMAL, follows the class rule of _zkf_exp2_pack); a tie of the
// computed value is an artifact of the kernel's own error, with both neighbors faithful.
//
// The core evaluates a polynomial of the table's DEGREE in the low argument bits w, as c0 + c1*w beside the chain
// w**2 * (c2 + c3*w + w**2 * (c4 + c5*w)). Each kind of product is a _zkf_pmul with a stage knob of its own:
//
//   STAGE_PRODUCT={0..4}         c1*w, the widest.
//   STAGE_PRODUCT_SQUARE={0..5}  w**2.
//   STAGE_PRODUCT_ODD={0..4}     c3*w and c5*w. No effect below DEGREE 3, which has neither.
//   STAGE_PRODUCT_CHAIN={0..4}   each product of the chain by w**2; DEGREE/2 of them follow one another, a register
//                                after each.
//
// The square and the odd products feed the chain, and c1*w has as long as the chain takes. The square's operands come
// straight from a register, so its first stage is the register behind the multiplier and only the rest are the
// multiplier's own; the other knobs are the multiplier's throughout. A knob is the widest product's of its kind: a
// narrower one takes the stages that _zkf_pmul admits at its width and waits out the rest. WMULTIPLIER optionally
// hints the native DSP tile argument width; see _zkf_pmul.
//
// DEGREE  = (((WMAN+18)/11)-1)
// ODD     = (DEGREE < 3) ? 0 : STAGE_PRODUCT_ODD
// FEED    = (STAGE_PRODUCT_SQUARE > ODD) ? STAGE_PRODUCT_SQUARE : ODD
// CHAIN   = FEED + (DEGREE / 2) * (1 + STAGE_PRODUCT_CHAIN)
// KERNEL  = (1 + STAGE_PRODUCT > CHAIN) ? (1 + STAGE_PRODUCT) : CHAIN
// LATENCY = STAGE_INPUT + STAGE_REDUCE + 1 + KERNEL + STAGE_PACK + STAGE_OUTPUT

`default_nettype none

module zkf_exp2 #(
    parameter WEXP                 = 6,
    parameter WMAN                 = 18,   // significand precision including the hidden bit
    parameter WMULTIPLIER          = 0,    // see _zkf_pmul
    parameter STAGE_INPUT          = 0,    // number of input register stages (>=0)
    parameter STAGE_REDUCE         = 0,    // 1: registers the reduced argument in front of the table
    parameter STAGE_PRODUCT        = 0,
    parameter STAGE_PRODUCT_SQUARE = 0,
    parameter STAGE_PRODUCT_ODD    = 0,
    parameter STAGE_PRODUCT_CHAIN  = 0,
    parameter STAGE_PACK           = 0,    // 1: registers the core's sum in front of the result assembly
    parameter STAGE_OUTPUT         = 0,    // 0: combinational outputs;  1: registered outputs
    parameter LATENCY              = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] x,

    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    localparam DEGREE = (((WMAN+18)/11)-1);                  // must match the generator's degree()
    localparam ODD    = (DEGREE < 3) ? 0 : STAGE_PRODUCT_ODD;
    localparam FEED   = (STAGE_PRODUCT_SQUARE > ODD) ? STAGE_PRODUCT_SQUARE : ODD;
    localparam CHAIN  = FEED + (DEGREE / 2) * (1 + STAGE_PRODUCT_CHAIN);
    localparam KERNEL = (1 + STAGE_PRODUCT > CHAIN) ? (1 + STAGE_PRODUCT) : CHAIN;  // core stages after the ROM's
    localparam LATENCY_REF = STAGE_INPUT + STAGE_REDUCE + 1 + KERNEL + STAGE_PACK + STAGE_OUTPUT;
    generate
        if ((WEXP < 2) || (WMAN < 4)) begin : g_invalid_wman
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        // The reduction and the result assembly use unsized integer shifts on WEXP; WEXP >= 31 would overflow
        // Verilog's 32-bit integer constant arithmetic.
        if (WEXP >= 31) begin : g_invalid_wexp_too_wide
            _zkf_invalid_exp2_wexp_too_wide_unportable u_invalid();
        end
        if ((STAGE_REDUCE != 0) && (STAGE_REDUCE != 1)) begin : g_invalid_stage_reduce
            _zkf_invalid_stage_reduce u_invalid();
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

    localparam WFULL = WEXP + WMAN;
    // WCF: fraction bits of the reduced argument and of the core's result; the core checks it against its own.
    localparam WCF   = WMAN + 12;

    wire             in_valid_q;
    wire [WFULL-1:0] x_q;
    zkf_pipe #(.W(WFULL), .N(STAGE_INPUT)) u_input_pipe (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in(x),
        .out_valid(in_valid_q), .out(x_q)
    );

    localparam WSB = WEXP + 2;   // {sign, zero, out of range, integer part}
    wire [WSB-1:0] r_sb;
    wire [WCF-1:0] r_f;
    _zkf_exp2_reduce #(.WEXP(WEXP), .WMAN(WMAN), .WFF(WCF)) u_reduce (
        .x(x_q), .sign(r_sb[WSB-1]), .zero(r_sb[WSB-2]), .oor(r_sb[WSB-3]), .ipart(r_sb[WEXP-2:0]), .f(r_f)
    );
    wire           f_valid;
    wire [WSB-1:0] f_sb;
    wire [WCF-1:0] f;
    zkf_pipe #(.W(WSB + WCF), .N(STAGE_REDUCE)) u_reduce_pipe (
        .clk(clk), .rst(rst), .in_valid(in_valid_q), .in({r_sb, r_f}), .out_valid(f_valid), .out({f_sb, f})
    );

    wire             c_valid;
    wire [WSB-1:0]   c_sb;
    wire [WCF+1:0]   s;
    // Intentional: unsupported in-range WMAN names missing _zkf_exp2_m<WMAN>, prompting table generation.
    `define ZKF_EXP2_TABLE(W) end else if (WMAN == W) begin \
        _zkf_exp2_m``W #( \
            .STAGES(KERNEL), .WCF(WCF), .WMULTIPLIER(WMULTIPLIER), .STAGE_PRODUCT(STAGE_PRODUCT), \
            .STAGE_PRODUCT_SQUARE(STAGE_PRODUCT_SQUARE), .STAGE_PRODUCT_ODD(STAGE_PRODUCT_ODD), \
            .STAGE_PRODUCT_CHAIN(STAGE_PRODUCT_CHAIN) \
        ) u_core ( \
            .clk(clk), .rst(rst), .in_valid(f_valid), .f(f), .out_valid(c_valid), .s(s));
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

    zkf_pipe #(.W(WSB), .N(1 + KERNEL)) u_side (
        .clk(clk), .rst(1'b0), .in_valid(1'b0), .in(f_sb), .out_valid(), .out(c_sb)
    );

    wire             p_valid;
    wire [WSB-1:0]   p_sb;
    wire [WCF+1:0]   p_s;
    zkf_pipe #(.W(WSB + WCF + 2), .N(STAGE_PACK)) u_pack_pipe (
        .clk(clk), .rst(rst), .in_valid(c_valid), .in({c_sb, s}), .out_valid(p_valid), .out({p_sb, p_s})
    );
    wire [WFULL-1:0] y_c;
    _zkf_exp2_pack #(.WEXP(WEXP), .WMAN(WMAN), .WCF(WCF)) u_pack (
        .sign(p_sb[WSB-1]), .zero(p_sb[WSB-2]), .oor(p_sb[WSB-3]), .ipart(p_sb[WEXP-2:0]), .s(p_s), .y(y_c)
    );
    zkf_pipe #(.W(WFULL), .N(STAGE_OUTPUT)) u_output_pipe (
        .clk(clk), .rst(rst), .in_valid(p_valid), .in(y_c), .out_valid(out_valid), .out(y)
    );
endmodule


// Range reduction of zkf_exp2: x = i + f with i = floor(x) and f in [0, 1) at WFF fraction bits.
// mag = floor(|x| * 2**WFF) = {1, frac, 0..} >> (ETOP - exp) holds floor(|x|) above its WFF fraction bits. A negative x
// takes the ones' complement of that fraction, which is floor(x * 2**WFF) or one below it, and i = -ipart-1: a
// negative integer becomes the integer below it and the core's last argument.
//
// The shift amount is offset by PAD so that its low WFINE bits are the complement of the exponent's: the fine shift
// starts with no carry chain in front of it, and only the coarse part waits for a short subtraction.
//
// ipart and f are meaningless when zero or oor is set.
module _zkf_exp2_reduce #(
    parameter WEXP = 6,
    parameter WMAN = 18,
    parameter WFF  = 30
) (
    input  wire [WEXP+WMAN-1:0] x,
    output wire                 sign,
    output wire                 zero,
    output wire                 oor,     // |x| >= 2**(WEXP-1)
    output wire      [WEXP-2:0] ipart,   // floor(|x|)
    output wire       [WFF-1:0] f
);
    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;
    localparam WMAG  = WEXP - 1 + WFF;  // |x| < 2**(WEXP-1) in range
    localparam integer BIAS = (1 << (WEXP - 1)) - 1;
    localparam integer ETOP = BIAS + WEXP - 2;  // largest in-range exponent: the significand sits at the top of mag
    localparam WFINE = (WEXP < 3) ? WEXP : 3;
    localparam integer PAD   = ((1 << WFINE) - ((ETOP + 1) % (1 << WFINE))) % (1 << WFINE);
    localparam integer CTOP  = (ETOP + PAD) >> WFINE;
    localparam integer NCRS  = ((WMAG + PAD - 1) >> WFINE) + 1;      // coarse amounts that leave bits
    localparam WCRS  = $clog2(NCRS + 1);
    localparam WHI   = (WEXP > WFINE) ? (WEXP - WFINE) : 1;
    localparam WCO   = (WHI + 1 > WCRS) ? (WHI + 1) : WCRS;

    wire     [WEXP-1:0] exp    = x[WFULL-2:WFRAC];
    wire      [WHI-1:0] exp_hi = (WEXP > WFINE) ? exp[WEXP-1 -: WHI] : 1'b0;
    wire      [WCO-1:0] coarse = CTOP[WCO-1:0] - {{(WCO-WHI){1'b0}}, exp_hi};   // non-negative in range
    wire [WMAG+PAD-1:0] fine   = {1'b1, x[WFRAC-1:0], {(WMAG-WMAN+PAD){1'b0}}} >> ~exp[WFINE-1:0];
    wire [WMAG+PAD-1:0] mag_p  = fine >> {coarse[WCRS-1:0], {WFINE{1'b0}}};
    wire     [WMAG-1:0] mag    = (coarse >= NCRS) ? {WMAG{1'b0}} : mag_p[WMAG-1:0];

    assign sign  = x[WFULL-1];
    assign zero  = ~|exp;
    assign oor   = exp >= (BIAS + WEXP - 1);
    assign ipart = mag[WMAG-1:WFF];
    assign f     = mag[WFF-1:0] ^ {WFF{sign}};
endmodule


// Result assembly of zkf_exp2: the float 2**i * v / 2**WCF, i = ipart for a positive x and -ipart-1 for a negative
// one, for a core sum s = v + half an ULP in [2**WCF, 2**(WCF+2)). Equivalent to _zkf_pack of v rounding half up, v
// taken as 1 where it falls short of 2**WCF and as 2 where s reaches 2**(WCF+1): s is truncated, and its late top bit
// only selects between two exponent fields.
//
// Where s reaches 2**(WCF+1) its fraction bits are zero by themselves, and a negative integer x, the core's last
// argument, gets there (zkf_trans_certify.py), which makes its 2**x exact: in the binade whose results are otherwise
// zero that is x = -BIAS alone, MIN_NORMAL/2, which rounds up.
//
// z = i + BIAS + 1 for both signs, so the exponent field is z-1, or z after the round-up. A field of -1 is below
// MIN_NORMAL/2 (zero), 0 is MIN_NORMAL, all ones is infinity.
module _zkf_exp2_pack #(
    parameter WEXP = 6,
    parameter WMAN = 18,
    parameter WCF  = 30    // fraction bits of s
) (
    input  wire                 sign,   // of x
    input  wire                 zero,   // x = 0; overrides the rest
    input  wire                 oor,    // |x| >= 2**(WEXP-1): +inf, or +0 for a negative x
    input  wire      [WEXP-2:0] ipart,  // floor(|x|)
    input  wire       [WCF+1:0] s,
    output wire [WEXP+WMAN-1:0] y
);
    localparam WFRAC = WMAN - 1;
    localparam integer BIAS = (1 << (WEXP - 1)) - 1;

    wire [WEXP-1:0] z = {~sign, ipart ^ {(WEXP-1){sign}}};
    reg  [WEXP-1:0] e_lo, e_hi;       // without and with the round-up
    reg             use_frac;
    always @* begin
        use_frac = 1'b0;
        case (1'b1)
            zero: begin
                e_lo = BIAS[WEXP-1:0];
                e_hi = BIAS[WEXP-1:0];
            end
            oor: begin
                e_lo = {WEXP{~sign}};
                e_hi = {WEXP{~sign}};
            end
            ~|z[WEXP-1:1]: begin
                e_lo = z;
                e_hi = {{(WEXP-1){1'b0}}, 1'b1};
            end
            default: begin
                e_lo     = z - 1'b1;
                e_hi     = z;
                use_frac = 1'b1;
            end
        endcase
    end
    assign y = {1'b0, s[WCF+1] ? e_hi : e_lo, s[WCF-1 -: WFRAC] & {WFRAC{use_frac}}};
endmodule

`default_nettype wire
