// Streamed Zubax Kulibin fused multiply-add: y = a*b + c.
// Subtraction a*b - c is achieved by inverting the sign bit of c.
// The exact 2*WMAN-bit product is carried through alignment, add, and normalize, so a*b+c is rounded once.
// That single rounding is the reason a true FMA is fundamentally wider than a chained zkf_mul -> zkf_add.
// The product, at its full width, and c are summed by _zkf_add_core, as zkf_add's operands are.
//
// LATENCY = 5 + STAGE_*

`default_nettype none

module zkf_fma #(
    parameter WEXP            = 6,
    parameter WMAN            = 18,   // significand precision including the hidden bit
    parameter WMULTIPLIER     = 0,    // see _zkf_pmul
    parameter STAGE_INPUT     = 0,    // number of input register stages
    parameter STAGE_PRODUCT   = 0,    // see _zkf_pmul
    parameter STAGE_DECODE    = 0,    // registers the product and c ahead of their ordering by magnitude
    parameter STAGE_ALIGN     = 0,    // see _zkf_add_core
    parameter STAGE_NORMALIZE = 0,    // see _zkf_add_core
    parameter STAGE_PACK      = 0,    // see _zkf_add_core
    parameter STAGE_OUTPUT    = 0,    // see _zkf_add_core
    parameter LATENCY         = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] a,
    input wire [WEXP+WMAN-1:0] b,
    input wire [WEXP+WMAN-1:0] c,

    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    localparam LATENCY_REF =
        5 + STAGE_INPUT + STAGE_PRODUCT + STAGE_DECODE + STAGE_ALIGN + STAGE_NORMALIZE + STAGE_PACK + STAGE_OUTPUT;
    generate
        if ((WEXP < 2) || (WMAN < 4)) begin : g_invalid_wman
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
        if ((STAGE_DECODE != 0) && (STAGE_DECODE != 1)) begin : g_invalid_stage_decode
            _zkf_invalid_stage_decode u_invalid();
        end
    endgenerate

    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;
    localparam WMAG  = 2 * WMAN;    // the full product
    localparam WEU   = WEXP + 2;    // signed biased exponent: holds a_exp + b_exp - BIAS + 1
    localparam WDIFF = WEU + 1;
    localparam [WEXP-1:0] EXP_BIAS = {1'b0, {WEXP-1{1'b1}}};
    // Below every finite exponent: a zero or non-finite operand, whose magnitude enters as zero, sorts as the smaller.
    localparam signed [WEU-1:0] EXP_MIN = {1'b1, {(WEU-1){1'b0}}};

    wire             in_valid_q;
    wire [WFULL-1:0] a_q;
    wire [WFULL-1:0] b_q;
    wire [WFULL-1:0] c_q;
    zkf_pipe #(.W(3*WFULL), .N(STAGE_INPUT)) u_input_pipe (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in({c, b, a}),
        .out_valid(in_valid_q), .out({c_q, b_q, a_q})
    );

    wire [WEXP-1:0] a_exp = a_q[WFULL-2:WFRAC];
    wire [WEXP-1:0] b_exp = b_q[WFULL-2:WFRAC];
    wire [WEXP-1:0] c_exp = c_q[WFULL-2:WFRAC];
    // As in zkf_mul, a zero factor gives a zero product, which then only c's infinity overrides; otherwise an infinite
    // factor gives an infinite one.
    wire p_zero = ~|a_exp | ~|b_exp;
    wire p_inf  = ~p_zero & (&a_exp | &b_exp);
    // The product's biased exponent less its leading bit, which is added once the product is known.
    wire signed [WEU-1:0] p_exp_base = {2'b00, a_exp} + {2'b00, b_exp} - {2'b00, EXP_BIAS};

    wire                  pr_valid;
    wire       [WMAG-1:0] pr_product;
    wire                  pr_p_sign, pr_p_zero, pr_p_inf, pr_c_sign, pr_c_zero, pr_c_inf;
    wire signed [WEU-1:0] pr_ep_base;
    wire       [WMAN-1:0] pr_c_sig;
    wire       [WEXP-1:0] pr_c_exp;
    _zkf_pmul #(
        .WA(WMAN), .WB(WMAN), .SIGNED(0),
        .WSB(WEU + WMAN + WEXP + 6), .WMULTIPLIER(WMULTIPLIER), .STAGE_PRODUCT(STAGE_PRODUCT)
    ) u_pmul (
        .clk(clk), .rst(rst), .in_valid(in_valid_q),
        .sb_in({
            a_q[WFULL-1] ^ b_q[WFULL-1], p_zero, p_inf, p_exp_base,
            {1'b1, c_q[WFRAC-1:0]}, c_exp, c_q[WFULL-1], ~|c_exp, &c_exp
        }),
        .a({1'b1, a_q[WFRAC-1:0]}), .b({1'b1, b_q[WFRAC-1:0]}),
        .out_valid(pr_valid),
        .sb_out({pr_p_sign, pr_p_zero, pr_p_inf, pr_ep_base, pr_c_sig, pr_c_exp, pr_c_sign, pr_c_zero, pr_c_inf}),
        .p(pr_product)
    );

    // The product leads at bit WMAG-1 or one below.
    wire                  mag_high    = pr_product[WMAG-1];
    wire       [WMAG-1:0] pr_norm     = mag_high ? pr_product : (pr_product << 1);
    wire signed [WEU-1:0] pr_ep       = pr_ep_base + $signed({{(WEU-1){1'b0}}, mag_high});
    wire                  pr_p_finite = ~pr_p_zero & ~pr_p_inf;
    wire                  pr_c_finite = ~pr_c_zero & ~pr_c_inf;

    wire                  d_valid;
    wire       [WMAG-1:0] d_p_key;
    wire       [WMAN-1:0] d_c_key;
    wire signed [WEU-1:0] d_ep_eff;
    wire signed [WEU-1:0] d_ec_eff;
    wire                  d_p_sign, d_c_sign, d_p_inf, d_c_inf;
    zkf_pipe #(.W(WMAG + WMAN + 2*WEU + 4), .N(STAGE_DECODE)) u_decode_pipe (
        .clk(clk), .rst(rst), .in_valid(pr_valid),
        .in({
            pr_p_finite ? pr_norm : {WMAG{1'b0}}, pr_c_finite ? pr_c_sig : {WMAN{1'b0}},
            pr_p_finite ? pr_ep : EXP_MIN, pr_c_finite ? $signed({2'b00, pr_c_exp}) : EXP_MIN,
            pr_p_sign, pr_c_sign, pr_p_inf, pr_c_inf
        }),
        .out_valid(d_valid),
        .out({d_p_key, d_c_key, d_ep_eff, d_ec_eff, d_p_sign, d_c_sign, d_p_inf, d_c_inf})
    );

    // Order by magnitude: by exponent, and at equal exponents by significand, c aligned to the product's top; the
    // product wins a tie.
    wire         [WMAG-1:0] c_key_wide = {d_c_key, {WMAN{1'b0}}};
    wire signed [WDIFF-1:0] ediff      = {d_ep_eff[WEU-1], d_ep_eff} - {d_ec_eff[WEU-1], d_ec_eff};
    // Exponent equality from the operands, beside the subtract rather than behind its carry chain.
    wire                    ediff_zero = ~|(d_ep_eff ^ d_ec_eff);
    wire           [WMAG:0] tie_diff   = {1'b0, d_p_key} - {1'b0, c_key_wide};
    wire product_ge_c = (~ediff[WDIFF-1] & ~ediff_zero) | (ediff_zero & ~tie_diff[WMAG]);

    _zkf_add_core #(
        .WEXP(WEXP), .WMAN(WMAN), .WSIG(WMAG), .WE(WEU), .WSHIFT(WDIFF),
        .STAGE_ALIGN(STAGE_ALIGN), .STAGE_NORMALIZE(STAGE_NORMALIZE), .STAGE_PACK(STAGE_PACK),
        .STAGE_OUTPUT(STAGE_OUTPUT)
    ) u_core (
        .clk(clk), .rst(rst),
        .in_valid(d_valid),
        .finite_sign(product_ge_c ? d_p_sign : d_c_sign),
        .inf_sign((d_p_inf & d_p_sign) | (d_c_inf & d_c_sign)),
        .same_sign(~(d_p_sign ^ d_c_sign)),
        .force_zero(d_p_inf & d_c_inf & (d_p_sign != d_c_sign)),
        .force_inf(d_p_inf | d_c_inf),
        .exp(product_ge_c ? d_ep_eff : d_ec_eff),
        .shamt(ediff[WDIFF-1] ? (~ediff + 1'b1) : ediff),
        .larger(product_ge_c ? d_p_key : c_key_wide),
        .smaller(product_ge_c ? c_key_wide : d_p_key),
        .out_valid(out_valid), .y(y)
    );
endmodule

`default_nettype wire
