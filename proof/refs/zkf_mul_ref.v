/// Combinational reference multiplier for formal equivalence proofs.
/// Independent from zkf_mul.v in style: combinational, single always-block, no pipeline,
/// no shared GRS-extraction expressions.

`default_nettype none

module zkf_mul_ref #(
    parameter WEXP = 6,
    parameter WMAN = 18
) (
    input  wire [WEXP+WMAN-1:0] a,
    input  wire [WEXP+WMAN-1:0] b,
    output wire [WEXP+WMAN-1:0] y
);
    localparam WFRAC         = WMAN - 1;
    localparam WFULL         = WEXP + WMAN;
    localparam WPROD         = 2 * WMAN;
    localparam WEXP_UNBIASED = WEXP + 2;

    // Decode.
    wire             a_sign  = a[WFULL-1];
    wire             b_sign  = b[WFULL-1];
    wire [WEXP-1:0]  a_exp   = a[WFULL-2:WFRAC];
    wire [WEXP-1:0]  b_exp   = b[WFULL-2:WFRAC];
    wire [WFRAC-1:0] a_frac  = a[WFRAC-1:0];
    wire [WFRAC-1:0] b_frac  = b[WFRAC-1:0];

    wire             a_zero  = ~|a_exp;
    wire             b_zero  = ~|b_exp;
    wire             a_inf   =  &a_exp;
    wire             b_inf   =  &b_exp;
    wire             rz      = a_zero || b_zero;
    wire             ri      = !rz && (a_inf || b_inf);

    wire [WMAN-1:0]  a_sig   = {1'b1, a_frac};
    wire [WMAN-1:0]  b_sig   = {1'b1, b_frac};

    wire [WPROD-1:0] product = a_sig * b_sig;
    wire             p_hi    = product[WPROD-1];

    // Unbiased exponent of the unnormalized product. Both operand biased exponents are in [1, EXP_INF-1].
    // bias_x2 = 2 * (2^(WEXP-1) - 1) fits in WEXP+1 unsigned bits.
    wire signed [WEXP_UNBIASED-1:0] a_exp_ext   = {{(WEXP_UNBIASED-WEXP){1'b0}}, a_exp};
    wire signed [WEXP_UNBIASED-1:0] b_exp_ext   = {{(WEXP_UNBIASED-WEXP){1'b0}}, b_exp};
    wire signed [WEXP_UNBIASED-1:0] bias_x2_ext = {{(WEXP_UNBIASED-WEXP-1){1'b0}},
                                                   {1'b0, {WEXP-1{1'b1}}}, 1'b0};  // 2*BIAS, unsigned
    wire signed [WEXP_UNBIASED-1:0] exp_base    = a_exp_ext + b_exp_ext - bias_x2_ext;

    reg signed [WEXP_UNBIASED-1:0] exp_unb;
    reg            [WMAN-1:0]      sig_v;
    reg                            g, r, s;

    always @(*) begin
        if (p_hi) begin
            exp_unb = exp_base + {{(WEXP_UNBIASED-1){1'b0}}, 1'b1};
            sig_v   = product[WPROD-1 -: WMAN];
            g       = product[WMAN-1];
            r       = product[WMAN-2];
            s       = |product[WMAN-3:0];
        end else begin
            exp_unb = exp_base;
            sig_v   = product[WPROD-2 -: WMAN];
            g       = product[WMAN-2];
            // For WMAN >= 4 this slice has width >= 1.
            r       = product[WMAN-3];
            // For WMAN == 4, this OR-reduction spans [0:0] only; that's well-defined.
            s       = (WMAN >= 5) ? |product[WMAN-4:0] : 1'b0;
        end
    end

    zkf_pack_ref #(.WEXP(WEXP), .WMAN(WMAN), .WEXP_UNBIASED(WEXP_UNBIASED)) u_pack (
        .sign(a_sign ^ b_sign),
        .force_zero(rz),
        .force_inf(ri),
        .exp_unbiased(exp_unb),
        .significand(sig_v),
        .guard(g),
        .round_bit(r),
        .sticky(s),
        .y(y)
    );
endmodule

`default_nettype wire
