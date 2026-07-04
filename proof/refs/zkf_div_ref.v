/// Combinational reference divider for formal equivalence proofs.
/// Implementation: special-case classification, then a wide integer division a_sig << QFRAC_REF / b_sig,
/// then GRS extraction and pack_ref.
/// Structurally different from the unrolled radix-4 chain in _zkf_div_core.v.

`default_nettype none

module zkf_div_ref #(
    parameter WEXP = 6,
    parameter WMAN = 18
) (
    input  wire [WEXP+WMAN-1:0] a,
    input  wire [WEXP+WMAN-1:0] b,
    output wire [WEXP+WMAN-1:0] q,
    output wire                 div0
);
    localparam WFRAC         = WMAN - 1;
    localparam WFULL         = WEXP + WMAN;
    localparam WEXP_UNBIASED = WEXP + 2;

    // Enough fractional precision so the bits of the quotient below the rounding position are well-defined.
    localparam QFRAC_REF = WMAN + 4;
    localparam WDIVIDEND = WMAN + QFRAC_REF;

    // Decode.
    wire             a_sign = a[WFULL-1];
    wire             b_sign = b[WFULL-1];
    wire [WEXP-1:0]  a_exp  = a[WFULL-2:WFRAC];
    wire [WEXP-1:0]  b_exp  = b[WFULL-2:WFRAC];
    wire [WFRAC-1:0] a_frac = a[WFRAC-1:0];
    wire [WFRAC-1:0] b_frac = b[WFRAC-1:0];

    wire             a_zero = ~|a_exp;
    wire             b_zero = ~|b_exp;
    wire             a_inf  =  &a_exp;
    wire             b_inf  =  &b_exp;
    wire [WMAN-1:0]  a_sig  = {1'b1, a_frac};
    wire [WMAN-1:0]  b_sig  = {1'b1, b_frac};

    assign div0 = b_zero;

    // Special-case overrides.
    wire force_zero  = a_zero || b_inf;
    wire force_inf   = !force_zero && (b_zero || a_inf);
    wire result_sign = b_zero ? a_sign : (a_sign ^ b_sign);

    // Wide division. Both significands have MSB=1 so b_sig is never 0 — safe to divide.
    // dividend = a_sig << QFRAC_REF, divisor = b_sig.
    // quotient is in (0.5, 2); after the shift, scaled quotient has bit (QFRAC_REF) or (QFRAC_REF-1) as the leading one.
    wire [WDIVIDEND-1:0] dividend = {a_sig, {QFRAC_REF{1'b0}}};
    wire [WDIVIDEND-1:0] divisor  = {{QFRAC_REF{1'b0}}, b_sig};
    wire [WDIVIDEND-1:0] q_wide   = dividend / divisor;
    wire [WDIVIDEND-1:0] r_wide   = dividend % divisor;
    wire                 rem_nz   = |r_wide;

    // q_high distinguishes the [1,2) case (bit QFRAC_REF set) from the [0.5,1) case.
    wire q_high = q_wide[QFRAC_REF];

    wire signed [WEXP_UNBIASED-1:0] a_exp_ext = {{(WEXP_UNBIASED-WEXP){1'b0}}, a_exp};
    wire signed [WEXP_UNBIASED-1:0] b_exp_ext = {{(WEXP_UNBIASED-WEXP){1'b0}}, b_exp};
    wire signed [WEXP_UNBIASED-1:0] exp_diff  = a_exp_ext - b_exp_ext;
    wire signed [WEXP_UNBIASED-1:0] one_ext   = {{(WEXP_UNBIASED-1){1'b0}}, 1'b1};

    reg signed [WEXP_UNBIASED-1:0] exp_out;
    reg            [WMAN-1:0]      sig_out;
    reg                            g_out;
    reg                            r_out;
    reg                            sticky_tail;

    always @(*) begin
        if (q_high) begin
            // hidden bit at QFRAC_REF
            exp_out     = exp_diff;
            sig_out     = q_wide[QFRAC_REF -: WMAN];
            g_out       = q_wide[QFRAC_REF - WMAN];
            r_out       = q_wide[QFRAC_REF - WMAN - 1];
            sticky_tail = |q_wide[QFRAC_REF - WMAN - 2 : 0];
        end else begin
            // hidden bit at QFRAC_REF - 1
            exp_out     = exp_diff - one_ext;
            sig_out     = q_wide[QFRAC_REF - 1 -: WMAN];
            g_out       = q_wide[QFRAC_REF - 1 - WMAN];
            r_out       = q_wide[QFRAC_REF - 1 - WMAN - 1];
            sticky_tail = |q_wide[QFRAC_REF - 1 - WMAN - 2 : 0];
        end
    end

    wire sticky_out = sticky_tail || rem_nz;

    zkf_pack_ref #(.WEXP(WEXP), .WMAN(WMAN), .WEXP_UNBIASED(WEXP_UNBIASED)) u_pack (
        .sign(result_sign),
        .force_zero(force_zero),
        .force_inf(force_inf),
        .exp_unbiased(exp_out),
        .significand(sig_out),
        .guard(g_out),
        .round_bit(r_out),
        .sticky(sticky_out),
        .y(q)
    );
endmodule

`default_nettype wire
