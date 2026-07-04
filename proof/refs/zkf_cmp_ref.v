/// Combinational reference comparator for formal equivalence proofs.
/// Implements the comparison spec by explicit case analysis on operand class
/// (zero / negative-finite / positive-finite / negative-inf / positive-inf),
/// deliberately different from the key-transform trick used in zkf_cmp_comb.v
/// so that a shared logic bug would not cancel out under equivalence checking.

`default_nettype none

module zkf_cmp_ref #(
    parameter WEXP = 6,
    parameter WMAN = 18
) (
    input  wire [WEXP+WMAN-1:0] a,
    input  wire [WEXP+WMAN-1:0] b,

    output reg  a_gt_b,
    output reg  a_eq_b,
    output reg  a_lt_b
);
    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;

    // Decompose.
    wire             a_sign_raw = a[WFULL-1];
    wire             b_sign_raw = b[WFULL-1];
    wire [WEXP-1:0]  a_exp      = a[WFULL-2:WFRAC];
    wire [WEXP-1:0]  b_exp      = b[WFULL-2:WFRAC];
    wire [WFRAC-1:0] a_frac     = a[WFRAC-1:0];
    wire [WFRAC-1:0] b_frac     = b[WFRAC-1:0];

    wire a_is_zero = (a_exp == {WEXP{1'b0}});
    wire b_is_zero = (b_exp == {WEXP{1'b0}});
    wire a_is_inf  = (a_exp == {WEXP{1'b1}});
    wire b_is_inf  = (b_exp == {WEXP{1'b1}});

    // Canonical zero has no sign; canonical inf keeps the sign.
    wire a_sign = a_is_zero ? 1'b0 : a_sign_raw;
    wire b_sign = b_is_zero ? 1'b0 : b_sign_raw;

    // Finite-magnitude key: concatenation of (exp, frac) gives an unsigned key whose ordering matches positive
    // numerical magnitude. Only meaningful when both operands are finite non-zero.
    wire [WEXP+WFRAC-1:0] a_mag = {a_exp, a_frac};
    wire [WEXP+WFRAC-1:0] b_mag = {b_exp, b_frac};
    wire mag_a_gt_b = a_mag >  b_mag;
    wire mag_a_lt_b = a_mag <  b_mag;
    wire mag_a_eq_b = a_mag == b_mag;

    // Equality cases:
    //   both zero (any sign/frac) — canonicalize to +0
    //   both same-signed infinity — canonicalize to {sign, exp_inf, 0}
    //   both same-signed finite non-zero with equal (exp, frac)
    wire both_zero          = a_is_zero && b_is_zero;
    wire both_inf_same_sign = a_is_inf  && b_is_inf  && (a_sign_raw == b_sign_raw);
    wire both_finite_eq     = !a_is_zero && !b_is_zero && !a_is_inf && !b_is_inf
                              && (a_sign_raw == b_sign_raw) && mag_a_eq_b;

    always @(*) begin
        a_gt_b = 1'b0;
        a_lt_b = 1'b0;
        a_eq_b = both_zero || both_inf_same_sign || both_finite_eq;

        if (a_eq_b) begin
            // covered above
        end else if (a_sign && !b_sign) begin
            // a strictly negative, b non-negative
            a_lt_b = 1'b1;
        end else if (!a_sign && b_sign) begin
            // a non-negative, b strictly negative
            a_gt_b = 1'b1;
        end else if (a_sign && b_sign) begin
            // both strictly negative (neither is +0; -0 has been canonicalised to +0 by sign forcing)
            if (a_is_inf && !b_is_inf) begin
                a_lt_b = 1'b1;             // -inf < -finite
            end else if (!a_is_inf && b_is_inf) begin
                a_gt_b = 1'b1;             // -finite > -inf
            end else begin
                // both negative finite, ordered by reverse magnitude
                if (mag_a_gt_b)      a_lt_b = 1'b1;
                else if (mag_a_lt_b) a_gt_b = 1'b1;
            end
        end else begin
            // both non-negative (sign = 0 effectively)
            if (a_is_zero && !b_is_zero) begin
                a_lt_b = 1'b1;             // +0 < +finite or +inf
            end else if (!a_is_zero && b_is_zero) begin
                a_gt_b = 1'b1;
            end else if (a_is_inf && !b_is_inf) begin
                a_gt_b = 1'b1;             // +inf > +finite
            end else if (!a_is_inf && b_is_inf) begin
                a_lt_b = 1'b1;
            end else begin
                // both positive finite, ordered by magnitude
                if (mag_a_gt_b)      a_gt_b = 1'b1;
                else if (mag_a_lt_b) a_lt_b = 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
