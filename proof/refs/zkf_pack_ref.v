/// Combinational reference packer for formal equivalence proofs.

`default_nettype none

module zkf_pack_ref #(
    parameter WEXP          = 6,
    parameter WMAN          = 18,
    parameter WEXP_UNBIASED = WEXP + 2,
    parameter EXP_IS_BIASED = 0,
    parameter SATURATE_ROUND_CARRY = 0
) (
    input  wire                            sign,
    input  wire                            force_zero,
    input  wire                            force_inf,
    input  wire signed [WEXP_UNBIASED-1:0] exp_unbiased,
    input  wire                 [WMAN-1:0] significand,
    input  wire                            guard,
    input  wire                            round_bit,
    input  wire                            sticky,
    output reg          [WEXP+WMAN-1:0]    y
);
    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;

    localparam [WEXP-1:0] EXP_BIAS       = {1'b0, {WEXP-1{1'b1}}};
    localparam [WEXP-1:0] EXP_INF        = {WEXP{1'b1}};
    localparam [WEXP-1:0] EXP_MAX_FINITE = EXP_INF - {{(WEXP-1){1'b0}}, 1'b1};

    // Bounds are carried one bit wider than the port: at WEXP_UNBIASED == WEXP the zero-extension below would be
    // zero-width, landing EXP_MAX_FINITE's top bit on the sign bit so it reads as -2 instead of EXP_INF-1.
    localparam integer WEXT = WEXP_UNBIASED + 1;

    wire signed [WEXT-1:0] exp_in             = {{(WEXT-WEXP_UNBIASED){exp_unbiased[WEXP_UNBIASED-1]}}, exp_unbiased};
    wire signed [WEXT-1:0] bias_ext           = {{(WEXT-WEXP){1'b0}}, EXP_BIAS};
    wire signed [WEXT-1:0] exp_max_finite_ext = {{(WEXT-WEXP){1'b0}}, EXP_MAX_FINITE};
    wire signed [WEXT-1:0] one_ext            = {{(WEXT-1){1'b0}}, 1'b1};
    // The port is the BIASED exponent when EXP_IS_BIASED, so the range bounds it is tested against shift with it.
    wire signed [WEXT-1:0] bias_adj           = EXP_IS_BIASED ? {WEXT{1'b0}} : bias_ext;
    wire signed [WEXT-1:0] min_exp_unbiased   = one_ext - bias_adj;
    wire signed [WEXT-1:0] max_exp_unbiased   = exp_max_finite_ext - bias_adj;

    reg signed [WEXT-1:0]          exp_biased_ext;
    reg            [WEXP-1:0]      exp_biased;
    reg                            exp_underflow_zero;
    reg                            exp_one_below_min;
    reg                            exp_overflow;
    reg                            round_increment;
    reg              [WMAN:0]      rounded_ext;
    reg                            round_carry;
    reg            [WMAN-1:0]      rounded_significand;
    reg            [WEXP-1:0]      exp_rounded;
    reg                            exp_round_overflow;
    reg                            infinity_flag;
    reg                            result_zero;
    reg                            result_infinity;
    reg                            result_saturate;

    always @(*) begin
        exp_biased_ext        = EXP_IS_BIASED ? exp_in : exp_in + bias_ext;
        exp_biased            = exp_biased_ext[WEXP-1:0];
        exp_underflow_zero    = exp_in < (min_exp_unbiased - one_ext);
        exp_one_below_min     = exp_in == (min_exp_unbiased - one_ext);
        exp_overflow          = exp_in > max_exp_unbiased;

        round_increment       = guard && (round_bit || sticky || significand[0]);
        rounded_ext           = {1'b0, significand} + {{WMAN{1'b0}}, round_increment};
        round_carry           = rounded_ext[WMAN];
        rounded_significand   = round_carry ? rounded_ext[WMAN:1] : rounded_ext[WMAN-1:0];
        exp_rounded           = exp_biased + {{(WEXP-1){1'b0}}, round_carry};
        exp_round_overflow    = (exp_biased == EXP_MAX_FINITE) && round_carry;
        infinity_flag         = force_inf || exp_overflow ||
                                (exp_round_overflow && (SATURATE_ROUND_CARRY == 0));

        result_zero           = force_zero || (!force_inf && exp_underflow_zero);
        result_infinity       = !result_zero && infinity_flag;
        // Saturation applies only to the ROUND-CARRY route, which exp_round_overflow already names exactly.
        result_saturate       = (SATURATE_ROUND_CARRY != 0) && exp_round_overflow;

        if (result_zero)
            y = {WFULL{1'b0}};
        else if (result_infinity)
            y = {sign, EXP_INF, {WFRAC{1'b0}}};
        else if (!force_inf && exp_one_below_min)
            y = {sign, {{(WEXP-1){1'b0}}, 1'b1}, {WFRAC{1'b0}}};
        else if (result_saturate)
            y = {sign, EXP_MAX_FINITE, {WFRAC{1'b1}}};
        else
            y = {sign, exp_rounded, rounded_significand[WFRAC-1:0]};
    end
endmodule

`default_nettype wire
