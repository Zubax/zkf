/// Combinational reference packer for formal equivalence proofs.

`default_nettype none

module zkf_pack_ref #(
    parameter WEXP          = 6,
    parameter WMAN          = 18,
    parameter WEXP_UNBIASED = WEXP + 2
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

    wire signed [WEXP_UNBIASED-1:0] bias_ext           = {{(WEXP_UNBIASED-WEXP){1'b0}}, EXP_BIAS};
    wire signed [WEXP_UNBIASED-1:0] exp_max_finite_ext = {{(WEXP_UNBIASED-WEXP){1'b0}}, EXP_MAX_FINITE};
    wire signed [WEXP_UNBIASED-1:0] one_ext            = {{(WEXP_UNBIASED-1){1'b0}}, 1'b1};
    wire signed [WEXP_UNBIASED-1:0] min_exp_unbiased   = one_ext - bias_ext;
    wire signed [WEXP_UNBIASED-1:0] max_exp_unbiased   = exp_max_finite_ext - bias_ext;

    reg signed [WEXP_UNBIASED-1:0] exp_biased_ext;
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

    always @(*) begin
        exp_biased_ext        = exp_unbiased + bias_ext;
        exp_biased            = exp_biased_ext[WEXP-1:0];
        exp_underflow_zero    = exp_unbiased < (min_exp_unbiased - one_ext);
        exp_one_below_min     = exp_unbiased == (min_exp_unbiased - one_ext);
        exp_overflow          = exp_unbiased > max_exp_unbiased;

        round_increment       = guard && (round_bit || sticky || significand[0]);
        rounded_ext           = {1'b0, significand} + {{WMAN{1'b0}}, round_increment};
        round_carry           = rounded_ext[WMAN];
        rounded_significand   = round_carry ? rounded_ext[WMAN:1] : rounded_ext[WMAN-1:0];
        exp_rounded           = exp_biased + {{(WEXP-1){1'b0}}, round_carry};
        exp_round_overflow    = (exp_biased == EXP_MAX_FINITE) && round_carry;
        infinity_flag         = force_inf || exp_overflow || exp_round_overflow;

        result_zero           = force_zero || (!force_inf && exp_underflow_zero);
        result_infinity       = !result_zero && infinity_flag;

        if (result_zero)
            y = {WFULL{1'b0}};
        else if (result_infinity)
            y = {sign, EXP_INF, {WFRAC{1'b0}}};
        else if (!force_inf && exp_one_below_min)
            y = {sign, {{(WEXP-1){1'b0}}, 1'b1}, {WFRAC{1'b0}}};
        else
            y = {sign, exp_rounded, rounded_significand[WFRAC-1:0]};
    end
endmodule

`default_nettype wire
