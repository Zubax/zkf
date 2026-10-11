// Combinational reference fused multiply-add for formal equivalence proofs: a*b + c.
// The exact product of the significands and c's significand are placed in one wide integer at a common scale, whose
// bit 0 weighs 2^(2 - 2*BIAS - 2*WFRAC), the least bit of the smallest product. They are added with their signs, the
// magnitude is normalized by an iterative leading-one scan, and the result is packed via zkf_pack_ref: one rounding.
// Structurally different from _zkf_add_core.v's ordered align + add + leading-zero-count pipeline.

`default_nettype none

module zkf_fma_ref #(
    parameter WEXP = 6,
    parameter WMAN = 18
) (
    input  wire [WEXP+WMAN-1:0] a,
    input  wire [WEXP+WMAN-1:0] b,
    input  wire [WEXP+WMAN-1:0] c,
    output wire [WEXP+WMAN-1:0] y
);
    localparam WFRAC         = WMAN - 1;
    localparam WFULL         = WEXP + WMAN;
    localparam integer BIAS  = (1 << (WEXP - 1)) - 1;
    localparam WEXP_UNBIASED = WEXP + $clog2(WMAN) + 4;

    // Shifts onto the common scale: the product by a_exp + b_exp - 2, c by c_exp + BIAS + WFRAC - 2.
    localparam WSHP     = WEXP + 1;
    localparam WSHC     = WEXP + $clog2(WMAN) + 2;
    // The largest product, both exponents at their finite maximum, tops c at its own; +1 for the carry of the sum.
    localparam WBIG_UNS = 2*WMAN + (1 << (WEXP + 1)) - 5;
    localparam WBIG     = WBIG_UNS + 1;                     // +1 for the sign

    // Decode.
    wire             a_sign = a[WFULL-1];
    wire             b_sign = b[WFULL-1];
    wire             c_sign = c[WFULL-1];
    wire [WEXP-1:0]  a_exp  = a[WFULL-2:WFRAC];
    wire [WEXP-1:0]  b_exp  = b[WFULL-2:WFRAC];
    wire [WEXP-1:0]  c_exp  = c[WFULL-2:WFRAC];
    wire             a_zero = ~|a_exp;
    wire             b_zero = ~|b_exp;
    wire             c_zero = ~|c_exp;
    wire             a_inf  =  &a_exp;
    wire             b_inf  =  &b_exp;
    wire             c_inf  =  &c_exp;
    wire [WMAN-1:0]  a_sig  = {1'b1, a[WFRAC-1:0]};
    wire [WMAN-1:0]  b_sig  = {1'b1, b[WFRAC-1:0]};
    wire [WMAN-1:0]  c_sig  = {1'b1, c[WFRAC-1:0]};

    // A zero factor makes the product zero, whatever the other one is; otherwise an infinite factor makes it infinite.
    wire p_zero   = a_zero || b_zero;
    wire p_inf    = !p_zero && (a_inf || b_inf);
    wire p_sign   = a_sign ^ b_sign;
    wire p_finite = !p_zero && !p_inf;
    wire c_finite = !c_zero && !c_inf;

    wire [2*WMAN-1:0]   prod    = a_sig * b_sig;
    wire [WSHP-1:0]     p_shamt = {1'b0, a_exp} + {1'b0, b_exp} - {{(WSHP-2){1'b0}}, 2'd2};
    wire [WSHC-1:0]     c_shamt = {{(WSHC-WEXP){1'b0}}, c_exp} + (BIAS + WFRAC - 2);
    wire [WBIG_UNS-1:0] p_uns   = p_finite ? ({{(WBIG_UNS-2*WMAN){1'b0}}, prod}  << p_shamt) : {WBIG_UNS{1'b0}};
    wire [WBIG_UNS-1:0] c_uns   = c_finite ? ({{(WBIG_UNS-WMAN){1'b0}},   c_sig} << c_shamt) : {WBIG_UNS{1'b0}};

    // Wide signed sum and its magnitude.
    wire signed [WBIG-1:0] p_pos   = {1'b0, p_uns};
    wire signed [WBIG-1:0] c_pos   = {1'b0, c_uns};
    wire signed [WBIG-1:0] sum_val = (p_sign ? -p_pos : p_pos) + (c_sign ? -c_pos : c_pos);
    wire                   sum_negative = sum_val[WBIG-1];
    wire signed [WBIG-1:0] sum_abs = sum_negative ? -sum_val : sum_val;
    wire    [WBIG_UNS-1:0] abs_sum = sum_abs[WBIG_UNS-1:0];

    // Infinity special cases.
    wire pp_force_zero = p_inf && c_inf && (p_sign != c_sign);
    wire pp_force_inf  = p_inf || c_inf;
    wire pp_inf_sign   = (p_inf && p_sign) || (c_inf && c_sign);

    // Leading-one scan: iterative low-to-high; last hit wins.
    integer i;
    reg [31:0] leading_int;
    reg        any_one;
    always @(*) begin
        leading_int = 32'd0;
        any_one     = 1'b0;
        for (i = 0; i < WBIG_UNS; i = i + 1) begin
            if (abs_sum[i]) begin
                leading_int = i[31:0];
                any_one     = 1'b1;
            end
        end
    end

    // Significand + GRS extraction. Pad abs_sum with WMAN+2 zero LSBs to keep all slices non-negative.
    localparam WPAD       = WMAN + 2;
    localparam WPAD_TOTAL = WBIG_UNS + WPAD;
    wire [WPAD_TOTAL-1:0] abs_pad  = {abs_sum, {WPAD{1'b0}}};
    wire [31:0]           pad_lead = leading_int + WPAD;

    reg [WMAN-1:0] f_sig;
    reg            f_g, f_r, f_s;
    always @(*) begin
        f_sig = abs_pad[pad_lead -: WMAN];
        f_g   = abs_pad[pad_lead - WMAN[31:0]];
        f_r   = abs_pad[pad_lead - WMAN[31:0] - 32'd1];
        f_s   = 1'b0;
        for (i = 0; i < WBIG_UNS; i = i + 1) begin
            if (i[31:0] + WMAN[31:0] + 32'd1 < leading_int) begin
                if (abs_sum[i]) f_s = 1'b1;
            end
        end
    end

    // A leading one at bit L is worth 2^(L + 2 - 2*BIAS - 2*WFRAC).
    integer exp_int;
    reg signed [WEXP_UNBIASED-1:0] f_exp_unb;
    always @(*) begin
        exp_int   = leading_int + 2 - 2*BIAS - 2*WFRAC;
        f_exp_unb = exp_int[WEXP_UNBIASED-1:0];
    end

    zkf_pack_ref #(.WEXP(WEXP), .WMAN(WMAN), .WEXP_UNBIASED(WEXP_UNBIASED)) u_pack (
        .sign(pp_force_inf ? pp_inf_sign : sum_negative),
        .force_zero(pp_force_zero || (!pp_force_inf && !any_one)),
        .force_inf(pp_force_inf),
        .exp_unbiased(f_exp_unb),
        .significand(f_sig),
        .guard(f_g),
        .round_bit(f_r),
        .sticky(f_s),
        .y(y)
    );
endmodule

`default_nettype wire
