/// Combinational reference adder for formal equivalence proofs.
/// Algorithm: each finite operand is converted to an unsigned integer at a common scale
/// (bit i of `a_uns` represents 2^(i + 1 - BIAS - WFRAC) of the real value, with the smallest
/// finite operand's hidden bit at bit position WMAN-1+0=WMAN-1, etc.). Signs convert the pair to a
/// wide signed sum, the magnitude is normalized by an iterative leading-one scan, and the result is
/// packed via zkf_pack_ref. Structurally different from zkf_add.v's six-stage exponent-aligned
/// right-shift + add + leading-zero-count pipeline.

`default_nettype none

module zkf_add_ref #(
    parameter WEXP = 6,
    parameter WMAN = 18
) (
    input  wire [WEXP+WMAN-1:0] a,
    input  wire [WEXP+WMAN-1:0] b,
    output wire [WEXP+WMAN-1:0] y
);
    localparam WFRAC         = WMAN - 1;
    localparam WFULL         = WEXP + WMAN;
    localparam WEXP_UNBIASED = WEXP + 2;

    // Wide accumulator. With shared scale OFFSET = 1 - BIAS - WFRAC, a finite operand at biased exp e
    // lands with its hidden-bit LSB at bit position (e - 1). Max e = 2^WEXP - 2, so max hidden-bit
    // position = 2^WEXP - 3, and the operand occupies WMAN bits upward from there. The largest single
    // operand thus needs WMAN + 2^WEXP - 3 bits unsigned. Sum needs +1 carry, signed needs +1 sign.
    localparam WMAX_POS = (1 << WEXP) - 3;          // largest hidden-bit position
    localparam WBIG_UNS = WMAN + WMAX_POS + 1;      // +1 for two-operand carry
    localparam WBIG     = WBIG_UNS + 1;             // +1 for sign

    // Decode.
    wire             a_sign      = a[WFULL-1];
    wire             b_sign      = b[WFULL-1];
    wire [WEXP-1:0]  a_exp       = a[WFULL-2:WFRAC];
    wire [WEXP-1:0]  b_exp       = b[WFULL-2:WFRAC];
    wire [WFRAC-1:0] a_frac      = a[WFRAC-1:0];
    wire [WFRAC-1:0] b_frac      = b[WFRAC-1:0];

    wire             a_zero      = ~|a_exp;
    wire             b_zero      = ~|b_exp;
    wire             a_inf       =  &a_exp;
    wire             b_inf       =  &b_exp;
    wire             a_finite_nz = !a_zero && !a_inf;
    wire             b_finite_nz = !b_zero && !b_inf;
    wire [WMAN-1:0]  a_sig       = {1'b1, a_frac};
    wire [WMAN-1:0]  b_sig       = {1'b1, b_frac};

    // Shift amount: e - 1 for a finite operand. Range [0, 2^WEXP - 3].
    wire [WEXP-1:0]  a_shamt     = a_exp - {{(WEXP-1){1'b0}}, 1'b1};
    wire [WEXP-1:0]  b_shamt     = b_exp - {{(WEXP-1){1'b0}}, 1'b1};

    // Wide unsigned representation: a_sig << a_shamt (zero-padded to WBIG_UNS bits beforehand).
    reg [WBIG_UNS-1:0] a_uns;
    reg [WBIG_UNS-1:0] b_uns;
    integer si;
    always @(*) begin
        a_uns = {WBIG_UNS{1'b0}};
        if (a_finite_nz) begin
            for (si = 0; si < WMAN; si = si + 1) begin
                if (a_sig[si]) a_uns[si + a_shamt] = 1'b1;
            end
        end
        b_uns = {WBIG_UNS{1'b0}};
        if (b_finite_nz) begin
            for (si = 0; si < WMAN; si = si + 1) begin
                if (b_sig[si]) b_uns[si + b_shamt] = 1'b1;
            end
        end
    end

    // Wide signed sum.
    wire signed [WBIG-1:0] a_pos = {1'b0, a_uns};
    wire signed [WBIG-1:0] b_pos = {1'b0, b_uns};
    wire signed [WBIG-1:0] a_val = a_sign ? -a_pos : a_pos;
    wire signed [WBIG-1:0] b_val = b_sign ? -b_pos : b_pos;
    wire signed [WBIG-1:0] sum_val = a_val + b_val;

    // Magnitude.
    wire                 sum_negative = sum_val < $signed({WBIG{1'b0}});
    wire signed [WBIG-1:0] sum_abs    = sum_negative ? -sum_val : sum_val;
    wire        [WBIG_UNS-1:0] abs_sum = sum_abs[WBIG_UNS-1:0];

    // Infinity special cases.
    wire pp_force_zero = a_inf && b_inf && (a_sign != b_sign);
    wire pp_force_inf  = a_inf || b_inf;
    wire pp_inf_sign   = (a_inf && a_sign) || (b_inf && b_sign);

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
    wire [WPAD_TOTAL-1:0] abs_pad = {abs_sum, {WPAD{1'b0}}};
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

    // Exponent: hidden bit at position leading_int has real value 2^(leading_int + OFFSET) where
    // OFFSET = 1 - BIAS - WFRAC = 2 - BIAS - WMAN. So exp_unbiased = leading_int + 2 - BIAS - WMAN.
    integer  bias_int;
    integer  exp_int;
    reg signed [WEXP_UNBIASED-1:0] f_exp_unb;
    always @(*) begin
        bias_int = (1 << (WEXP - 1)) - 1;
        exp_int  = leading_int + 2 - bias_int - WMAN;
        f_exp_unb = exp_int[WEXP_UNBIASED-1:0];
    end

    // Final pack inputs.
    reg                            final_force_zero;
    reg                            final_force_inf;
    reg                            final_sign;
    reg signed [WEXP_UNBIASED-1:0] final_exp;
    reg            [WMAN-1:0]      final_sig;
    reg                            final_g, final_r, final_s;
    always @(*) begin
        if (pp_force_zero) begin
            final_force_zero = 1'b1;
            final_force_inf  = 1'b0;
            final_sign       = 1'b0;
            final_exp        = {WEXP_UNBIASED{1'b0}};
            final_sig        = {WMAN{1'b0}};
            final_g          = 1'b0;
            final_r          = 1'b0;
            final_s          = 1'b0;
        end else if (pp_force_inf) begin
            final_force_zero = 1'b0;
            final_force_inf  = 1'b1;
            final_sign       = pp_inf_sign;
            final_exp        = {WEXP_UNBIASED{1'b0}};
            final_sig        = {WMAN{1'b0}};
            final_g          = 1'b0;
            final_r          = 1'b0;
            final_s          = 1'b0;
        end else if (!any_one) begin
            final_force_zero = 1'b1;
            final_force_inf  = 1'b0;
            final_sign       = 1'b0;
            final_exp        = {WEXP_UNBIASED{1'b0}};
            final_sig        = {WMAN{1'b0}};
            final_g          = 1'b0;
            final_r          = 1'b0;
            final_s          = 1'b0;
        end else begin
            final_force_zero = 1'b0;
            final_force_inf  = 1'b0;
            final_sign       = sum_negative;
            final_exp        = f_exp_unb;
            final_sig        = f_sig;
            final_g          = f_g;
            final_r          = f_r;
            final_s          = f_s;
        end
    end

    zkf_pack_ref #(.WEXP(WEXP), .WMAN(WMAN), .WEXP_UNBIASED(WEXP_UNBIASED)) u_pack (
        .sign(final_sign),
        .force_zero(final_force_zero),
        .force_inf(final_force_inf),
        .exp_unbiased(final_exp),
        .significand(final_sig),
        .guard(final_g),
        .round_bit(final_r),
        .sticky(final_s),
        .y(y)
    );
endmodule

`default_nettype wire
