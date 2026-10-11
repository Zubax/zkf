// Shared core of zkf_add and zkf_fma: the sum or difference of two magnitudes, aligned, normalized, rounded once and
// packed.
//
// The operands arrive ordered: `larger` has its leading one on its top bit and the biased exponent `exp`; `smaller` is
// shifted right by `shamt` against it. `same_sign` selects their sum, otherwise the difference larger - smaller. A zero
// or non-finite operand arrives as a zero magnitude, with force_zero / force_inf deciding such a result.
//
// Register stages: 4 + STAGE_ALIGN + STAGE_NORMALIZE + STAGE_PACK + STAGE_OUTPUT.
//
// STAGE_ALIGN=1 splits the alignment shifter. STAGE_NORMALIZE={0,1,2} is the number of register barriers in the
// normalizer of the difference; see _zkf_normshift. STAGE_PACK=1 registers the packer's inputs, STAGE_OUTPUT=1 the
// result; see _zkf_pack.

`default_nettype none

module _zkf_add_core #(
    parameter WEXP            = 6,
    parameter WMAN            = 18,
    parameter WSIG            = WMAN,       // bits of an operand's magnitude
    parameter WE              = WEXP + 1,   // bits of `exp`
    parameter WSHIFT          = WEXP,
    parameter STAGE_ALIGN     = 0,
    parameter STAGE_NORMALIZE = 0,
    parameter STAGE_PACK      = 0,
    parameter STAGE_OUTPUT    = 0
) (
    input wire clk,
    input wire rst,

    input wire                   in_valid,
    input wire                   finite_sign,   // of a finite result: the larger operand's
    input wire                   inf_sign,
    input wire                   same_sign,
    input wire                   force_zero,
    input wire                   force_inf,
    input wire signed   [WE-1:0] exp,
    input wire      [WSHIFT-1:0] shamt,
    input wire        [WSIG-1:0] larger,
    input wire        [WSIG-1:0] smaller,

    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    localparam WGRS   = 3;                  // guard/round/sticky below the operands
    localparam W      = WSIG + WGRS;
    localparam WINDEX = $clog2(W);          // normalize count
    localparam WEN    = (WE > (WINDEX + 2)) ? WE : (WINDEX + 2);   // exp less a normalize count

    // Stages 0 and 1: `smaller` is aligned; the rest waits beside it.
    wire                 s1_valid;
    wire                 s1_finite_sign, s1_inf_sign, s1_same_sign, s1_force_zero, s1_force_inf;
    wire signed [WE-1:0] s1_exp;
    wire      [WSIG-1:0] s1_large;
    zkf_pipe #(.W(5 + WE + WSIG), .N(2 + STAGE_ALIGN)) u_side (
        .clk(clk), .rst(rst), .in_valid(in_valid),
        .in({finite_sign, inf_sign, same_sign, force_zero, force_inf, exp, larger}),
        .out_valid(s1_valid),
        .out({s1_finite_sign, s1_inf_sign, s1_same_sign, s1_force_zero, s1_force_inf, s1_exp, s1_large})
    );
    reg [WSHIFT-1:0] s0_shamt;
    reg   [WSIG-1:0] s0_small;
    wire     [W-1:0] s0_aligned;
    reg      [W-1:0] s1_small;
    _zkf_rshift_sticky #(.W(W), .WSHIFT(WSHIFT), .STAGE_SPLIT(STAGE_ALIGN)) u_align (
        .clk(clk), .x({s0_small, {WGRS{1'b0}}}), .shamt(s0_shamt), .y(s0_aligned)
    );

    // One carry chain adds or subtracts. `larger` has nothing below its WSIG bits, so the sticky that the shifter jams
    // into bit 0 of `smaller` survives.
    wire [W:0] s1_b   = s1_same_sign ? {1'b0, s1_small} : ~{1'b0, s1_small};
    wire [W:0] s1_raw = {1'b0, s1_large, {WGRS{1'b0}}} + s1_b + {{W{1'b0}}, !s1_same_sign};

    // Stage 2: the raw result. A sum leads at bit W-1 or carries to bit W, which a difference never does; a difference
    // is normalized below, with the sum riding the normalizer's sideband so that both reach stage 3 together.
    reg                  s2_valid;
    reg                  s2_sign, s2_same_sign, s2_force_zero, s2_force_inf;
    reg signed [WEN-1:0] s2_exp;
    reg            [W:0] s2_raw;
    wire                 s2_carry = s2_raw[W];
    localparam WQ = 4 + WEN + WMAN + 3;
    wire              q_valid;
    wire     [WQ-1:0] q;
    wire              n_zero;
    wire [WINDEX-1:0] n_count;
    wire      [W-1:0] n_aligned;
    _zkf_normshift #(.W(W), .STAGE_SPLIT(STAGE_NORMALIZE), .WSB(WQ)) u_normalize (
        .clk(clk), .rst(rst),
        .in_valid(s2_valid),
        .sb_in({
            s2_sign, s2_same_sign, s2_force_zero, s2_force_inf, s2_exp + $signed({{(WEN-1){1'b0}}, s2_carry}),
            s2_carry ? s2_raw[W -: WMAN]     : s2_raw[W-1 -: WMAN],
            s2_carry ? s2_raw[W-WMAN]        : s2_raw[W-WMAN-1],
            s2_carry ? s2_raw[W-WMAN-1]      : s2_raw[W-WMAN-2],
            s2_carry ? (|s2_raw[W-WMAN-2:0]) : (|s2_raw[W-WMAN-3:0])
        }),
        .x(s2_raw[W-1:0]),
        .out_valid(q_valid), .sb_out(q),
        .zero(n_zero), .count(n_count), .y(n_aligned)
    );

    // Stage 3: both paths.
    reg                  s3_valid;
    reg                  s3_sign, s3_same_sign, s3_force_zero, s3_force_inf;
    reg signed [WEN-1:0] s3_exp;
    reg       [WMAN-1:0] s3_add_sig;
    reg                  s3_add_guard, s3_add_round, s3_add_sticky;
    reg                  s3_zero;
    reg     [WINDEX-1:0] s3_count;
    reg          [W-1:0] s3_aligned;
    wire s3_finite_zero = s3_same_sign ? ~|{s3_add_sig, s3_add_guard, s3_add_round, s3_add_sticky} : s3_zero;

    _zkf_pack #(
        .WEXP(WEXP), .WMAN(WMAN), .WEXP_UNBIASED(WEN), .EXP_IS_BIASED(1),
        .STAGE_INPUT(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT)
    ) u_pack (
        .clk(clk), .rst(rst),
        .in_valid(s3_valid),
        .sign(s3_sign),
        .force_zero(s3_force_zero || (!s3_force_inf && s3_finite_zero)),
        .force_inf(s3_force_inf),
        .exp_unbiased(s3_same_sign ? s3_exp : s3_exp - $signed({{(WEN-WINDEX){1'b0}}, s3_count})),
        .significand(s3_same_sign ? s3_add_sig : s3_aligned[W-1 -: WMAN]),
        .guard(s3_same_sign ? s3_add_guard : s3_aligned[W-WMAN-1]),
        .round(s3_same_sign ? s3_add_round : s3_aligned[W-WMAN-2]),
        .sticky(s3_same_sign ? s3_add_sticky : (|s3_aligned[W-WMAN-3:0])),
        .out_valid(out_valid), .y(y)
    );

    always @(posedge clk) begin
        if (rst) begin
            s2_valid <= 1'b0;
            s3_valid <= 1'b0;
        end else begin
            s2_valid <= s1_valid;
            s3_valid <= q_valid;
        end

        s0_shamt      <= shamt;
        s0_small      <= smaller;
        s1_small      <= s0_aligned;

        s2_sign       <= s1_force_inf ? s1_inf_sign : s1_finite_sign;
        s2_same_sign  <= s1_same_sign;
        s2_force_zero <= s1_force_zero;
        s2_force_inf  <= s1_force_inf;
        s2_exp        <= s1_exp;
        s2_raw        <= s1_raw;

        {s3_sign, s3_same_sign, s3_force_zero, s3_force_inf, s3_exp, s3_add_sig,
         s3_add_guard, s3_add_round, s3_add_sticky} <= q;
        s3_zero       <= n_zero;
        s3_count      <= n_count;
        s3_aligned    <= n_aligned;
    end
endmodule

`default_nettype wire
