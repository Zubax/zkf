// Power-of-two multiplier: y = a * 2^k, where k is a signed integer (ldexp/scalbn).
// This is far cheaper than full multiplication (zkf_mul) or division (zkf_divsqrt) because the significand is preserved
// bit-for-bit and only the biased exponent is shifted by k, and no rounding is required -- the operation is exact
// in the format's normal range.
//
// k is a signed value WK bits wide. Any k is legal: shifts that push the result past the format's range simply
// saturate to signed infinity (overflow) or flush to zero (underflow), exactly as ldexp would. The default width
// spans the entire useful range; widen WK if k is driven from a wider computed value.
//
// STAGE_INPUT=0: operand and k feed the decode combinationally (default).
// STAGE_INPUT=1: latch {a, k} before any combinational logic, isolating them from upstream paths (+1 cycle).
// STAGE_INPUT>1: add extra dummy stages; helps in routing-congested designs (+STAGE_INPUT cycles).
//
// STAGE_DECODE: registers between the decode and the output mux; splits the long route from the input to the output
//     register (the dominant delay path at wide WMAN on placement-sensitive tools).

`default_nettype none

module zkf_mul_ilog2 #(
    parameter         WEXP         = 6,
    parameter         WMAN         = 18,        // significand precision including the hidden bit
    parameter         WK           = WEXP + 1,  // width of the signed exponent shift k; default spans the useful range
    parameter         STAGE_INPUT  = 0,         // number of input register stages (>=0); +STAGE_INPUT cycles
    parameter         STAGE_DECODE = 0,         // number of decode register stages (>=0); +STAGE_DECODE cycles
    parameter         LATENCY      = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] a,
    input wire signed [WK-1:0] k,

    output reg                 out_valid,
    output reg [WEXP+WMAN-1:0] y           // y = a * 2^k
);
    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;
    localparam WK_NRW = WEXP + 1;
    localparam WACC   = WEXP + 2;

    localparam [WEXP-1:0] EXP_INF = {WEXP{1'b1}};   // = 2^WEXP-1; new_exp >= EXP_INF is the overflow boundary

    localparam LATENCY_REF = 1 + STAGE_INPUT + STAGE_DECODE;
    generate
        if ((WEXP < 2) || (WMAN < 4)) begin : g_invalid_wm
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        if (WK < 1) begin : g_invalid_wk
            _zkf_invalid_mul_ilog2_wk u_invalid();
        end
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    // Optional input register stage: latch the operand and k together before any combinational logic.
    wire             in_valid_q;
    wire [WFULL-1:0] a_q;
    wire [WK-1:0]    k_q;
    zkf_pipe #(.W(WFULL + WK), .N(STAGE_INPUT)) u_input_pipe (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in({k, a}),
        .out_valid(in_valid_q), .out({k_q, a_q})
    );

    // Decode and classify.
    wire             a_sign    = a_q[WFULL-1];
    wire [WEXP-1:0]  a_exp     = a_q[WFULL-2:WFRAC];
    wire [WFRAC-1:0] a_frac    = a_q[WFRAC-1:0];
    wire             a_zero    = ~|a_exp;
    wire             a_inf     =  &a_exp;
    wire             a_special = a_zero || a_inf;   // zero/inf pass through regardless of k, with absolute priority

    wire signed [WK_NRW-1:0] k_nrw;
    wire k_sat_pos;
    wire k_sat_neg;
    generate
        if (WK <= WK_NRW) begin : g_k_extend
            assign k_nrw = {{(WK_NRW-WK){k_q[WK-1]}}, k_q};
            assign k_sat_pos = 1'b0;
            assign k_sat_neg = 1'b0;
        end else begin : g_k_saturate
            wire [WK-WK_NRW:0] k_upper = k_q[WK-1:WK_NRW-1];
            wire k_in_range = &(k_upper ~^ {WK-WK_NRW+1{k_q[WK-1]}});
            assign k_nrw = $signed(k_q[WK_NRW-1:0]);
            // Discarded shifts already force every normal input to zero or infinity, so class saturation is exact.
            assign k_sat_pos = !k_in_range && !k_q[WK-1];
            assign k_sat_neg = !k_in_range &&  k_q[WK-1];
        end
    endgenerate

    wire signed [WACC-1:0] a_exp_acc = $signed({{(WACC-WEXP){1'b0}}, a_exp});
    wire signed [WACC-1:0] k_exp_acc = $signed({{(WACC-WK_NRW){k_nrw[WK_NRW-1]}}, k_nrw});
    wire signed [WACC-1:0] new_exp_acc = a_exp_acc + k_exp_acc;
    // The subtraction's sign bit implements >= EXP_INF as a carry chain; wide comparators mapped poorly in some tools.
    wire signed [WACC-1:0] of_acc = new_exp_acc - $signed({{(WACC-WEXP){1'b0}}, EXP_INF});
    wire overflow   = !a_special && (k_sat_pos || (!k_sat_neg && ~of_acc[WACC-1]));      // >= EXP_INF
    wire underflow  = !a_special && (k_sat_neg || (!k_sat_pos && new_exp_acc[WACC-1]));  // < 0

    wire result_is_zero       = a_zero || underflow;
    wire result_is_inf        = a_inf  || overflow;
    wire result_is_min_normal = !a_special && ~|new_exp_acc;  // == 0 rounds up to MIN_NORMAL (>= 0.5*MIN_NORMAL)

    // Normal-output exponent: low WEXP bits of (a_exp + k). The truncating slice is fine because the normal-output mux
    // is suppressed whenever result_is_zero or result_is_inf is asserted.
    wire [WEXP-1:0] new_exp = new_exp_acc[WEXP-1:0];

    wire             o_valid, o_sign, o_zero, o_inf, o_min_normal;
    wire [WEXP-1:0]  o_exp;
    wire [WFRAC-1:0] o_frac;
    zkf_pipe #(.W(WFULL + 3), .N(STAGE_DECODE)) u_decode_pipe (
        .clk(clk), .rst(rst), .in_valid(in_valid_q),
        .in({a_sign, result_is_zero, result_is_inf, result_is_min_normal, new_exp, a_frac}),
        .out_valid(o_valid), .out({o_sign, o_zero, o_inf, o_min_normal, o_exp, o_frac})
    );

    // Canonicalization is implicit: zero has sign/frac cleared, infinity has frac cleared. Zero and infinity exclude
    // each other; both take priority over min-normal, which a saturated k can raise alongside them.
    // Kept by ZKF_ATTRIBUTE_KEEP: LSE otherwise forces the special results through the flip-flops' synchronous reset,
    // whose net place-and-route may promote onto clock routing.
`ifdef ZKF_ATTRIBUTE_KEEP
    `ZKF_ATTRIBUTE_KEEP
`endif
    reg [WFULL-1:0] y_next;
    always @* begin
        case ({o_zero, o_inf, o_min_normal})
            3'b100, 3'b101, 3'b110, 3'b111: y_next = {WFULL{1'b0}};
            3'b010, 3'b011:                 y_next = {o_sign, EXP_INF, {WFRAC{1'b0}}};
            3'b001:                         y_next = {o_sign, {{(WEXP-1){1'b0}}, 1'b1}, {WFRAC{1'b0}}};
            default:                        y_next = {o_sign, o_exp, o_frac};
        endcase
    end

    // Reset only stream validity. Payload register intentionally free-runs (project Reset strategy).
    always @(posedge clk) begin
        if (rst) out_valid <= 1'b0;
        else     out_valid <= o_valid;
        y <= y_next;
    end
endmodule

`default_nettype wire
