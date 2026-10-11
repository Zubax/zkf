// Streamed floating-point compare producing three mutually exclusive flags and the ordered pair. Any exponent-zero
// pattern compares equal to canonical +0, and infinities of the same sign compare equal regardless of their fraction
// bits. min and max are the operands themselves, not canonicalized; when they compare equal, min is b and max is a.
// An unused result may be left unconnected; its logic is pruned.
//
// The operands become monotonic keys (sign-magnitude to ordered unsigned) beside the zero/infinity classification, so
// the one wide compare does not wait on it: `<` is the borrow of a subtraction, which maps onto the carry chain, `==`
// an XOR-reduce, and `>` the remaining case.
//
// STAGE_INPUT: registers ahead of the compare.
// STAGE_OUTPUT: registers after it.
// Each knob is a register count costing as many cycles; values above one add dummy stages for routing-congested
// designs. With both zero the module is combinational and clk/rst are ignored.

`default_nettype none

module zkf_cmp #(
    parameter WEXP         = 6,
    parameter WMAN         = 18,
    parameter STAGE_INPUT  = 0,
    parameter STAGE_OUTPUT = 0,
    parameter LATENCY      = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] a,
    input wire [WEXP+WMAN-1:0] b,

    output wire                 out_valid,
    output wire                 a_gt_b,
    output wire                 a_eq_b,
    output wire                 a_lt_b,
    output wire [WEXP+WMAN-1:0] min,
    output wire [WEXP+WMAN-1:0] max
);
    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;

    localparam LATENCY_REF = STAGE_INPUT + STAGE_OUTPUT;
    generate
        if ((WEXP < 2) || (WMAN < 4)) begin : g_invalid_wexp_or_wman
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    wire             valid_q;
    wire [WFULL-1:0] a_q;
    wire [WFULL-1:0] b_q;
    zkf_pipe #(.W(2*WFULL), .N(STAGE_INPUT)) u_input_pipe (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in({b, a}),
        .out_valid(valid_q), .out({b_q, a_q})
    );

    wire [WFULL-1:0] a_key = {~a_q[WFULL-1], a_q[WFULL-2:0] ^ {(WFULL-1){a_q[WFULL-1]}}};
    wire [WFULL-1:0] b_key = {~b_q[WFULL-1], b_q[WFULL-2:0] ^ {(WFULL-1){b_q[WFULL-1]}}};
    wire             a_zero = ~|a_q[WFULL-2:WFRAC];
    wire             b_zero = ~|b_q[WFULL-2:WFRAC];
    wire             a_inf  = &a_q[WFULL-2:WFRAC];
    wire             b_inf  = &b_q[WFULL-2:WFRAC];
    wire             same   = (a_zero & b_zero) | (a_inf & b_inf & ~(a_q[WFULL-1] ^ b_q[WFULL-1]));
    wire [WFULL:0]   diff   = {1'b0, a_key} - {1'b0, b_key};
    wire             eq     = (a_key == b_key) | same;
    wire             lt     = diff[WFULL] & ~same;

    zkf_pipe #(.W(3 + 2*WFULL), .N(STAGE_OUTPUT)) u_output_pipe (
        .clk(clk),
        .rst(rst),
        .in_valid(valid_q),
        .in({~(lt | eq), eq, lt, lt ? a_q : b_q, lt ? b_q : a_q}),
        .out_valid(out_valid),
        .out({a_gt_b, a_eq_b, a_lt_b, min, max})
    );
endmodule

`default_nettype wire
