// Streamed sign manipulation: `absolute` is x with the sign bit cleared, `negated` is x with it flipped. Both are bit
// operations, so +0 negates to a -0 pattern, which every operator decodes as +0.
//
// STAGE_INPUT: registers ahead of the logic.
// STAGE_OUTPUT: registers after it.
// Each knob is a register count costing as many cycles; values above one add dummy stages for routing-congested
// designs. With both zero the module is combinational and clk/rst are ignored.

`default_nettype none

module zkf_absneg #(
    parameter WEXP         = 6,
    parameter WMAN         = 18,
    parameter STAGE_INPUT  = 0,
    parameter STAGE_OUTPUT = 0,
    parameter LATENCY      = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] x,

    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] absolute,
    output wire [WEXP+WMAN-1:0] negated
);
    localparam WFULL       = WEXP + WMAN;
    localparam LATENCY_REF = STAGE_INPUT + STAGE_OUTPUT;
    generate
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    wire             valid_q;
    wire [WFULL-1:0] x_q;
    zkf_pipe #(.W(WFULL), .N(STAGE_INPUT)) u_input_pipe (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in(x),
        .out_valid(valid_q), .out(x_q)
    );
    zkf_pipe #(.W(WFULL), .N(STAGE_OUTPUT)) u_output_pipe (
        .clk(clk), .rst(rst), .in_valid(valid_q), .in({~x_q[WFULL-1], x_q[WFULL-2:0]}),
        .out_valid(out_valid), .out(negated)
    );
    assign absolute = {1'b0, negated[WFULL-2:0]};
endmodule

`default_nettype wire
