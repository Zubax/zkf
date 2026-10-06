// Streamed finiteness: `finite` is set unless x is an infinity; `saturated` is x when finite and otherwise the largest
// finite magnitude with the sign of x. A finite x passes through bit for bit, without canonicalization.
//
// STAGE_INPUT: registers ahead of the logic; values above one add dummy stages for routing-congested designs.
// STAGE_OUTPUT: register the results.
// Each knob costs as many cycles as its value. With both zero the module is combinational and clk/rst are ignored.

`default_nettype none

module zkf_finite #(
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
    output wire                 finite,
    output wire [WEXP+WMAN-1:0] saturated
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
    wire [WFULL-1:0] x_q;
    zkf_pipe #(.W(WFULL), .N(STAGE_INPUT)) u_input_pipe (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in(x),
        .out_valid(valid_q), .out(x_q)
    );

    wire fin = ~&x_q[WFULL-2:WFRAC];
    zkf_pipe #(.W(WFULL + 1), .N(STAGE_OUTPUT)) u_output_pipe (
        .clk(clk), .rst(rst), .in_valid(valid_q),
        .in({fin, fin ? x_q : {x_q[WFULL-1], {(WEXP-1){1'b1}}, 1'b0, {WFRAC{1'b1}}}}),
        .out_valid(out_valid), .out({finite, saturated})
    );
endmodule

`default_nettype wire
