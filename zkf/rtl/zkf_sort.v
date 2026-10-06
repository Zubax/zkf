// Streamed min/max sorter built on a combinational zkf_cmp, whose canonical-zero and same-sign-infinity equality it
// inherits.
//
// STAGE_INPUT=0: operands feed the sorter combinationally (default).
// STAGE_INPUT=1: latch the inputs before any combinational logic, isolating them from upstream paths (+1 cycle).
// STAGE_INPUT>1: add extra dummy stages; helps in routing-congested designs (+STAGE_INPUT cycles).

`default_nettype none

module zkf_sort #(
    parameter WEXP        = 6,
    parameter WMAN        = 18,
    parameter STAGE_INPUT = 0,
    parameter LATENCY     = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] a,
    input wire [WEXP+WMAN-1:0] b,

    output reg                 out_valid,
    output reg [WEXP+WMAN-1:0] min,
    output reg [WEXP+WMAN-1:0] max
);
    localparam WFULL = WEXP + WMAN;

    localparam LATENCY_REF = 1 + STAGE_INPUT;
    generate
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    // -- Optional input register stage: latch the operands before any combinational logic.
    wire             in_valid_q;
    wire [WFULL-1:0] a_q;
    wire [WFULL-1:0] b_q;
    zkf_pipe #(.W(2*WFULL), .N(STAGE_INPUT)) u_input_pipe (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in({b, a}),
        .out_valid(in_valid_q), .out({b_q, a_q})
    );

    wire a_lt_b;

    // Only the less-than result orders the pair.
    zkf_cmp #(.WEXP(WEXP), .WMAN(WMAN)) u_cmp (
        .clk(clk), .rst(rst), .in_valid(in_valid_q),
        .a(a_q), .b(b_q),
        .out_valid(), .a_gt_b(), .a_eq_b(), .a_lt_b(a_lt_b)
    );

    // Reset only stream validity. Payload registers intentionally free-run.
    always @(posedge clk) begin
        if (rst) out_valid <= 1'b0;
        else     out_valid <= in_valid_q;
        min <= a_lt_b ? a_q : b_q;
        max <= a_lt_b ? b_q : a_q;
    end
endmodule

`default_nettype wire
