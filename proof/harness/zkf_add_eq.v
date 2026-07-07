/// Formal harness: zkf_add DUT vs zkf_add_ref.

`default_nettype none

module zkf_add_eq #(parameter WEXP = 4, parameter WMAN = 6, parameter STAGE_INPUT = 0,
                    parameter STAGE_DECODE = 0, parameter STAGE_ALIGN = 0, parameter STAGE_NORMALIZE = 0,
                    parameter STAGE_PACK = 0, parameter STAGE_OUTPUT = 0, parameter LATENCY = 0) (
    input wire clk,
    input wire rst,
    input wire in_valid,
    input wire [WEXP+WMAN-1:0] a,
    input wire [WEXP+WMAN-1:0] b
);
    localparam WFULL           = WEXP + WMAN;
    localparam integer T_RESULT = 1 + LATENCY;
    localparam integer CYCLE_W  = (T_RESULT < 2) ? 2 : $clog2(T_RESULT + 2);

    reg [CYCLE_W-1:0] cycle = {CYCLE_W{1'b0}};
    always @(posedge clk) if (cycle < T_RESULT + 1) cycle <= cycle + 1'b1;

    always @(*) begin
        if (cycle == 0) begin
            assume(rst == 1'b1);
            assume(in_valid == 1'b0);
        end else if (cycle == 1) begin
            assume(rst == 1'b0);
            assume(in_valid == 1'b1);
        end else begin
            assume(rst == 1'b0);
            assume(in_valid == 1'b0);
        end
    end

    reg [WFULL-1:0] a_shadow, b_shadow;
    always @(posedge clk) if (cycle == 1) begin
        a_shadow <= a;
        b_shadow <= b;
    end

    wire             dut_valid;
    wire [WFULL-1:0] dut_y;
    zkf_add #(
        .WEXP(WEXP),
        .WMAN(WMAN),
        .STAGE_INPUT(STAGE_INPUT),
        .STAGE_DECODE(STAGE_DECODE),
        .STAGE_ALIGN(STAGE_ALIGN),
        .STAGE_NORMALIZE(STAGE_NORMALIZE),
        .STAGE_PACK(STAGE_PACK),
        .STAGE_OUTPUT(STAGE_OUTPUT),
        .LATENCY(LATENCY)
    ) u_dut (
        .clk(clk), .rst(rst), .in_valid(in_valid),
        .a(a), .b(b),
        .out_valid(dut_valid), .y(dut_y)
    );

    wire [WFULL-1:0] ref_y;
    zkf_add_ref #(.WEXP(WEXP), .WMAN(WMAN)) u_ref (
        .a(a_shadow), .b(b_shadow), .y(ref_y)
    );

    always @(posedge clk) begin
        if (cycle == T_RESULT) begin
            assert(dut_valid == 1'b1);
            assert(dut_y == ref_y);
        end
        if (cycle >= 1 && cycle < T_RESULT) assert(dut_valid == 1'b0);
    end
endmodule

`default_nettype wire
