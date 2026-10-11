// Formal harness: zkf_fma DUT vs zkf_fma_ref.

`default_nettype none

module zkf_fma_eq #(parameter WEXP = 4, parameter WMAN = 6, parameter STAGE_INPUT = 0,
                    parameter STAGE_PRODUCT = 0,
                    parameter STAGE_DECODE = 0, parameter STAGE_ALIGN = 0, parameter STAGE_NORMALIZE = 0,
                    parameter STAGE_PACK = 0, parameter STAGE_OUTPUT = 0, parameter LATENCY = 0) (
    input wire clk,
    input wire rst,
    input wire in_valid,
    input wire [WEXP+WMAN-1:0] a,
    input wire [WEXP+WMAN-1:0] b,
    input wire [WEXP+WMAN-1:0] c
);
    localparam WFULL           = WEXP + WMAN;
    localparam integer T_RESULT = 1 + LATENCY;
    localparam integer CYCLE_W  = (T_RESULT < 2) ? 2 : $clog2(T_RESULT + 2);

    reg [CYCLE_W-1:0] cycle = {CYCLE_W{1'b0}};
    always @(posedge clk) if (cycle < T_RESULT + 1) cycle <= cycle + 1'b1;

    always @(*) begin
        assume(rst == (cycle == 0));
        assume(in_valid == (cycle == 1));
    end

    reg [WFULL-1:0] a_shadow, b_shadow, c_shadow;
    always @(posedge clk) if (cycle == 1) begin
        a_shadow <= a;
        b_shadow <= b;
        c_shadow <= c;
    end

    wire             dut_valid;
    wire [WFULL-1:0] dut_y;
    zkf_fma #(
        .WEXP(WEXP),
        .WMAN(WMAN),
        .STAGE_INPUT(STAGE_INPUT),
        .STAGE_PRODUCT(STAGE_PRODUCT),
        .STAGE_DECODE(STAGE_DECODE),
        .STAGE_ALIGN(STAGE_ALIGN),
        .STAGE_NORMALIZE(STAGE_NORMALIZE),
        .STAGE_PACK(STAGE_PACK),
        .STAGE_OUTPUT(STAGE_OUTPUT),
        .LATENCY(LATENCY)
    ) u_dut (
        .clk(clk), .rst(rst), .in_valid(in_valid),
        .a(a), .b(b), .c(c),
        .out_valid(dut_valid), .y(dut_y)
    );

    wire [WFULL-1:0] ref_y;
    zkf_fma_ref #(.WEXP(WEXP), .WMAN(WMAN)) u_ref (
        .a(a_shadow), .b(b_shadow), .c(c_shadow), .y(ref_y)
    );

    always @(posedge clk) begin
        if (cycle >= 1) assert(dut_valid == (cycle == T_RESULT));
        if (cycle == T_RESULT) assert(dut_y == ref_y);
    end
endmodule

`default_nettype wire
