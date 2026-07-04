/// Formal harness: zkf_mul DUT (3 stages) vs zkf_mul_ref.

`default_nettype none

module zkf_mul_eq #(parameter WEXP = 5, parameter WMAN = 11, parameter STAGE_OUTPUT = 0) (
    input wire clk,
    input wire rst,
    input wire in_valid,
    input wire [WEXP+WMAN-1:0] a,
    input wire [WEXP+WMAN-1:0] b
);
    localparam WFULL    = WEXP + WMAN;
    // 1 product stage + STAGE_OUTPUT (registered pack output) -> result at cycle 1 + (1+STAGE_OUTPUT). STAGE_OUTPUT=0
    // makes the packed result combinational (the consumer registers it), so it is valid one cycle earlier.
    localparam T_RESULT = 2 + STAGE_OUTPUT;

    reg [3:0] cycle = 4'd0;
    always @(posedge clk) cycle <= (cycle == 4'd15) ? cycle : cycle + 4'd1;

    always @(*) begin
        if (cycle == 4'd0) begin
            assume(rst == 1'b1);
            assume(in_valid == 1'b0);
        end else if (cycle == 4'd1) begin
            assume(rst == 1'b0);
            assume(in_valid == 1'b1);
        end else begin
            assume(rst == 1'b0);
            assume(in_valid == 1'b0);
        end
    end

    reg [WFULL-1:0] a_shadow, b_shadow;
    always @(posedge clk) if (cycle == 4'd1) begin
        a_shadow <= a;
        b_shadow <= b;
    end

    wire             dut_valid;
    wire [WFULL-1:0] dut_y;
    zkf_mul #(.WEXP(WEXP), .WMAN(WMAN), .STAGE_OUTPUT(STAGE_OUTPUT)) u_dut (
        .clk(clk), .rst(rst), .in_valid(in_valid),
        .a(a), .b(b),
        .out_valid(dut_valid), .y(dut_y)
    );

    wire [WFULL-1:0] ref_y;
    zkf_mul_ref #(.WEXP(WEXP), .WMAN(WMAN)) u_ref (
        .a(a_shadow), .b(b_shadow), .y(ref_y)
    );

    always @(posedge clk) begin
        if (cycle == T_RESULT) begin
            assert(dut_valid == 1'b1);
            assert(dut_y == ref_y);
        end
        if (cycle >= 4'd1 && cycle < T_RESULT) assert(dut_valid == 1'b0);
    end
endmodule

`default_nettype wire
