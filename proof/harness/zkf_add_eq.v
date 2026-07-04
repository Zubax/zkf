/// Formal harness: zkf_add DUT (6 stages) vs zkf_add_ref.

`default_nettype none

module zkf_add_eq #(parameter WEXP = 4, parameter WMAN = 6, parameter STAGE_OUTPUT = 0) (
    input wire clk,
    input wire rst,
    input wire in_valid,
    input wire [WEXP+WMAN-1:0] a,
    input wire [WEXP+WMAN-1:0] b
);
    localparam WFULL    = WEXP + WMAN;
    // 4 internal stages + STAGE_OUTPUT (registered pack output); result at cycle 1 + 4 + STAGE_OUTPUT.
    localparam T_RESULT = 5 + STAGE_OUTPUT;

    reg [4:0] cycle = 5'd0;
    always @(posedge clk) cycle <= (cycle == 5'd31) ? cycle : cycle + 5'd1;

    always @(*) begin
        if (cycle == 5'd0) begin
            assume(rst == 1'b1);
            assume(in_valid == 1'b0);
        end else if (cycle == 5'd1) begin
            assume(rst == 1'b0);
            assume(in_valid == 1'b1);
        end else begin
            assume(rst == 1'b0);
            assume(in_valid == 1'b0);
        end
    end

    reg [WFULL-1:0] a_shadow, b_shadow;
    always @(posedge clk) if (cycle == 5'd1) begin
        a_shadow <= a;
        b_shadow <= b;
    end

    wire             dut_valid;
    wire [WFULL-1:0] dut_y;
    zkf_add #(.WEXP(WEXP), .WMAN(WMAN), .STAGE_OUTPUT(STAGE_OUTPUT)) u_dut (
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
        if (cycle >= 5'd1 && cycle < T_RESULT) assert(dut_valid == 1'b0);
    end
endmodule

`default_nettype wire
