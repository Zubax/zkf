/// Formal harness: zkf_sort DUT.
/// Pipeline depth 1. Asserts:
///   - DUT.min and DUT.max are a permutation of the latched (a, b)
///   - cmp_ref(min, max) is lt-or-eq (sort orders correctly)
///   - validity latency exactly 1 cycle

`default_nettype none

module zkf_sort_eq #(parameter WEXP = 6, parameter WMAN = 18) (
    input wire clk,
    input wire rst,
    input wire in_valid,
    input wire [WEXP+WMAN-1:0] a,
    input wire [WEXP+WMAN-1:0] b
);
    localparam WFULL = WEXP + WMAN;
    localparam T_RESULT = 2;

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

    // DUT.
    wire             dut_valid;
    wire [WFULL-1:0] dut_min, dut_max;
    zkf_sort #(.WEXP(WEXP), .WMAN(WMAN)) u_dut (
        .clk(clk), .rst(rst), .in_valid(in_valid),
        .a(a), .b(b),
        .out_valid(dut_valid), .min(dut_min), .max(dut_max)
    );

    // Ordering check via reference: cmp_ref(min, max).gt must be 0 (min <= max).
    wire ord_gt, ord_eq, ord_lt;
    zkf_cmp_ref #(.WEXP(WEXP), .WMAN(WMAN)) u_cmp_ord (
        .a(dut_min), .b(dut_max),
        .a_gt_b(ord_gt), .a_eq_b(ord_eq), .a_lt_b(ord_lt)
    );

    // Permutation check (bit-equality on raw bits). We accept either {a,b} or {b,a}.
    wire perm_ab = (dut_min == a_shadow) && (dut_max == b_shadow);
    wire perm_ba = (dut_min == b_shadow) && (dut_max == a_shadow);

    always @(posedge clk) begin
        if (cycle == T_RESULT) begin
            assert(dut_valid == 1'b1);
            // min <= max.
            assert(!ord_gt);
            // multiset preservation.
            assert(perm_ab || perm_ba);
        end
        if (cycle == 4'd1) assert(dut_valid == 1'b0);
    end
endmodule

`default_nettype wire
