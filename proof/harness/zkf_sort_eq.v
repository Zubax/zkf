/// Formal harness: zkf_sort DUT.
/// Asserts:
///   - DUT.min and DUT.max are a permutation of the latched (a, b)
///   - cmp_ref(min, max) is lt-or-eq (sort orders correctly)
/// LATENCY is supplied by run_proofs.py from the shared Python model.

`default_nettype none

module zkf_sort_eq #(parameter WEXP = 6, parameter WMAN = 18, parameter STAGE_INPUT = 0, parameter LATENCY = 0) (
    input wire clk,
    input wire rst,
    input wire in_valid,
    input wire [WEXP+WMAN-1:0] a,
    input wire [WEXP+WMAN-1:0] b
);
    localparam WFULL            = WEXP + WMAN;
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
    wire [WFULL-1:0] dut_min, dut_max;
    zkf_sort #(.WEXP(WEXP), .WMAN(WMAN), .STAGE_INPUT(STAGE_INPUT), .LATENCY(LATENCY)) u_dut (
        .clk(clk), .rst(rst), .in_valid(in_valid),
        .a(a), .b(b),
        .out_valid(dut_valid), .min(dut_min), .max(dut_max)
    );

    wire ord_gt, ord_eq, ord_lt;
    zkf_cmp_ref #(.WEXP(WEXP), .WMAN(WMAN)) u_cmp_ord (
        .a(dut_min), .b(dut_max),
        .a_gt_b(ord_gt), .a_eq_b(ord_eq), .a_lt_b(ord_lt)
    );

    wire perm_ab = (dut_min == a_shadow) && (dut_max == b_shadow);
    wire perm_ba = (dut_min == b_shadow) && (dut_max == a_shadow);

    always @(posedge clk) begin
        if (cycle == T_RESULT) begin
            assert(dut_valid == 1'b1);
            assert(!ord_gt);
            assert(perm_ab || perm_ba);
        end
        if (cycle >= 1 && cycle < T_RESULT) assert(dut_valid == 1'b0);
    end
endmodule

`default_nettype wire
