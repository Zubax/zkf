// Formal harness: zkf_cmp DUT vs. zkf_cmp_ref (case-analysis reference); min/max must order the operands by it.
// LATENCY is supplied by run_proofs.py from the shared Python model. At LATENCY 0 the DUT is checked as combinational
// logic under any rst and in_valid; otherwise through the single-pulse driver against the shadowed operands.

`default_nettype none

module zkf_cmp_eq #(
    parameter WEXP = 6, parameter WMAN = 18, parameter STAGE_INPUT = 0, parameter STAGE_OUTPUT = 0,
    parameter LATENCY = 0
) (
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

    reg [WFULL-1:0] a_shadow, b_shadow;
    always @(posedge clk) if (cycle == 1) begin
        a_shadow <= a;
        b_shadow <= b;
    end

    wire             dut_valid, dut_gt, dut_eq, dut_lt;
    wire [WFULL-1:0] dut_min, dut_max;
    zkf_cmp #(
        .WEXP(WEXP), .WMAN(WMAN), .STAGE_INPUT(STAGE_INPUT), .STAGE_OUTPUT(STAGE_OUTPUT), .LATENCY(LATENCY)
    ) u_dut (
        .clk(clk), .rst(rst), .in_valid(in_valid),
        .a(a), .b(b),
        .out_valid(dut_valid),
        .a_gt_b(dut_gt), .a_eq_b(dut_eq), .a_lt_b(dut_lt), .min(dut_min), .max(dut_max)
    );

    // Second DUT instance with the operands swapped. Lets the harness prove anti-symmetry
    // (cmp(a,b).gt == cmp(b,a).lt and cmp is reflexive on equality) on the same arbitrary inputs as the
    // primary equivalence check, without leaning on the reference for those properties.
    wire dut_swap_valid, dut_swap_gt, dut_swap_eq, dut_swap_lt;
    zkf_cmp #(
        .WEXP(WEXP), .WMAN(WMAN), .STAGE_INPUT(STAGE_INPUT), .STAGE_OUTPUT(STAGE_OUTPUT), .LATENCY(LATENCY)
    ) u_dut_swap (
        .clk(clk), .rst(rst), .in_valid(in_valid),
        .a(b), .b(a),
        .out_valid(dut_swap_valid),
        .a_gt_b(dut_swap_gt), .a_eq_b(dut_swap_eq), .a_lt_b(dut_swap_lt), .min(), .max()
    );

    wire dut_self_valid, dut_self_gt, dut_self_eq, dut_self_lt;
    zkf_cmp #(
        .WEXP(WEXP), .WMAN(WMAN), .STAGE_INPUT(STAGE_INPUT), .STAGE_OUTPUT(STAGE_OUTPUT), .LATENCY(LATENCY)
    ) u_dut_self (
        .clk(clk), .rst(rst), .in_valid(in_valid),
        .a(a), .b(a),
        .out_valid(dut_self_valid),
        .a_gt_b(dut_self_gt), .a_eq_b(dut_self_eq), .a_lt_b(dut_self_lt), .min(), .max()
    );

    wire [WFULL-1:0] a_ref = (LATENCY == 0) ? a : a_shadow;
    wire [WFULL-1:0] b_ref = (LATENCY == 0) ? b : b_shadow;
    wire ref_gt, ref_eq, ref_lt;
    zkf_cmp_ref #(.WEXP(WEXP), .WMAN(WMAN)) u_ref (
        .a(a_ref), .b(b_ref),
        .a_gt_b(ref_gt), .a_eq_b(ref_eq), .a_lt_b(ref_lt)
    );

    wire results_ok = (dut_gt == ref_gt) && (dut_eq == ref_eq) && (dut_lt == ref_lt) &&
                      (dut_min == (ref_lt ? a_ref : b_ref)) && (dut_max == (ref_lt ? b_ref : a_ref)) &&
                      ((dut_gt + dut_eq + dut_lt) == 3'd1) &&
                      (dut_gt == dut_swap_lt) && (dut_lt == dut_swap_gt) && (dut_eq == dut_swap_eq) &&
                      dut_self_eq && !dut_self_gt && !dut_self_lt;

    generate
        if (LATENCY == 0) begin : g_combinational
            always @(*) begin
                assert(dut_valid == in_valid);
                assert(results_ok);
            end
        end else begin : g_pulse
            // Pin reset and valid pattern: rst=1 cycle 0; rst=0 + in_valid=1 cycle 1; in_valid=0 thereafter.
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
            always @(posedge clk) begin
                if (cycle == T_RESULT) begin
                    assert(dut_valid == 1'b1);
                    assert(results_ok);
                end
                if (cycle >= 1 && cycle < T_RESULT) assert(dut_valid == 1'b0);
            end
        end
    endgenerate
endmodule

`default_nettype wire
