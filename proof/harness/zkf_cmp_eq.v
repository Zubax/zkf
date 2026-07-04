/// Formal harness: zkf_cmp DUT vs. zkf_cmp_ref (case-analysis reference).
/// Pipeline depth 1: drive at cycle 1, expect output at cycle 2.

`default_nettype none

module zkf_cmp_eq #(parameter WEXP = 6, parameter WMAN = 18) (
    input wire clk,
    input wire rst,
    input wire in_valid,
    input wire [WEXP+WMAN-1:0] a,
    input wire [WEXP+WMAN-1:0] b
);
    localparam WFULL = WEXP + WMAN;
    localparam N_STAGES = 1;
    localparam T_RESULT = N_STAGES + 1;

    // Free-running cycle counter, starts at 0.
    reg [3:0] cycle = 4'd0;
    always @(posedge clk) cycle <= (cycle == 4'd15) ? cycle : cycle + 4'd1;

    // Pin reset and valid pattern: rst=1 cycle 0; rst=0 + in_valid=1 cycle 1; in_valid=0 thereafter.
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

    // Latch inputs at cycle 1 into shadow registers.
    reg [WFULL-1:0] a_shadow, b_shadow;
    always @(posedge clk) if (cycle == 4'd1) begin
        a_shadow <= a;
        b_shadow <= b;
    end

    // DUT.
    wire dut_valid, dut_gt, dut_eq, dut_lt;
    zkf_cmp #(.WEXP(WEXP), .WMAN(WMAN)) u_dut (
        .clk(clk), .rst(rst), .in_valid(in_valid),
        .a(a), .b(b),
        .out_valid(dut_valid),
        .a_gt_b(dut_gt), .a_eq_b(dut_eq), .a_lt_b(dut_lt)
    );

    // Second DUT instance with the operands swapped. Lets the harness prove anti-symmetry
    // (cmp(a,b).gt == cmp(b,a).lt and cmp is reflexive on equality) on the same arbitrary inputs as the
    // primary equivalence check, without leaning on the reference for those properties.
    wire dut_swap_valid, dut_swap_gt, dut_swap_eq, dut_swap_lt;
    zkf_cmp #(.WEXP(WEXP), .WMAN(WMAN)) u_dut_swap (
        .clk(clk), .rst(rst), .in_valid(in_valid),
        .a(b), .b(a),
        .out_valid(dut_swap_valid),
        .a_gt_b(dut_swap_gt), .a_eq_b(dut_swap_eq), .a_lt_b(dut_swap_lt)
    );

    // Third DUT instance comparing a against itself, used to assert reflexivity.
    wire dut_self_valid, dut_self_gt, dut_self_eq, dut_self_lt;
    zkf_cmp #(.WEXP(WEXP), .WMAN(WMAN)) u_dut_self (
        .clk(clk), .rst(rst), .in_valid(in_valid),
        .a(a), .b(a),
        .out_valid(dut_self_valid),
        .a_gt_b(dut_self_gt), .a_eq_b(dut_self_eq), .a_lt_b(dut_self_lt)
    );

    // Reference applied to the latched inputs.
    wire ref_gt, ref_eq, ref_lt;
    zkf_cmp_ref #(.WEXP(WEXP), .WMAN(WMAN)) u_ref (
        .a(a_shadow), .b(b_shadow),
        .a_gt_b(ref_gt), .a_eq_b(ref_eq), .a_lt_b(ref_lt)
    );

    always @(posedge clk) begin
        if (cycle == T_RESULT) begin
            assert(dut_valid == 1'b1);
            assert(dut_gt == ref_gt);
            assert(dut_eq == ref_eq);
            assert(dut_lt == ref_lt);
            // One-hot output.
            assert((dut_gt + dut_eq + dut_lt) == 3'd1);
            // Anti-symmetry: cmp(a,b) and cmp(b,a) must agree with gt/lt swapped, eq preserved.
            assert(dut_gt == dut_swap_lt);
            assert(dut_lt == dut_swap_gt);
            assert(dut_eq == dut_swap_eq);
            // Reflexivity: cmp(a,a) always reports equality.
            assert(dut_self_eq == 1'b1);
            assert(dut_self_gt == 1'b0);
            assert(dut_self_lt == 1'b0);
        end
        if (cycle == 4'd1) begin
            assert(dut_valid == 1'b0);
        end
    end
endmodule

`default_nettype wire
