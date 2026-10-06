// Formal harness: zkf_absneg DUT vs. the spec inlined. LATENCY is supplied by run_proofs.py from the shared Python
// model. At LATENCY 0 the DUT is checked as combinational logic under any rst and in_valid; otherwise through the
// single-pulse driver against the shadowed operand. A combinational instance on the DUT's results checks that negation
// is an involution and the absolute value idempotent.

`default_nettype none

module zkf_absneg_eq #(
    parameter WEXP = 6, parameter WMAN = 18, parameter STAGE_INPUT = 0, parameter STAGE_OUTPUT = 0,
    parameter LATENCY = 0
) (
    input wire clk,
    input wire rst,
    input wire in_valid,
    input wire [WEXP+WMAN-1:0] x
);
    localparam WFULL            = WEXP + WMAN;
    localparam integer T_RESULT = 1 + LATENCY;
    localparam integer CYCLE_W  = (T_RESULT < 2) ? 2 : $clog2(T_RESULT + 2);

    reg [CYCLE_W-1:0] cycle = {CYCLE_W{1'b0}};
    always @(posedge clk) if (cycle < T_RESULT + 1) cycle <= cycle + 1'b1;

    reg [WFULL-1:0] x_shadow;
    always @(posedge clk) if (cycle == 1) x_shadow <= x;

    wire             dut_valid;
    wire [WFULL-1:0] dut_absolute, dut_negated;
    zkf_absneg #(
        .WEXP(WEXP), .WMAN(WMAN), .STAGE_INPUT(STAGE_INPUT), .STAGE_OUTPUT(STAGE_OUTPUT), .LATENCY(LATENCY)
    ) u_dut (
        .clk(clk), .rst(rst), .in_valid(in_valid), .x(x),
        .out_valid(dut_valid), .absolute(dut_absolute), .negated(dut_negated)
    );

    wire [WFULL-1:0] abs_of_abs, neg_of_neg;
    zkf_absneg #(.WEXP(WEXP), .WMAN(WMAN)) u_abs_again (
        .clk(clk), .rst(rst), .in_valid(1'b0), .x(dut_absolute),
        .out_valid(), .absolute(abs_of_abs), .negated()
    );
    zkf_absneg #(.WEXP(WEXP), .WMAN(WMAN)) u_neg_again (
        .clk(clk), .rst(rst), .in_valid(1'b0), .x(dut_negated),
        .out_valid(), .absolute(), .negated(neg_of_neg)
    );

    wire [WFULL-1:0] x_ref      = (LATENCY == 0) ? x : x_shadow;
    wire             results_ok = (dut_absolute == {1'b0, x_ref[WFULL-2:0]}) &&
                                  (dut_negated == {~x_ref[WFULL-1], x_ref[WFULL-2:0]}) &&
                                  (abs_of_abs == dut_absolute) && (neg_of_neg == x_ref);

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
