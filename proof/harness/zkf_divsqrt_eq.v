// Formal harness: zkf_divsqrt vs zkf_div_ref / zkf_sqrt_ref under a free-running stimulus. Every cycle may carry a
// transaction of either operation (or none), reset may assert at any time after the first cycle, and each result is
// compared with the references' answer for the operands it was issued with, LATENCY cycles earlier.
// LATENCY is supplied by run_proofs.py from the shared Python model.

`default_nettype none

module zkf_divsqrt_eq #(
    parameter WEXP = 4, parameter WMAN = 6, parameter STAGE_INPUT = 0, parameter STAGE_DECODE = 0,
    parameter STAGE_PACK = 0, parameter STAGE_OUTPUT = 0, parameter MODE = 2, parameter LATENCY = 0
) (
    input wire                 clk,
    input wire                 rst,
    input wire                 in_valid,
    input wire                 op_sqrt,
    input wire [WEXP+WMAN-1:0] a,
    input wire [WEXP+WMAN-1:0] b
);
    localparam WFULL = WEXP + WMAN;
    localparam WR    = WFULL + 1;

    reg init = 1'b1;
    always @(posedge clk) init <= 1'b0;
    always @(*) if (init) assume(rst);

    wire             dut_valid, dut_error;
    wire [WFULL-1:0] dut_y;
    zkf_divsqrt #(
        .WEXP(WEXP), .WMAN(WMAN), .STAGE_INPUT(STAGE_INPUT), .STAGE_DECODE(STAGE_DECODE), .STAGE_PACK(STAGE_PACK),
        .STAGE_OUTPUT(STAGE_OUTPUT), .MODE(MODE), .LATENCY(LATENCY)
    ) u_dut (
        .clk(clk), .rst(rst), .in_valid(in_valid), .op_sqrt(op_sqrt), .a(a), .b(b),
        .out_valid(dut_valid), .y(dut_y), .error(dut_error)
    );

    wire             op = (MODE == 2) ? op_sqrt : (MODE == 1);
    wire [WFULL-1:0] q, root;
    wire             div0, de;
    zkf_div_ref #(.WEXP(WEXP), .WMAN(WMAN)) u_div (.a(a), .b(b), .q(q), .div0(div0));
    zkf_sqrt_ref #(.WEXP(WEXP), .WMAN(WMAN)) u_sqrt (
        .x(a), .y(root), .domain_error(de), .dbg_y(), .dbg_lo(), .dbg_exp()
    );
    wire [WR-1:0] want = op ? {root, de} : {q, div0};

    // The expected stream, delayed like the DUT: validity is cleared by reset at every stage, as in the pipeline.
    reg          v_line [0:LATENCY-1];
    reg [WR-1:0] w_line [0:LATENCY-1];
    integer i;
    always @(posedge clk) begin
        v_line[0] <= rst ? 1'b0 : in_valid;
        w_line[0] <= want;
        for (i = 1; i < LATENCY; i = i + 1) begin
            v_line[i] <= rst ? 1'b0 : v_line[i-1];
            w_line[i] <= w_line[i-1];
        end
    end
    // Before the line has filled from the first reset, its contents are unconstrained.
    reg [7:0] age = 8'd0;
    always @(posedge clk) if (age != 8'hff) age <= age + 8'd1;

    always @(posedge clk) begin
        if (age > LATENCY) begin
            assert(dut_valid == v_line[LATENCY-1]);
            if (v_line[LATENCY-1]) assert({dut_y, dut_error} == w_line[LATENCY-1]);
        end
    end
endmodule

`default_nettype wire
