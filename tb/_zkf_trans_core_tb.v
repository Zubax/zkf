// Self-checking bench for one generated zkf_exp2 / zkf_log2 core: its raw kernel value at every vector that
// tb/zkf_trans_core_check.py computed with zkf._reference.trans_kernel, one argument per cycle with X bubbles.

`timescale 1ns/1ps
`default_nettype none

module _zkf_trans_core_tb;
    parameter WARG   = 30;
    parameter WOUT   = 32;
    parameter N      = 16;
    parameter STAGES = 1;  // of the core after the ROM's register, as the bench script counts them
    parameter SP     = 0;
    parameter SQ     = 0;
    parameter OD     = 0;
    parameter CH     = 0;
    parameter WM     = 0;
    localparam LAT = 1 + STAGES;
`define ZKF_CORE_KNOBS .STAGES(STAGES), .WMULTIPLIER(WM), .STAGE_PRODUCT(SP), .STAGE_PRODUCT_SQUARE(SQ), \
                       .STAGE_PRODUCT_ODD(OD), .STAGE_PRODUCT_CHAIN(CH)

    reg              clk = 1'b0;
    reg              rst = 1'b1;
    reg              in_valid = 1'b0;
    reg  [WARG-1:0]  arg;
    reg  [WARG+WOUT:0] vec [0:N-1];  // {rc, value, argument}
    wire             out_valid;
    wire [WOUT-1:0]  value;
    wire             rc;
    integer i, checked, errors, cyc;
    integer sent_at [0:N-1];

`ifdef ZKF_CORE_LOG2
    `ZKF_CORE #(`ZKF_CORE_KNOBS) dut (
        .clk(clk), .rst(rst), .in_valid(in_valid), .frac(arg), .rc(rc), .out_valid(out_valid), .c(value)
    );
`else
    `ZKF_CORE #(`ZKF_CORE_KNOBS) dut (
        .clk(clk), .rst(rst), .in_valid(in_valid), .f(arg), .out_valid(out_valid), .s(value)
    );
    assign rc = 1'b0;
`endif
`undef ZKF_CORE_KNOBS

    always #5 clk = ~clk;
    always @(posedge clk) cyc <= cyc + 1;
    initial begin
        cyc = 0;
        checked = 0;
        errors = 0;
        $readmemh(`ZKF_VECTORS, vec);
        repeat (4) @(posedge clk);
        rst <= 1'b0;
        i = 0;
        while (i < N) begin
            if (($random & 15) != 0) begin
                in_valid <= 1'b1;
                arg <= vec[i][WARG-1:0];
                sent_at[i] = cyc + 1;
                i = i + 1;
            end else begin
                in_valid <= 1'b0;
                arg <= {WARG{1'bx}};
            end
            @(posedge clk);
        end
        in_valid <= 1'b0;
        repeat (LAT + 4) @(posedge clk);
        if ((errors == 0) && (checked == N)) $display("PASS %0d", checked);
        else $display("FAIL checked=%0d of %0d errors=%0d", checked, N, errors);
        $finish;
    end
    // rc is combinational from the argument last driven.
    always @(negedge clk) if (in_valid && (rc !== vec[i-1][WARG+WOUT])) errors = errors + 1;
    always @(posedge clk) if (!rst && (out_valid === 1'b1)) begin
        if ((value !== vec[checked][WARG +: WOUT]) || ((cyc - sent_at[checked]) != LAT)) begin
            errors = errors + 1;
            if (errors < 4)
                $display("MISMATCH %0d: argument %h value %h expected %h latency %0d", checked, vec[checked][WARG-1:0],
                         value, vec[checked][WARG +: WOUT], cyc - sent_at[checked]);
        end
        checked = checked + 1;
    end
endmodule

`default_nettype wire
