// Exactness bench for the shared pipelined multiply _zkf_pmul: a transaction every clock through each configuration
// below, checked cycle by cycle against an independent reference delayed by the configuration's latency
// (STAGE_PRODUCT + STAGE_OUTPUT): the product, the sideband and the valid flag.

`timescale 1ns / 1ps
`default_nettype none

// One device under test and its reference. a_sel / b_sel replace the driven operand with an extreme of the instance's
// own width: 1 = 0111..1, 2 = 1000..0, 3 = all ones. `bad` is high while an output differs from the reference or the
// reference is unknown.
module pmul_check #(
    parameter integer WA = 8, parameter integer WB = 8,
    parameter integer SIGNED = 1, parameter SP = 0,
    parameter integer WMULTIPLIER = 0, parameter SO = 1   // SP and SO untyped: a row may give them one bit wide
) (
    input  wire         clk,
    input  wire         rst,
    input  wire         in_valid,
    input  wire   [1:0] a_sel,
    input  wire   [1:0] b_sel,
    input  wire [127:0] a_drv,
    input  wire [127:0] b_drv,
    output wire         bad
);
    localparam integer WP  = WA + WB;
    localparam integer WSB = 8;
    localparam integer WQ  = 1 + WSB + WP;
    localparam integer L   = SP + SO;
    wire [WA-1:0] a = (a_sel == 2'd0) ? a_drv[WA-1:0] : {a_sel[1], {(WA-1){a_sel[0]}}};
    wire [WB-1:0] b = (b_sel == 2'd0) ? b_drv[WB-1:0] : {b_sel[1], {(WB-1){b_sel[0]}}};
    wire [WSB-1:0] sb_in = a_drv[WSB-1:0] ^ b_drv[127 -: WSB];

    wire           out_valid;
    wire [WSB-1:0] sb_out;
    wire [WP-1:0]  p;
    _zkf_pmul #(.WA(WA), .WB(WB), .SIGNED(SIGNED), .WSB(WSB),
                .STAGE_PRODUCT(SP), .WMULTIPLIER(WMULTIPLIER), .STAGE_OUTPUT(SO)) dut (
        .clk(clk), .rst(rst), .in_valid(in_valid), .sb_in(sb_in),
        .a(a), .b(b), .out_valid(out_valid), .sb_out(sb_out), .p(p));

    wire signed [WA:0] ar  = $signed({(SIGNED != 0) && a[WA-1], a});
    wire signed [WB:0] br  = $signed({(SIGNED != 0) && b[WB-1], b});
    wire [WP-1:0]      rf  = ar * br;
    wire [WQ-1:0]      now = {in_valid, sb_in, rf};
    reg [WQ-1:0] line [0:L];   // line[k]: the reference k cycles late
    integer i;
    always @* line[0] = now;
    always @(posedge clk) for (i = 1; i <= L; i = i + 1) line[i] <= {line[i-1][WQ-1] & ~rst, line[i-1][WQ-2:0]};
    wire [WQ-1:0] want = line[L];
    assign bad = ({out_valid, sb_out, p} !== want) || (^want === 1'bx);
endmodule

module _zkf_pmul_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg          rst      = 1'b1;
    reg          in_valid = 1'b0;
    reg    [1:0] a_sel    = 2'd0;
    reg    [1:0] b_sel    = 2'd0;
    reg  [127:0] a_drv    = 128'd0;
    reg  [127:0] b_drv    = 128'd0;
    integer      fails    = 0;
    integer      cycle    = 0;

    // Widths cover the exact cordic shared-multiply sizes (29x20 = WMAN11, 39x24 = WMAN18, 66x41 = WMAN36) plus
    // asymmetric (13x25), matched (18x18), and the minimum widths of a symmetric split (8 bits by 2, 12 by 3).
    localparam integer N = 85;
    wire [N-1:0] bad;
    // Positional connections keep this coverage table readable; the helper has a fixed, obvious interface.
    `define IO clk, rst, in_valid, a_sel, b_sel, a_drv, b_drv
    // verilog_lint: waive-start module-parameter
    // verilog_lint: waive-start module-port
    genvar gs;
    generate
        for (gs = 0; gs < 4; gs = gs + 1) begin : g_depth
            pmul_check #(29, 20, 1, gs) k0 (`IO, bad[0 + gs]);
            pmul_check #(29, 20, 0, gs) k1 (`IO, bad[4 + gs]);
            pmul_check #(39, 24, 1, gs) k2 (`IO, bad[8 + gs]);
            pmul_check #(18, 18, 0, gs) k3 (`IO, bad[12 + gs]);
            pmul_check #(66, 41, 1, gs) k4 (`IO, bad[16 + gs]);
            pmul_check #(13, 25, 1, gs) k5 (`IO, bad[20 + gs]);
            pmul_check #(12, 12, 1, gs) k6 (`IO, bad[24 + gs]);
        end
    endgenerate
    pmul_check #(12, 12, 0, 3) k28 (`IO, bad[28]);
    pmul_check #(18, 18, 1, 0) k29 (`IO, bad[29]);
    // WMULTIPLIER-derived grids (asymmetric where WA != WB), signed and unsigned, at STAGE_PRODUCT 2 and 3.
    pmul_check #(66, 41, 1, 3, 18) k30 (`IO, bad[30]);  // signed 4x3 (M36 actual)
    pmul_check #(66, 41, 1, 2, 18) k31 (`IO, bad[31]);  // signed 4x3, one sum
    pmul_check #(30, 50, 1, 3, 18) k32 (`IO, bad[32]);  // signed 2x3 (b wider)
    pmul_check #(18, 18, 0, 2, 18) k33 (`IO, bad[33]);  // unsigned 1x1 (single full tile)
    pmul_check #(36, 36, 0, 3, 18) k34 (`IO, bad[34]);  // unsigned 2x2 (full-width slices)
    pmul_check #(40, 20, 0, 3, 18) k35 (`IO, bad[35]);  // unsigned 3x2
    pmul_check #(50, 50, 0, 2, 12) k36 (`IO, bad[36]);  // unsigned 5x5 (small tile, big grid)
    // STAGE_PRODUCT=4: same grid as 3, with the rows summed in pairs and registered ahead of the last sum. Covers
    // both symmetric (3x3) and WMULTIPLIER grids, signed and unsigned; GA in {2,3,4} gives one pair, a pair and a lone
    // row, and two pairs.
    pmul_check #(53, 28, 1, 4, 18) k37 (`IO, bad[37]);  // signed 4x2, GA=4
    pmul_check #(35, 52, 0, 4, 18) k38 (`IO, bad[38]);  // unsigned 2x3, GA=2 (log2 final)
    pmul_check #(66, 41, 1, 4, 18) k39 (`IO, bad[39]);  // signed 4x3, GA=4
    pmul_check #(66, 41, 1, 4, 0)  k40 (`IO, bad[40]);  // signed symmetric 3x3, GA=3 (odd)
    pmul_check #(29, 20, 1, 4, 0)  k41 (`IO, bad[41]);  // signed symmetric, GA=3
    pmul_check #(29, 20, 0, 4, 0)  k42 (`IO, bad[42]);  // unsigned symmetric, GA=3
    pmul_check #(39, 24, 1, 4, 0)  k43 (`IO, bad[43]);  // signed symmetric, GA=3
    pmul_check #(18, 18, 0, 4, 0)  k44 (`IO, bad[44]);  // unsigned symmetric, GA=3
    pmul_check #(12, 12, 1, 4, 0)  k45 (`IO, bad[45]);  // minimum width signed, GA=3
    pmul_check #(40, 20, 0, 4, 18) k46 (`IO, bad[46]);  // unsigned 3x2, GA=3
    // STAGE_PRODUCT=2 on asymmetric unsigned grids: the diagonal concatenation with unequal slice widths.
    pmul_check #(23, 40, 0, 2, 18) k47 (`IO, bad[47]);  // unsigned 2x3 (log2 final, WMAN=24)
    pmul_check #(40, 36, 0, 2, 18) k48 (`IO, bad[48]);  // unsigned 3x2 (exp2 kernel, WMAN=36)
    pmul_check #(43, 49, 0, 2, 18) k49 (`IO, bad[49]);  // unsigned 3x3
    pmul_check #(64, 20, 0, 2, 18) k50 (`IO, bad[50]);  // unsigned 4x2
    pmul_check #(19, 55, 0, 2, 18) k51 (`IO, bad[51]);  // unsigned 2x4
    // STAGE_OUTPUT=0: the result is combinational from the last stage.
    generate
        for (gs = 0; gs < 5; gs = gs + 1) begin : g_no_output
            pmul_check #(29, 20, 1, gs, 0, 0) ks (`IO, bad[52 + gs]);
            pmul_check #(29, 20, 0, gs, 0, 0) ku (`IO, bad[57 + gs]);
        end
    endgenerate
    pmul_check #(66, 41, 1, 3, 18, 0) k62 (`IO, bad[62]);
    pmul_check #(40, 36, 0, 2, 18, 0) k63 (`IO, bad[63]);
    pmul_check #(43, 49, 0, 2, 18, 0) k64 (`IO, bad[64]);
    pmul_check #(36, 36, 1, 3, 18, 0) k65 (`IO, bad[65]);
    // The widest one-sum grids.
    pmul_check #(53, 53, 1, 2, 0, 0)  k66 (`IO, bad[66]);
    pmul_check #(53, 53, 1, 2, 0, 1)  k67 (`IO, bad[67]);
    pmul_check #(53, 53, 1, 2, 18, 0) k68 (`IO, bad[68]);
    pmul_check #(53, 53, 1, 2, 18, 1) k69 (`IO, bad[69]);
    pmul_check #(66, 41, 0, 2, 18, 0) k70 (`IO, bad[70]);
    pmul_check #(66, 41, 0, 2, 18, 1) k71 (`IO, bad[71]);
    pmul_check #(66, 41, 1, 2, 18, 0) k72 (`IO, bad[72]);
    // Sums of more than four rows: small tiles under a signed operand.
    pmul_check #(22, 43, 1, 2, 8)    k73 (`IO, bad[73]);
    pmul_check #(9,  70, 1, 2, 8, 0) k74 (`IO, bad[74]);
    pmul_check #(61, 41, 1, 4, 9)    k75 (`IO, bad[75]);
    pmul_check #(18, 18, 0, 1'b1, 0, 1'b1) k76 (`IO, bad[76]);  // stage counts one bit wide, as a comparison gives
    // Three pair levels (13 diagonals), a registered pair level over ten rows, and a grid of one column.
    pmul_check #(53, 53, 0, 2, 8)  k77 (`IO, bad[77]);
    pmul_check #(66, 41, 1, 4, 8)  k78 (`IO, bad[78]);
    pmul_check #(25, 16, 1, 3, 18) k79 (`IO, bad[79]);
    pmul_check #(12, 12, 0, 4, 0)    k80 (`IO, bad[80]);  // minimum width unsigned, GA=3
    pmul_check #(12, 12, 0, 2, 0)    k81 (`IO, bad[81]);
    pmul_check #(8,  9,  1, 2, 0)    k82 (`IO, bad[82]);  // minimum width, GA=2
    pmul_check #(9,  8,  0, 2, 0, 0) k83 (`IO, bad[83]);
    pmul_check #(11, 8,  0, 2, 0)    k84 (`IO, bad[84]);
    `undef IO
    // verilog_lint: waive-stop module-port
    // verilog_lint: waive-stop module-parameter

    // One clock: the inputs are driven after an edge and the outputs checked ahead of the next. The first cycles are
    // not checked: the deepest pipeline (latency 5) and its reference hold nothing yet.
    task automatic step(input v, input [1:0] sa, input [1:0] sb, input [127:0] av, input [127:0] bv);
        begin
            in_valid = v; a_sel = sa; b_sel = sb; a_drv = av; b_drv = bv;
            @(negedge clk);
            if ((cycle >= 8) && (bad !== {N{1'b0}})) begin
                $display("FAIL cycle=%0d a=%h b=%h sel=%0d,%0d bad=%h", cycle, av, bv, sa, sb, bad);
                fails = fails + 1;
            end
            cycle = cycle + 1;
            @(posedge clk); #1;
        end
    endtask

    integer i, ci, cj, r;
    reg [127:0] corners [0:6];
    initial begin
        corners[0] = {128{1'b0}};
        corners[1] = {{127{1'b0}}, 1'b1};
        corners[2] = {128{1'b1}};
        corners[3] = {64{2'b10}};
        corners[4] = {64{2'b01}};
        corners[5] = {{111{1'b0}}, 17'h1_0000};
        corners[6] = {{125{1'b1}}, 3'b011};      // -5

        repeat (3) @(posedge clk);
        #1 rst = 1'b0;
        for (i = 0; i < 8; i = i + 1)
            step(1'b1, 2'd0, 2'd0, {4{$urandom}}, {4{$urandom}});

        // Every pairing of the extremes at each instance's own width, then of the wide patterns.
        for (ci = 0; ci < 4; ci = ci + 1)
            for (cj = 0; cj < 4; cj = cj + 1)
                step(1'b1, ci[1:0], cj[1:0], corners[1], corners[6]);
        for (ci = 0; ci <= 6; ci = ci + 1)
            for (cj = 0; cj <= 6; cj = cj + 1)
                step(1'b1, 2'd0, 2'd0, corners[ci], corners[cj]);

        // Random transactions with bubbles and extremes mixed in, and a reset of the valid flags now and then.
        for (i = 0; i < 12000; i = i + 1) begin
            r   = $urandom;
            rst = (r[15:6] == 10'd0);
            step(r[1:0] != 2'd0, (r[5:2] < 4'd4) ? r[3:2] : 2'd0, (r[19:16] < 4'd4) ? r[17:16] : 2'd0,
                 {$urandom, $urandom, $urandom, $urandom}, {$urandom, $urandom, $urandom, $urandom});
        end
        rst = 1'b0;
        for (i = 0; i < 8; i = i + 1)
            step(1'b0, 2'd0, 2'd0, 128'd0, 128'd0);

        if (fails != 0) begin
            $display("_zkf_pmul_tb: %0d FAILURE(S)", fails);
            $fatal;
        end
        $display("_zkf_pmul_tb: PASS (every configuration exact and aligned)");
        $finish;
    end
endmodule

// Every operand pair through the narrowest symmetric grid that _zkf_pmul admits, 8x8 as 2x2; the 3x3 one of 12x12 is
// too many for a bench and takes the random transactions above.
module _zkf_pmul_min_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    localparam integer N = 4;
    reg  [16:0]  n   = 17'd0;
    reg          rst = 1'b1;
    wire [N-1:0] bad;
    wire [127:0] a = {120'd0, n[7:0]}, b = {120'd0, n[15:8]};
    // verilog_lint: waive-start module-parameter
    // verilog_lint: waive-start module-port
    `define IO clk, rst, 1'b1, 2'd0, 2'd0, a, b
    pmul_check #(8, 8, 1, 2, 0)    m0 (`IO, bad[0]);
    pmul_check #(8, 8, 0, 2, 0)    m1 (`IO, bad[1]);
    pmul_check #(8, 8, 1, 2, 0, 0) m2 (`IO, bad[2]);
    pmul_check #(8, 8, 0, 2, 0, 0) m3 (`IO, bad[3]);
    `undef IO
    // verilog_lint: waive-stop module-port
    // verilog_lint: waive-stop module-parameter

    always @(negedge clk) begin
        if ((n >= 17'd8) && (bad !== {N{1'b0}})) begin
            $display("FAIL n=%h bad=%b", n, bad);
            $fatal;
        end
    end
    always @(posedge clk) begin
        rst <= 1'b0;
        if (!rst) n <= n + 17'd1;
        if (n == 17'h1_0010) begin
            $display("_zkf_pmul_min_tb: PASS (every operand pair exact)");
            $finish;
        end
    end
endmodule

`default_nettype wire
