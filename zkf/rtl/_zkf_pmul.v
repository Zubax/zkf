// Shared pipelined integer multiply: p = a * b, exact; the operands are both signed or both unsigned, and p is the raw
// WA+WB product bits.
//
// STAGE_PRODUCT is the number of register stages ahead of the output stage:
//
//   0: a*b.
//   1: the operands are registered first, so that a register can sit at the multiplier's inputs.
//   2: registered operands, then a GA x GB grid of slice products, registered, then their sum.
//   3: as 2, with the sum of each row of the grid registered ahead of the sum of the rows.
//   4: as 3, with the rows summed in pairs and registered ahead of the last sum.
//
// STAGE_OUTPUT=1 registers the result; with 0 it is combinational from the last stage, for a product that its consumer
// registers. Latency: STAGE_PRODUCT + STAGE_OUTPUT.
//
// WMULTIPLIER shapes the grid and has no effect below STAGE_PRODUCT 2:
//
//   0:    a 2x2 grid at STAGE_PRODUCT 2 and a 3x3 grid at 3 and 4, whatever the operand widths.
//   else: the smallest grid whose slices fit a multiplier with WMULTIPLIER-bit operands, e.g. 66x41 -> 4x3 at 18. A
//         slice takes WMULTIPLIER bits of an unsigned operand and one less of a signed one, since every slice but the
//         top one enters a signed product with a zero on top.
//
// A split that leaves an operand less than 4 bits per slice is refused.

`default_nettype none

module _zkf_pmul #(
    parameter WA            = 16,
    parameter WB            = 16,
    parameter SIGNED        = 1,
    parameter WSB           = 1,   // sideband carried beside the product
    parameter WMULTIPLIER   = 0,
    parameter STAGE_PRODUCT = 0,
    parameter STAGE_OUTPUT  = 1
) (
    input  wire             clk,
    input  wire             rst,        // of the valid flags only
    input  wire             in_valid,
    input  wire   [WSB-1:0] sb_in,
    input  wire    [WA-1:0] a,
    input  wire    [WB-1:0] b,
    output wire             out_valid,
    output wire   [WSB-1:0] sb_out,
    output wire [WA+WB-1:0] p
);
    localparam integer WP     = WA + WB;
    localparam integer STAGES = STAGE_PRODUCT + STAGE_OUTPUT;
    localparam integer GRID   = (STAGE_PRODUCT >= 2) ? 1 : 0;
    localparam integer SYM    = (STAGE_PRODUCT >= 3) ? 3 : 2;
    localparam integer P      = (WMULTIPLIER == 0) ? 1 : (WMULTIPLIER - (SIGNED != 0));   // 1: the division is defined
    // verilator coverage_off
    function automatic integer slices(input integer w);   // of a w-bit operand
        begin
            if (GRID == 0) slices = 1;
            else if (WMULTIPLIER != 0) slices = (w + P - 1) / P;
            else slices = SYM;
        end
    endfunction
    // verilator coverage_on
    localparam integer GA     = slices(WA);
    localparam integer GB     = slices(WB);
    // Unsigned slice products on one diagonal of the grid (gi - gj constant) abut, so a diagonal is one operand of a
    // sum by concatenation. Sign-extended ones overlap: they are summed by rows, then the rows.
    localparam integer DIAG   = ((SIGNED == 0) && (STAGE_PRODUCT <= 2)) ? 1 : 0;
    localparam integer ND     = GA + GB - 1;

    generate
        if ((STAGE_PRODUCT < 0) || (STAGE_PRODUCT > 4)) begin : g_invalid_stage_product
            _zkf_invalid_stage_product_out_of_range u_invalid();
        end
        if ((STAGE_OUTPUT != 0) && (STAGE_OUTPUT != 1)) begin : g_invalid_stage_output
            _zkf_invalid_stage_output u_invalid();
        end
        if ((WMULTIPLIER != 0) && (WMULTIPLIER < 8)) begin : g_invalid_wmultiplier
            _zkf_invalid_pmul_wmultiplier u_invalid();
        end
        if (((GA > 1) && (WA < 4*GA)) || ((GB > 1) && (WB < 4*GB))) begin : g_invalid_slice
            _zkf_invalid_pmul_slice_too_narrow u_invalid();
        end
    endgenerate

    // Elaboration-time geometry.
    // verilator coverage_off
    // Slice k of an n-way split of a w-bit operand: the low slices take the remainder.
    function automatic integer slc_w(input integer w, input integer n, input integer k);
        slc_w = (w + n - 1 - k) / n;
    endfunction
    function automatic integer slc_off(input integer w, input integer n, input integer k);
        integer i;
        begin
            slc_off = 0;
            for (i = 0; i < k; i = i + 1) slc_off = slc_off + slc_w(w, n, i);
        end
    endfunction
    // Bits of slice k as a factor: in a signed product every slice but the top one has a zero on top.
    function automatic integer fac_w(input integer w, input integer n, input integer k);
        fac_w = slc_w(w, n, k) + (((SIGNED != 0) && (k != n - 1)) ? 1 : 0);
    endfunction
    localparam integer WXA = fac_w(WA, GA, 0);   // slice 0 is the widest factor
    localparam integer WXB = fac_w(WB, GB, 0);
    localparam integer WPP = WXA + WXB;
    // The product of slices i, j in place: its lowest bit, and the bit above its highest (above its sign).
    function automatic integer pp_lo(input integer i, input integer j);
        pp_lo = slc_off(WA, GA, i) + slc_off(WB, GB, j);
    endfunction
    function automatic integer pp_hi(input integer i, input integer j);
        pp_hi = pp_lo(i, j) + fac_w(WA, GA, i) + fac_w(WB, GB, j);
    endfunction
    // Diagonal d holds the products i - j = d - (GB-1); t counts the diagonals by their lowest bit, the main one first.
    function automatic integer diag_lo(input integer d);
        diag_lo = (d > GB - 1) ? slc_off(WA, GA, d - (GB - 1)) : slc_off(WB, GB, GB - 1 - d);
    endfunction
    function automatic integer diag_hi(input integer d);
        diag_hi = (d < GA) ? pp_hi(d, GB - 1) : pp_hi(GA - 1, GA - 1 - d + GB - 1);
    endfunction
    function automatic integer diag_at(input integer t);
        integer n, up, dn;
        begin
            diag_at = GB - 1;
            up      = GB;
            dn      = GB - 2;
            for (n = 0; n < t; n = n + 1) begin
                if ((dn < 0) || ((up < ND) && (diag_lo(up) <= diag_lo(dn)))) begin
                    diag_at = up;
                    up      = up + 1;
                end else begin
                    diag_at = dn;
                    dn      = dn - 1;
                end
            end
        end
    endfunction
    // Extents of the operands of a sum, 16 bits each. kind 0: the diagonals; 1: the products of row r; 2: the rows.
    function automatic [16*(GA+GB)-1:0] extents(input integer kind, input integer r, input integer top);
        integer t;
        begin
            extents = {(16*(GA+GB)){1'b0}};
            for (t = 0; t < GA + GB; t = t + 1) begin
                case (kind)
                    0:       if (t < ND) extents[16*t +: 16] = top ? diag_hi(diag_at(t)) : diag_lo(diag_at(t));
                    1:       if (t < GB) extents[16*t +: 16] = top ? pp_hi(r, t) : pp_lo(r, t);
                    default: if (t < GA) extents[16*t +: 16] = top ? pp_hi(t, GB - 1) : pp_lo(t, 0);
                endcase
            end
        end
    endfunction
    // verilator coverage_on

    zkf_pipe #(.W(WSB), .N(STAGES)) u_side (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in(sb_in), .out_valid(out_valid), .out(sb_out)
    );

    wire [WA-1:0] c_a;
    wire [WB-1:0] c_b;
    zkf_pipe #(.W(WP), .N(STAGE_PRODUCT != 0)) u_capture (
        .clk(clk), .rst(1'b0), .in_valid(1'b0), .in({a, b}), .out_valid(), .out({c_a, c_b})
    );

    wire [WXA-1:0] a_s [0:GA-1];
    wire [WXB-1:0] b_s [0:GB-1];
    wire [WP-1:0]  pp  [0:GA*GB-1];   // slice products, registered in a grid, extended in place
    wire [WP-1:0]  sum;
    genvar gi, gj;
    generate
        for (gi = 0; gi < GA; gi = gi + 1) begin : g_aslice
            localparam integer WI = slc_w(WA, GA, gi);
            localparam integer OI = slc_off(WA, GA, gi);
            wire               sx  = ((gi == GA - 1) && (SIGNED != 0)) ? c_a[WA-1] : 1'b0;
            wire [WXA+WI-1:0]  ext = {{WXA{sx}}, c_a[OI +: WI]};
            assign a_s[gi] = ext[WXA-1:0];
        end
        for (gj = 0; gj < GB; gj = gj + 1) begin : g_bslice
            localparam integer WJ = slc_w(WB, GB, gj);
            localparam integer OJ = slc_off(WB, GB, gj);
            wire               sx  = ((gj == GB - 1) && (SIGNED != 0)) ? c_b[WB-1] : 1'b0;
            wire [WXB+WJ-1:0]  ext = {{WXB{sx}}, c_b[OJ +: WJ]};
            assign b_s[gj] = ext[WXB-1:0];
        end
        for (gi = 0; gi < GA; gi = gi + 1) begin : g_pp_row
            for (gj = 0; gj < GB; gj = gj + 1) begin : g_pp_col
                localparam integer LO = pp_lo(gi, gj);
                localparam integer WT = pp_hi(gi, gj) - LO;
                wire [WPP-1:0] m, m_q;
                if (SIGNED != 0) begin : g_signed
                    assign m = $signed(a_s[gi]) * $signed(b_s[gj]);
                end else begin : g_unsigned
                    assign m = a_s[gi] * b_s[gj];
                end
                zkf_pipe #(.W(WPP), .N(GRID)) u_pp (
                    .clk(clk), .rst(1'b0), .in_valid(1'b0), .in(m), .out_valid(), .out(m_q)
                );
                wire [WP+WT-1:0] ext = {{WP{(SIGNED != 0) & m_q[WT-1]}}, m_q[WT-1:0]} << LO;
                assign pp[gi*GB + gj] = ext[WP-1:0];
            end
        end

        if (DIAG != 0) begin : g_diagonals
            wire [ND*WP-1:0] operand;
            genvar gt, gd;
            for (gt = 0; gt < ND; gt = gt + 1) begin : g_operand
                localparam integer D = diag_at(gt);
                wire [WP-1:0] cat [0:GA];
                assign cat[0] = {WP{1'b0}};
                for (gd = 0; gd < GA; gd = gd + 1) begin : g_cat
                    if ((gd + GB - 1 - D >= 0) && (gd + GB - 1 - D < GB)) begin : g_on
                        assign cat[gd + 1] = cat[gd] | pp[gd*GB + gd + GB - 1 - D];
                    end else begin : g_off
                        assign cat[gd + 1] = cat[gd];
                    end
                end
                assign operand[gt*WP +: WP] = cat[GA];
            end
            _zkf_pmul_sum #(
                .N(ND), .W(WP), .LO(extents(0, 0, 0)), .HI(extents(0, 0, 1))
            ) u_sum (.clk(clk), .x(operand), .y(sum));
        end else begin : g_rows
            wire [GA*WP-1:0] row;
            for (gi = 0; gi < GA; gi = gi + 1) begin : g_row
                wire [GB*WP-1:0] operand;
                for (gj = 0; gj < GB; gj = gj + 1) begin : g_operand
                    assign operand[gj*WP +: WP] = pp[gi*GB + gj];
                end
                wire [WP-1:0] s;
                _zkf_pmul_sum #(
                    .N(GB), .W(WP), .SIGNED(SIGNED), .HO(pp_hi(gi, GB - 1)),
                    .LO(extents(1, gi, 0)), .HI(extents(1, gi, 1))
                ) u_sum (.clk(clk), .x(operand), .y(s));
                zkf_pipe #(.W(WP), .N(STAGE_PRODUCT >= 3)) u_row (
                    .clk(clk), .rst(1'b0), .in_valid(1'b0), .in(s), .out_valid(), .out(row[gi*WP +: WP])
                );
            end
            _zkf_pmul_sum #(
                .N(GA), .W(WP), .SIGNED(SIGNED), .STAGE(STAGE_PRODUCT == 4),
                .LO(extents(2, 0, 0)), .HI(extents(2, 0, 1))
            ) u_sum (.clk(clk), .x(row), .y(sum));
        end
    endgenerate

    zkf_pipe #(.W(WP), .N(STAGE_OUTPUT)) u_output (
        .clk(clk), .rst(1'b0), .in_valid(1'b0), .in(sum), .out_valid(), .out(p)
    );
endmodule


// The sum of N operands of W bits. Operand k has bits LO[k] up to HI[k] only (16 bits per entry; LO ascending, zero
// only for operand 0), with its sign extension above when SIGNED; the sum fits below bit HO. While more than three
// remain, neighbors are added in pairs, level by level; the rest are added at once. Bits below every operand but the
// lowest are that one's and pass through, and no adder is wider than its extents call for, so that the structure does
// not depend on what a tool infers. STAGE=1 registers the first level of pairs, which then stands whatever N is.
module _zkf_pmul_sum #(
    parameter N      = 1,
    parameter W      = 8,
    parameter SIGNED = 0,
    parameter STAGE  = 0,
    parameter HO     = W,
    parameter [16*N-1:0] LO = 0,
    parameter [16*N-1:0] HI = 0
) (
    input  wire           clk,
    input  wire [N*W-1:0] x,
    output wire   [W-1:0] y
);
    // verilator coverage_off
    // The bit above the highest of the sum of the 2**level operands from `first` on.
    function automatic integer top(input integer first, input integer level);
        integer i;
        begin
            top = 0;
            for (i = first; (i < first + (1 << level)) && (i < N); i = i + 1) begin
                if (HI[16*i +: 16] > top) top = HI[16*i +: 16];
            end
            top = (top + level > HO) ? HO : (top + level);
        end
    endfunction
    // verilator coverage_on
    localparam integer LEVELS = (N > 3) ? $clog2((N + 2) / 3) : STAGE;
    localparam integer R      = (N + (1 << LEVELS) - 1) >> LEVELS;   // operands of the last sum

    wire [W-1:0] node [0:(LEVELS+1)*N-1];
    genvar gl, gk;
    generate
        for (gk = 1; gk < N; gk = gk + 1) begin : g_order
            if ((LO[16*gk +: 16] == 0) || (LO[16*gk +: 16] < LO[16*(gk-1) +: 16])) begin : g_invalid_order
                _zkf_invalid_pmul_sum_order u_invalid();
            end
        end
        for (gk = 0; gk < N; gk = gk + 1) begin : g_leaf
            assign node[gk] = x[gk*W +: W];
        end
        for (gl = 0; gl < LEVELS; gl = gl + 1) begin : g_level
            for (gk = 0; (gk << (gl + 1)) < N; gk = gk + 1) begin : g_node
                localparam integer FB = (2*gk + 1) << gl;    // first operand of the upper member
                wire [W-1:0] s;
                if (FB >= N) begin : g_single
                    assign s = node[gl*N + 2*gk];
                end else begin : g_pair
                    localparam integer L = LO[16*FB +: 16];
                    localparam integer H = top(2*gk << gl, gl + 1);
                    wire [H-L-1:0]   hi = node[gl*N + 2*gk][H-1:L] + node[gl*N + 2*gk + 1][H-1:L];
                    wire [W+H-L-1:0] up = {{W{(SIGNED != 0) & hi[H-L-1]}}, hi};
                    assign s = {up[W-L-1:0], node[gl*N + 2*gk][L-1:0]};
                end
                if ((gl == 0) && (STAGE != 0)) begin : g_registered
                    reg [W-1:0] s_q;
                    always @(posedge clk) s_q <= s;
                    assign node[N + gk] = s_q;
                end else begin : g_comb
                    assign node[(gl + 1)*N + gk] = s;
                end
            end
        end

        if (R == 1) begin : g_one
            assign y = node[LEVELS*N];
        end else begin : g_last
            localparam integer L  = LO[(16 << LEVELS) +: 16];
            localparam integer T0 = top(0, LEVELS);
            localparam integer T1 = top(1 << LEVELS, LEVELS);
            localparam integer T2 = (R > 2) ? top(2 << LEVELS, LEVELS) : 0;
            // The sum starts from an operand that reaches its top: then no partial sum is narrower than the whole.
            localparam integer F  = ((T0 >= T1) && (T0 >= T2)) ? 0 : (T1 >= T2) ? 1 : 2;
            if ((SIGNED == 0) && (T0 != HO) && (T1 != HO) && (T2 != HO)) begin : g_invalid_top
                _zkf_invalid_pmul_sum_top u_invalid();
            end
            wire [HO-L-1:0] hi;
            if (R == 2) begin : g_two
                assign hi = node[LEVELS*N + F][HO-1:L] + node[LEVELS*N + 1 - F][HO-1:L];
            end else begin : g_three
                assign hi = node[LEVELS*N + F][HO-1:L] + node[LEVELS*N + (F + 1) % 3][HO-1:L]
                          + node[LEVELS*N + (F + 2) % 3][HO-1:L];
            end
            wire [W+HO-L-1:0] up = {{W{(SIGNED != 0) & hi[HO-L-1]}}, hi};
            assign y = {up[W-L-1:0], node[LEVELS*N][L-1:0]};
        end
    endgenerate
endmodule

`default_nettype wire
