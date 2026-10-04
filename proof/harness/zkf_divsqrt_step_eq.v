// Formal harness: one _zkf_divsqrt_step per stage position S of a WMAN / MODE build, from the reachable-state
// invariant of each operation to the next state. Division (sqrt = 0): d = den << SH with den normalized, d3 = 3d,
// 0 <= w < d. Root (sqrt = 1): u = ulp(Q) after S digits, d = 2Q with Q in [1, 2) a multiple of u, d3 = 3d + 2u,
// 0 <= w < d + u. The digit must be the largest whose subtrahend 4w covers, and the state must stay invariant.

`default_nettype none

module zkf_divsqrt_step_check #(parameter WMAN = 18, parameter MODE = 2, parameter S = 0) ();
    (* anyseq *) wire        sqrt_in;
    (* anyseq *) wire [63:0] w_in;
    (* anyseq *) wire [63:0] d_in;
    (* anyseq *) wire [63:0] d3_in;
    localparam WFRAC    = WMAN - 1;
    localparam EVEN     = (WMAN % 2) == 0;
    localparam HAS_ROOT = MODE != 0;
    localparam FW       = (HAS_ROOT && EVEN) ? WMAN : WFRAC;
    localparam WI       = HAS_ROOT ? 2 : 1;
    localparam WW       = WI + FW;
    localparam SH       = FW - WFRAC;
    localparam IQ       = HAS_ROOT ? (FW - 2 * S - 2) : 0;   // as zkf_divsqrt's stage S
    localparam WX       = WW + 6;   // headroom for 4w and the subtrahends

    wire          sqrt = (MODE == 2) ? sqrt_in : (MODE == 1);
    wire [WW-1:0] w  = w_in[WW-1:0];
    wire [WW-1:0] d  = d_in[WW-1:0];
    wire [WW+1:0] d3 = d3_in[WW+1:0];

    wire     [1:0] digit;
    wire  [WW-1:0] w_next, d_next;
    wire  [WW+1:0] d3_next;
    _zkf_divsqrt_step #(.WW(WW), .IQ(IQ), .MODE(MODE)) u_dut (
        .sqrt(sqrt), .w(w), .d(d), .d3(d3), .inj(sqrt ? 3'b111 : {d3[IQ], d[IQ+1], d[IQ]}),
        .digit(digit), .w_next(w_next), .d_next(d_next), .d3_next(d3_next)
    );

    wire [WX-1:0] one = {{(WX-1){1'b0}}, 1'b1} << FW;
    wire [WX-1:0] u   = {{(WX-1){1'b0}}, 1'b1} << (FW - 2 * S);
    wire [WX-1:0] wx  = {{(WX-WW){1'b0}}, w};
    wire [WX-1:0] dx  = {{(WX-WW){1'b0}}, d};
    wire [WX-1:0] d3x = {{(WX-WW-2){1'b0}}, d3};
    wire [WX-1:0] m   = wx << 2;
    wire [WX-1:0] f1  = sqrt ? (dx + (u >> 2)) : dx;
    wire [WX-1:0] f2  = sqrt ? ((dx << 1) + u) : (dx << 1);
    wire [WX-1:0] f3  = sqrt ? (d3x + (u >> 2)) : d3x;
    wire    [1:0] want = (m >= f3) ? 2'd3 : (m >= f2) ? 2'd2 : (m >= f1) ? 2'd1 : 2'd0;
    wire [WX-1:0] fw  = (want == 2'd3) ? f3 : (want == 2'd2) ? f2 : (want == 2'd1) ? f1 : {WX{1'b0}};
    wire [WX-1:0] dn  = dx + ({{(WX-2){1'b0}}, want} * (u >> 1));
    wire [WX-1:0] wn  = {{(WX-WW){1'b0}}, w_next};

    wire [WX-1:0] den = dx >> SH;
    always @(*) begin
        if (!sqrt) begin
            assume(den[WMAN-1] == 1'b1);
            assume((den >> WMAN) == 0);
            assume((dx & ((one >> WFRAC) - 1)) == 0);   // den's LSB sits at frame bit SH
            assume(d3x == 3 * dx);
            assume(wx < dx);
            assert(digit == want);
            assert(wn == m - fw);
            assert(wn < dx);
            assert(d_next == d);
            assert(d3_next == d3);
        end else begin
            assume(dx >= (one << 1) && dx < (one << 2));
            assume((dx & ((u << 1) - 1)) == 0);
            assume(d3x == 3 * dx + (u << 1));
            assume(wx < dx + u);
            assert(digit == want);
            assert(wn == m - fw);
            assert({{(WX-WW){1'b0}}, d_next} == dn);
            assert({{(WX-WW-2){1'b0}}, d3_next} == 3 * dn + (u >> 1));
            assert(wn < dn + (u >> 2));
        end
    end
endmodule

// Every digit stage of one WMAN / MODE build.
module zkf_divsqrt_step_stages #(parameter WMAN = 18, parameter MODE = 2) ();
    genvar s;
    generate
        for (s = 1; s <= WMAN / 2 - 1; s = s + 1) begin : g_s
            zkf_divsqrt_step_check #(.WMAN(WMAN), .MODE(MODE), .S(s)) u_check ();
        end
    endgenerate
endmodule

// The production widths of both WMAN parities, for one MODE.
module zkf_divsqrt_step_eq #(parameter MODE = 2) ();
    zkf_divsqrt_step_stages #(.WMAN(18), .MODE(MODE)) u_m18 ();
    zkf_divsqrt_step_stages #(.WMAN(27), .MODE(MODE)) u_m27 ();
    zkf_divsqrt_step_stages #(.WMAN(36), .MODE(MODE)) u_m36 ();
    zkf_divsqrt_step_stages #(.WMAN(53), .MODE(MODE)) u_m53 ();
endmodule

`default_nettype wire
