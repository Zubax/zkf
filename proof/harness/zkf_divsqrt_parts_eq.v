// Formal harnesses at production widths for the parts of zkf_divsqrt outside the digit step: the divider's and the
// root's stage 0, and the last digit decision with its rounding select. Inputs are free (anyseq) within each part's
// reachable-state invariant; the expected values are computed arithmetically.

`default_nettype none

module zkf_divsqrt_div0_eq #(parameter WMAN = 18, parameter FOLD = 1) ();
    (* anyseq *) wire [WMAN-1:0] sa;
    (* anyseq *) wire [WMAN-1:0] sb;
    wire            h;
    wire      [1:0] digit;
    wire [WMAN-1:0] rem;
    _zkf_divsqrt_div0 #(.WMAN(WMAN), .FOLD(FOLD)) u_dut (.sa(sa), .sb(sb), .h(h), .digit(digit), .rem(rem));

    localparam WX = WMAN + 6;
    wire [WX-1:0] a = {{(WX-WMAN){1'b0}}, sa};
    wire [WX-1:0] b = {{(WX-WMAN){1'b0}}, sb};
    wire          h_want = a < b;
    wire [WX-1:0] n = h_want ? (a << 1) : a;
    wire    [1:0] d_want = ((n << 2) >= 7 * b) ? 2'd3 : ((n << 2) >= 6 * b) ? 2'd2 : ((n << 2) >= 5 * b) ? 2'd1 : 2'd0;
    wire [WX-1:0] r_want = FOLD ? ((n << 2) - (4 + d_want) * b) : (n - b);
    always @(*) begin
        assume(sa[WMAN-1] && sb[WMAN-1]);
        assert(h == h_want);
        if (FOLD) assert(digit == d_want);
        assert({{(WX-WMAN){1'b0}}, rem} == r_want);
        assert(r_want < b);
    end
endmodule

// The root's state after its first digit, from any radicand: the maximal digit, w = 4*w0 - F_digit, D = 2Q, D3 =
// 2(3Q + u), and the invariant 0 <= w < D + u, with u = 1/4. Frame bit WFRAME is 1.
module zkf_divsqrt_root0_eq #(parameter WMAN = 18) ();
    localparam WFRAC  = WMAN - 1;
    localparam WFRAME = ((WMAN % 2) == 0) ? WMAN : WFRAC;
    localparam WX     = WFRAME + 8;
    (* anyseq *) wire [WFRAC-1:0] frac;
    (* anyseq *) wire             r;
    wire          [1:0] digit;
    wire [WFRAME+1:0] w, d;
    wire [WFRAME+3:0] d3;
    _zkf_divsqrt_root0 #(.WMAN(WMAN), .WFRAME(WFRAME)) u_dut (.frac(frac), .r(r), .digit(digit), .w(w), .d(d), .d3(d3));

    wire [WX-1:0] one = {{(WX-1){1'b0}}, 1'b1} << WFRAME;
    wire [WX-1:0] q   = one >> 2;   // a quarter: u after the first digit
    wire [WX-1:0] m   = ((one + ({{(WX-WFRAC){1'b0}}, frac} << (WFRAME - WFRAC))) << r);
    wire [WX-1:0] m4  = (m - one) << 2;
    wire    [1:0] want = (m4 >= 33 * q) ? 2'd3 : (m4 >= 20 * q) ? 2'd2 : (m4 >= 9 * q) ? 2'd1 : 2'd0;
    wire [WX-1:0] f    = (want == 2'd3) ? 33 * q : (want == 2'd2) ? 20 * q : (want == 2'd1) ? 9 * q : {WX{1'b0}};
    wire [WX-1:0] dq   = 8 * q + {{(WX-2){1'b0}}, want} * (2 * q);
    always @(*) begin
        assert(digit == want);
        assert({{(WX-WFRAME-2){1'b0}}, w} == m4 - f);
        assert({{(WX-WFRAME-2){1'b0}}, d} == dq);
        assert({{(WX-WFRAME-4){1'b0}}, d3} == 3 * dq + 2 * q);
        assert(m4 - f < dq + q);
    end
endmodule

module zkf_divsqrt_last_eq #(parameter WMAN = 18, parameter MODE = 2, parameter FOLD = 1) ();
    localparam WFRAC    = WMAN - 1;
    localparam EVEN     = (WMAN % 2) == 0;
    localparam K        = WMAN / 2;
    localparam NREG     = EVEN ? (K - 1 + (FOLD ? 0 : 1)) : K;
    localparam HAS_ROOT = MODE != 0;
    localparam FW       = (HAS_ROOT && EVEN) ? WMAN : WFRAC;
    localparam WW       = (HAS_ROOT ? 2 : 1) + FW;
    localparam SH       = FW - WFRAC;
    localparam WP       = 2 * NREG + 1;
    localparam DIGITS   = NREG;            // root digits in the prefix (stage 0 + NREG - 1 stages)
    localparam WX       = WW + 8;

    (* anyseq *) wire          sqrt_in;
    (* anyseq *) wire [WW-1:0] w;
    (* anyseq *) wire [WW-1:0] d;
    (* anyseq *) wire [WP-1:0] prefix;
    wire sqrt = (MODE == 2) ? sqrt_in : (MODE == 1);

    wire [WX-1:0] wx  = {{(WX-WW){1'b0}}, w};
    wire [WX-1:0] dx  = {{(WX-WW){1'b0}}, d};
    wire [WX-1:0] u   = {{(WX-1){1'b0}}, 1'b1} << (FW - 2 * DIGITS);   // ulp of the root's prefix
    wire [WX-1:0] d3x = sqrt ? (3 * dx + (u << 1)) : 3 * dx;
    wire [WP-1:0] q_pref  = (MODE == 1) ? d[FW+1 -: WP] : prefix;   // the root's prefix, D = 2Q
    wire [WP-1:0] prefix1 = q_pref + {{(WP-1){1'b0}}, 1'b1};
    wire [WMAN-1:0] significand;
    _zkf_divsqrt_last #(.WMAN(WMAN), .MODE(MODE), .FOLD(FOLD), .WW(WW), .WP(WP)) u_dut (
        .sqrt(sqrt), .w(w), .d(d), .d3(d3x[WW+1:0]), .d5(5 * dx), .d7(7 * dx), .inj({2{sqrt}} | {d3x[0], dx[0]}),
        .prefix((MODE == 1) ? {WP{1'b0}} : prefix), .prefix1(prefix1), .significand(significand)
    );

    // Expected: the truncated digits extended by the last decision, rounded half up (ties never reach it).
    wire [WX-1:0] m4 = wx << 2;
    wire    [1:0] dd = (m4 >= 3 * dx) ? 2'd3 : (m4 >= 2 * dx) ? 2'd2 : (m4 >= dx) ? 2'd1 : 2'd0;
    wire    [2:0] t8 = ((wx << 3) >= 7 * dx) ? 3'd7 : ((wx << 3) >= 6 * dx) ? 3'd6 : ((wx << 3) >= 5 * dx) ? 3'd5 :
                       ((wx << 3) >= 4 * dx) ? 3'd4 : ((wx << 3) >= 3 * dx) ? 3'd3 : ((wx << 3) >= 2 * dx) ? 3'd2 :
                       ((wx << 3) >= dx) ? 3'd1 : 3'd0;
    wire    [1:0] dr = (m4 >= d3x + (u >> 2)) ? 2'd3 : (m4 >= (dx << 1) + u) ? 2'd2 :
                       (m4 >= dx + (u >> 2)) ? 2'd1 : 2'd0;
    wire          gr = m4 >= (dx << 1) + u;
    wire [WMAN:0] q_div  = EVEN ? {prefix[WMAN-2:0], dd} : {prefix[WMAN-3:0], t8};
    wire [WMAN:0] q_root = EVEN ? (FOLD ? {q_pref[WMAN-2:0], dr} : q_pref[WMAN:0]) : {q_pref[WMAN-1:0], gr};
    wire [WMAN:0] r_div  = q_div + 1'b1;
    wire [WMAN:0] r_root = q_root + 1'b1;
    wire [WX-1:0] den = dx >> SH;
    always @(*) begin
        if (!sqrt) begin
            assume(den[WMAN-1] && (den >> WMAN) == 0 && (dx & ((1 << SH) - 1)) == 0);
            assume(wx < dx);
            assert(significand == r_div[WMAN:1]);
        end else begin
            assume(dx >= (u << (2 * DIGITS + 1)) && dx < (u << (2 * DIGITS + 2)));   // Q in [1, 2)
            assume((dx & ((u << 1) - 1)) == 0);
            assume(wx < dx + u);
            if (MODE == 2) assume(prefix == d[FW+1 -: WP]);
            assert(significand == r_root[WMAN:1]);
        end
    end
endmodule

// The production widths: the divider's stage 0 folded (even WMAN) and plain, and the last decision of every build.
module zkf_divsqrt_parts_eq ();
    zkf_divsqrt_div0_eq #(.WMAN(18), .FOLD(1)) u_div0_m18 ();
    zkf_divsqrt_div0_eq #(.WMAN(36), .FOLD(1)) u_div0_m36 ();
    zkf_divsqrt_div0_eq #(.WMAN(18), .FOLD(0)) u_div0_m18_plain ();
    zkf_divsqrt_div0_eq #(.WMAN(53), .FOLD(0)) u_div0_m53_plain ();
    zkf_divsqrt_root0_eq #(.WMAN(4)) u_root0_m4 ();
    zkf_divsqrt_root0_eq #(.WMAN(5)) u_root0_m5 ();
    zkf_divsqrt_root0_eq #(.WMAN(18)) u_root0_m18 ();
    zkf_divsqrt_root0_eq #(.WMAN(27)) u_root0_m27 ();
    zkf_divsqrt_root0_eq #(.WMAN(36)) u_root0_m36 ();
    zkf_divsqrt_root0_eq #(.WMAN(53)) u_root0_m53 ();
    genvar mode;
    generate
        for (mode = 0; mode <= 2; mode = mode + 1) begin : g_mode
            zkf_divsqrt_last_eq #(.WMAN(18), .MODE(mode), .FOLD(1)) u_m18 ();
            zkf_divsqrt_last_eq #(.WMAN(18), .MODE(mode), .FOLD(0)) u_m18_unfolded ();
            zkf_divsqrt_last_eq #(.WMAN(27), .MODE(mode), .FOLD(0)) u_m27 ();
            zkf_divsqrt_last_eq #(.WMAN(36), .MODE(mode), .FOLD(1)) u_m36 ();
            zkf_divsqrt_last_eq #(.WMAN(36), .MODE(mode), .FOLD(0)) u_m36_unfolded ();
            zkf_divsqrt_last_eq #(.WMAN(53), .MODE(mode), .FOLD(0)) u_m53 ();
        end
    endgenerate
endmodule

`default_nettype wire
