// Formal harnesses for the parts of zkf_exp2 and zkf_log2 around their table cores: range reduction, decode and the two
// result assemblies, each with free inputs (anyseq) inside the frame its operator guarantees.

`default_nettype none

// _zkf_exp2_reduce against the plain definition, floor(|x| * 2**WFF) by one shift of the significand, split at the
// binary point, with the fraction complemented for a negative x. Combinational.

module zkf_exp2_reduce_eq #(parameter WEXP = 6, parameter WMAN = 18, parameter WFF = WMAN + 12) ();
    (* anyseq *) wire [WEXP+WMAN-1:0] x;
    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;
    localparam WMAG  = WEXP - 1 + WFF;
    localparam integer BIAS = (1 << (WEXP - 1)) - 1;

    wire            dut_sign, dut_zero, dut_oor;
    wire [WEXP-2:0] dut_ipart;
    wire [WFF-1:0]   dut_f;
    _zkf_exp2_reduce #(.WEXP(WEXP), .WMAN(WMAN), .WFF(WFF)) u_dut (
        .x(x), .sign(dut_sign), .zero(dut_zero), .oor(dut_oor), .ipart(dut_ipart), .f(dut_f)
    );

    // |x| = {1, frac} * 2**(exp - BIAS - WFRAC): at WFF fraction bits the significand shifts by k.
    wire            sign = x[WFULL-1];
    wire [WEXP-1:0] exp  = x[WFULL-2:WFRAC];
    wire signed [WEXP+7:0] k   = $signed({8'b0, exp}) - BIAS + (WFF - WFRAC);
    wire        [WMAG-1:0] sig = {{(WMAG-WMAN){1'b0}}, 1'b1, x[WFRAC-1:0]};
    wire        [WMAG-1:0] mag = k[WEXP+7] ? (sig >> (-k)) : (sig << k);
    wire                   oor = exp >= BIAS + WEXP - 1;

    always @(*) begin
        assert(dut_sign == sign);
        assert(dut_zero == (exp == 0));
        assert(dut_oor == oor);
        if ((exp != 0) && !oor) begin
            assert(dut_ipart == mag[WMAG-1:WFF]);
            assert(dut_f == (mag[WFF-1:0] ^ {WFF{sign}}));
        end
    end
endmodule

// _zkf_exp2_pack against _zkf_pack of v = s - half an ULP, taken as 1 below 2**WCF and as 2 where s reaches
// 2**(WCF+1), rounding half up (its round input held high). Combinational.

module zkf_exp2_pack_eq #(parameter WEXP = 6, parameter WMAN = 18, parameter WCF = WMAN + 12) ();
    (* anyseq *) wire            sign, zero, oor;
    (* anyseq *) wire [WEXP-2:0] ipart;
    (* anyseq *) wire [WCF+1:0]  s;
    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;

    wire [WFULL-1:0] dut_y;
    _zkf_exp2_pack #(.WEXP(WEXP), .WMAN(WMAN), .WCF(WCF)) u_dut (
        .sign(sign), .zero(zero), .oor(oor), .ipart(ipart), .s(s), .y(dut_y)
    );

    // The value is 2**i * v / 2**WCF, i = floor(x): ipart, or -ipart-1 for a negative x; vn is v with its leading one
    // on bit WCF.
    wire [WCF+1:0]       lsb = {{(WCF+1){1'b0}}, 1'b1};
    wire [WCF+1:0]       vr  = s - (lsb << (WCF - WMAN));
    wire [WCF+1:0]       v   = s[WCF+1] ? (lsb << (WCF + 1)) : (vr < (lsb << WCF)) ? (lsb << WCF) : vr;
    wire                 up  = v[WCF+1];
    wire [WCF+1:0]       vn  = up ? (v >> 1) : v;
    wire signed [WEXP:0] i   = sign ? ~{2'b00, ipart} : {2'b00, ipart};
    wire [WFULL-1:0]     ref_y;
    _zkf_pack #(.WEXP(WEXP), .WMAN(WMAN), .WEXP_UNBIASED(WEXP + 1)) u_ref (
        .clk(1'b0), .rst(1'b0), .in_valid(1'b1),
        .sign(1'b0), .force_zero(oor & sign), .force_inf(oor & ~sign),
        .exp_unbiased(i + $signed({1'b0, up})), .significand(vn[WCF -: WMAN]),
        .guard(vn[WCF-WMAN]), .round(1'b1), .sticky(1'b0),
        .out_valid(), .y(ref_y)
    );
    wire [WFULL-1:0] one = {2'b00, {(WEXP-1){1'b1}}, {WFRAC{1'b0}}};

    always @(*) begin
        assume(s[WCF+1] | s[WCF]);
        assume(!s[WCF+1] || (s[WCF -: WMAN] == {WMAN{1'b0}}));
        assert(dut_y == (zero ? one : ref_y));
    end
endmodule

// _zkf_log2_decode against a transliteration of the reference model (zkf._value.Zkf.log2): e, |e|, |f|, the predicted
// leading-one position and the classes, each from its definition. Combinational, or a cycle later with STAGE.

module zkf_log2_decode_eq #(parameter WEXP = 6, parameter WMAN = 18, parameter STAGE = 0) (
    input wire clk
);
    (* anyseq *) wire [WEXP+WMAN:0] x;
    reg  [WEXP+WMAN:0] x_q;
    reg                seen = 1'b0;
    always @(posedge clk) begin
        x_q  <= x;
        seen <= 1'b1;
    end
    wire            rc, sign;
    wire [WEXP-1:0] exp;
    wire [WMAN-2:0] frac;
    assign {rc, sign, exp, frac} = (STAGE != 0) ? x_q : x;
    localparam WFRAC = WMAN - 1;
    localparam WP    = $clog2(WEXP + 1);
    localparam integer BIAS = (1 << (WEXP - 1)) - 1;

    wire [WFRAC-1:0] dut_fmag;
    wire [WP-1:0]    dut_sh, dut_p1;
    wire [WEXP-1:0]  dut_emag;
    wire             dut_ezero, dut_rc, dut_sub, dut_neg, dut_one, dut_inf, dut_domain_error, dut_pole;
    _zkf_log2_decode #(.WEXP(WEXP), .WMAN(WMAN), .STAGE(STAGE)) u_dut (
        .clk(clk), .rst(1'b0), .in_valid(1'b1),  // every operand is one
        .rc(x[WEXP+WMAN]), .sign(x[WEXP+WMAN-1]), .exp(x[WMAN-1 +: WEXP]), .frac(x[WMAN-2:0]),
        .out_valid(), .fmag(dut_fmag), .sh(dut_sh), .p1(dut_p1), .emag(dut_emag), .ezero(dut_ezero), .rc_q(dut_rc),
        .sub(dut_sub), .neg(dut_neg), .one(dut_one), .is_inf(dut_inf), .domain_error(dut_domain_error),
        .pole(dut_pole)
    );

    wire signed [WEXP+1:0] e    = $signed({2'b00, exp}) - BIAS + $signed({1'b0, rc});
    wire        [WEXP+1:0] ea   = e[WEXP+1] ? -e : e;
    wire        [WFRAC:0]  fm   = rc ? ({1'b1, {WFRAC{1'b0}}} - {1'b0, frac}) : {1'b0, frac};
    wire                   sub  = (e != 0) && ((e > 0) == rc);
    wire                   zero = exp == 0;
    wire                   is_inf  = zero || (&exp) || sign;
    reg  signed [WP+1:0] p;    // predicted leading-one position of |e| +- l
    integer i;
    always @(*) begin
        p = 0;
        for (i = 0; i <= WEXP; i = i + 1) if (ea[i]) p = i;
        if (ea == (1 << (WEXP - 1))) p = WEXP - 2;
        else if (sub && (ea == 1) && (fm != 0)) p = -1;
    end

    always @(*) begin
        assume(!x[WEXP+WMAN] || (x[WMAN-2:0] != 0));  // re-centering starts above the first table segment
        if ((STAGE == 0) || seen) begin
            assert(dut_fmag == fm[WFRAC-1:0]);
            assert(!fm[WFRAC]);
            assert(dut_emag == ea[WEXP-1:0]);
            assert(ea[WEXP+1:WEXP] == 2'b00);
            assert(dut_ezero == (e == 0));
            assert(dut_rc == rc);
            assert(dut_sub == sub);
            assert(dut_one == ((e == 0) && (fm == 0)));
            assert(dut_inf == is_inf);
            assert(dut_neg == (is_inf ? (zero || sign) : ((e == 0) ? rc : e[WEXP+1])));
            assert(dut_domain_error == (sign && !zero));
            assert(dut_pole == zero);
            if (e == 0) begin
                assert(dut_sh == 1);
            end else if (!(&exp)) begin  // an infinite x has no position to predict
                assert($signed({2'b00, dut_p1}) == p + 1);
                assert($signed({2'b00, dut_sh}) == p + 1 + $signed({1'b0, rc}));
            end
        end
    end
endmodule

// _zkf_log2_pack against _zkf_pack of the exact sum |e| +- prod, rounding half up (its round input held high). The
// side inputs are free at cycle 0 and prod at cycle STAGE.

module zkf_log2_pack_eq #(
    parameter WEXP = 6, parameter WMAN = 18, parameter WGB = 12, parameter STAGE = 1, parameter STAGE_SELECT = 0
) (
    input wire clk
);
    (* anyseq *) wire [$clog2(WMAN-1)-1:0] lz;
    (* anyseq *) wire [$clog2(WEXP+1)-1:0] p1;
    (* anyseq *) wire           [WEXP-1:0] eal;
    (* anyseq *) wire                      ezero, rc, sub, neg, one, is_inf;
    (* anyseq *) wire                [1:0] sb;
    (* anyseq *) wire       [WMAN+WGB+2:0] prod;
    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;
    localparam WLZ   = $clog2(WFRAC);
    localparam WP    = $clog2(WEXP + 1);
    localparam WS    = WFRAC + WGB + 4;
    localparam WX    = ((WLZ > WEXP) ? WLZ : WEXP) + 3;
    localparam integer LAST    = STAGE + STAGE_SELECT;
    localparam integer CYCLE_W = $clog2(LAST + 2);

    reg [CYCLE_W-1:0] cycle = {CYCLE_W{1'b0}};
    always @(posedge clk) if (cycle <= LAST) cycle <= cycle + 1'b1;

    reg [WLZ-1:0]  sh_lz;
    reg [WP-1:0]   sh_p1;
    reg [WEXP-1:0] sh_eal;
    reg            sh_ezero, sh_rc, sh_sub, sh_neg, sh_one, sh_inf;
    reg [1:0]      sh_sb;
    always @(posedge clk) if (cycle == 0) begin
        sh_lz    <= lz;
        sh_p1    <= p1;
        sh_eal   <= eal;
        sh_ezero <= ezero;
        sh_rc    <= rc;
        sh_sub   <= sub;
        sh_neg   <= neg;
        sh_one   <= one;
        sh_inf   <= is_inf;
        sh_sb    <= sb;
    end

    wire [WFULL-1:0] dut_y;
    wire [1:0]       dut_sb;
    _zkf_log2_pack #(
        .WEXP(WEXP), .WMAN(WMAN), .WGB(WGB), .STAGE(STAGE), .STAGE_SELECT(STAGE_SELECT), .WSB(2)
    ) u_dut (
        .clk(clk), .rst(1'b0), .lz(lz), .p1(p1), .eal(eal), .ezero(ezero), .rc(rc), .sub(sub), .neg(neg), .one(one),
        .is_inf(is_inf), .sb_in(sb), .in_valid(1'b1), .prod(prod), .out_valid(), .y(dut_y), .sb_out(dut_sb)
    );

    // The exact sum, one bit wider than the frame so that a wrap shows.
    wire [WS:0]  e_al  = {1'b0, sh_eal, {(WS-WEXP){1'b0}}};
    wire [WS:0]  v     = sh_sub ? (e_al - {1'b0, prod}) : (e_al + {1'b0, prod});
    wire         top   = v[WS-1];
    wire         frame = ~v[WS] & (top ? ~|v[WS-2:0] : (v[WS-2] | v[WS-3]));
    wire         hi    = v[WS-2];
    wire [WS:0]  vn    = top ? (v >> 1) : (hi ? v : (v << 1));   // v with its leading one on bit WS-2
    wire signed [WX-1:0] e_hi  = sh_ezero ? -$signed({{(WX-WLZ){1'b0}}, sh_lz}) - $signed({{(WX-1){1'b0}}, sh_rc})
                                          : $signed({{(WX-WP){1'b0}}, sh_p1}) - $signed({{(WX-1){1'b0}}, 1'b1});
    wire signed [WX-1:0] e_res = e_hi + $signed({{(WX-1){1'b0}}, top}) - $signed({{(WX-1){1'b0}}, ~hi & ~top});
    wire [WFULL-1:0] ref_y;
    _zkf_pack #(.WEXP(WEXP), .WMAN(WMAN), .WEXP_UNBIASED(WX)) u_ref (
        .clk(1'b0), .rst(1'b0), .in_valid(1'b1),
        .sign(sh_neg), .force_zero(sh_one & ~sh_inf), .force_inf(sh_inf),
        .exp_unbiased(e_res), .significand(vn[WS-2 -: WMAN]),
        .guard(vn[WS-2-WMAN]), .round(1'b1), .sticky(1'b0),
        .out_valid(), .y(ref_y)
    );

    reg [WFULL-1:0] ref_q;
    always @(posedge clk) if (cycle == STAGE) ref_q <= ref_y;
    always @(*) begin
        if (cycle == 0) assume(p1 < WEXP);
        if (cycle == STAGE) assume(sh_inf || sh_one || frame);
        if (cycle == LAST) begin
            assert(dut_y == ((STAGE_SELECT != 0) ? ref_q : ref_y));
            assert(dut_sb == sh_sb);
        end
    end
endmodule

module zkf_trans_parts_eq (input wire clk);
    genvar w;
    generate
        for (w = 3; w <= 11; w = w + 1) begin : g_wexp
            zkf_exp2_reduce_eq #(.WEXP(w), .WMAN(16)) u_reduce ();
            if ((w <= 5) || (w == 8) || (w == 11)) begin : g_decode
                zkf_log2_decode_eq #(.WEXP(w), .WMAN(16)) u_decode (.clk(clk));
            end
        end
    endgenerate
`define ZKF_PARTS(NAME, WE, WM, ST) \
    zkf_exp2_reduce_eq #(.WEXP(WE), .WMAN(WM)) u_reduce_``NAME (); \
    zkf_exp2_pack_eq   #(.WEXP(WE), .WMAN(WM)) u_exp2_pack_``NAME (); \
    zkf_log2_decode_eq #(.WEXP(WE), .WMAN(WM)) u_decode_``NAME (.clk(clk)); \
    zkf_log2_decode_eq #(.WEXP(WE), .WMAN(WM), .STAGE(1)) u_decode_stage_``NAME (.clk(clk)); \
    zkf_log2_pack_eq   #(.WEXP(WE), .WMAN(WM), .STAGE(ST)) u_log2_pack_``NAME (.clk(clk));
    `ZKF_PARTS(w6m18, 6, 18, 2)
    `ZKF_PARTS(w2m16, 2, 16, 1)
    `ZKF_PARTS(w8m36, 8, 36, 3)
    `ZKF_PARTS(w11m53, 11, 53, 3)
    `ZKF_PARTS(w30m16, 30, 16, 1)
    `ZKF_PARTS(w17m16, 17, 16, 1)
`undef ZKF_PARTS
    zkf_log2_pack_eq #(.WEXP(6), .WMAN(18), .STAGE(2), .STAGE_SELECT(1)) u_log2_pack_select (.clk(clk));
endmodule

`default_nettype wire
