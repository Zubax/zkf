// zkf_sincos (MODE 0), zkf_atan2 (MODE 1) and zkf_cordic (MODE 2), ports as there. The algorithms are documented in
// zkf_sincos and zkf_atan2.

`default_nettype none

module _zkf_cordic_unit #(
    parameter WEXP              = 6,
    parameter WMAN              = 18,
    parameter MODE              = 0,
    parameter WMULTIPLIER       = 0,
    parameter UNROLL100         = 100,
    parameter STAGE_INPUT       = 0,
    parameter STAGE_PRODUCT     = 0,
    parameter STAGE_NORMALIZE   = 0,
    parameter STAGE_PACK        = 0,
    parameter STAGE_OUTPUT      = 0,
    parameter PARALLEL          = (MODE != 1) && (UNROLL100 < 100),
    parameter LATENCY_ROTATION  = 0,
    parameter LATENCY_VECTORING = 0
) (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 in_valid,
    output wire                 in_ready,
    input  wire                 vectoring,
    input  wire [WEXP+WMAN-1:0] a,
    input  wire [WEXP+WMAN-1:0] b,
    output wire                 out_valid,
    input  wire                 out_ready,
    output wire [WEXP+WMAN-1:0] r0,
    output wire [WEXP+WMAN-1:0] r1,
    output wire [1:0]           quadrant
);
    localparam integer WFRAC    = WMAN - 1;
    localparam integer WFULL    = WEXP + WMAN;
    localparam integer WE       = WEXP + 1;                     // signed unbiased exponent e = exp - BIAS
    localparam integer BIAS     = (1 << (WEXP - 1)) - 1;
    // MUST match zkf_trig.py; the per-WMAN table checks it.
    localparam integer GUARD_FF = (12 > (WMAN / 2 + 2)) ? 12 : (WMAN / 2 + 2);
    localparam integer GUARD_XY = 6;
    localparam integer GUARD_ZF = 6;
    localparam integer GUARD_Z  = 3;
    localparam integer N        = ((WMAN + 1) / 2) + 2;         // CORDIC iterations (GUARD_ITER_*)
    localparam integer WFF      = WMAN + GUARD_FF;              // reduced fraction frac(x) at scale 2**-WFF
    localparam integer WT       = WFF - 2;                      // quadrant-local coordinate width
    localparam integer XF       = ((3 * WMAN + 1) / 2) + GUARD_XY;  // x/y fractional scale
    localparam integer WX       = XF + 2;                       // signed x/y width
    localparam integer ZF       = WT + 2 + GUARD_ZF;            // angle (turns) fractional scale
    localparam integer WZ       = ZF + GUARD_Z;                 // signed angle width
    localparam integer WCONST2PI  = WMAN + 5;
    localparam integer CONST2PI_S = WMAN + 2;
    localparam integer WINVTAU    = WMAN + 5;
    localparam integer INVTAU_S   = WMAN + 7;
    localparam integer WKINV_MAG  = WMAN + 5;
    localparam integer KINV_S     = WMAN + 5;

    // Rotation's multiply keeps the top WPHI bits of phi and the top WXC bits of x_K/y_K.
    localparam integer TSA_BITS = (WT + 2) - ((WMAN + 1) / 2) - 3;  // small-angle handoff: t' < 2**TSA_BITS
    localparam integer WOP      = (WMAN > (TSA_BITS + 1)) ? WMAN : (TSA_BITS + 1);  // small-angle operand width
    localparam integer WPHI_NAT = XF - N + 2;
    localparam integer WPHI     = (WMAN + 6 < WPHI_NAT) ? (WMAN + 6) : ((WPHI_NAT > 2) ? WPHI_NAT : 2);
    localparam integer WXC      = WMAN + 6;
    localparam integer WCP      = ((WOP > (ZF - N)) ? WOP : (ZF - N)) + 2;
    localparam integer WA_ROT   = ((WCONST2PI + 1) > WXC) ? (WCONST2PI + 1) : WXC;
    localparam integer WB_ROT   = (WCP > WPHI) ? WCP : WPHI;
    localparam integer WMAG_ROT = WCONST2PI + WT + 1;           // the small-angle full product is the widest
    localparam integer WEU_ROT  = $clog2(WMAG_ROT + 1) + 2;

    localparam integer STEPS    = (XF + 1) / 2;
    localparam integer F        = 2 * STEPS;
    localparam integer WQUO     = F + 1;                        // the bypass quotient of significands can exceed 1
    localparam integer WA_MAG   = WX - 1;                       // x_K unsigned: finite vectoring outputs are > 0
    localparam integer WA_MUL   = (WA_MAG > WQUO) ? WA_MAG : WQUO;
    localparam integer WMAG_AB  = ((WA_MAG + WKINV_MAG) > (WQUO + WINVTAU)) ? (WA_MAG + WKINV_MAG) : (WQUO + WINVTAU);
    localparam integer WMAG_VEC = (WMAG_AB > (WZ + 2)) ? WMAG_AB : (WZ + 2);

    // MODE 2 takes the union: a signed multiplier (vectoring's unsigned operands +1 bit), a biased-exponent packer.
    localparam integer WA_PM   = (MODE == 0) ? WA_ROT : (MODE == 1) ? WA_MUL
                               : ((WA_ROT > (WA_MUL + 1)) ? WA_ROT : (WA_MUL + 1));
    localparam integer WB_PM   = (MODE == 0) ? WB_ROT : (MODE == 1) ? WKINV_MAG
                               : ((WB_ROT > (WKINV_MAG + 1)) ? WB_ROT : (WKINV_MAG + 1));
    localparam integer WSB_PM  = (MODE == 1) ? 1 : 2;
    localparam integer WMAG_PK = (MODE == 0) ? WMAG_ROT : (MODE == 1) ? WMAG_VEC
                               : ((WMAG_ROT > WMAG_VEC) ? WMAG_ROT : WMAG_VEC);
    localparam integer WEU_PK  = (MODE != 0) ? (WEXP + $clog2(WMAG_PK + 1) + 3)
                               : ((WEU_ROT > (WEXP + 3)) ? WEU_ROT : (WEXP + 3));
    localparam integer WSB_PK  = (MODE == 1) ? 1 : 3;

    localparam integer XYCYC  = (N * 100 + UNROLL100 - 1) / UNROLL100;
    // Rotation's decoupled z-path overlaps the PHI product with the CORDIC.
    localparam integer SAVED  = (PARALLEL == 0) ? 0
                              : (((XYCYC - N) < (1 + STAGE_PRODUCT)) ? (XYCYC - N) : (1 + STAGE_PRODUCT));
    localparam integer STAGES = STAGE_INPUT + STAGE_NORMALIZE + STAGE_PACK + STAGE_OUTPUT;
    localparam LATENCY_ROTATION_REF  = 11 + (2 * STAGE_PRODUCT) + XYCYC - SAVED + STAGES;
    localparam LATENCY_VECTORING_REF = 7 + XYCYC + (STEPS + 1) + STAGE_PRODUCT + STAGES;
    generate
        // Vectoring: turn8's octant constants are normal only while BIAS-3 >= 1, and at WEXP 2 theta's codomain sits
        // below min_normal; WEXP 4 is out of scope. BIAS's unsized integer shift caps WEXP at 30, and nothing
        // downstream refuses more.
        if (((MODE != 0) && (WEXP < 5)) || (WEXP >= 31)) begin : g_invalid_wexp_or_wman
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        if (((LATENCY_ROTATION != 0) && (LATENCY_ROTATION != LATENCY_ROTATION_REF)) ||
            ((LATENCY_VECTORING != 0) && (LATENCY_VECTORING != LATENCY_VECTORING_REF))) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    wire             accept, res_valid;
    wire [WFULL-1:0] res_r0, res_r1;
    wire [1:0]       res_quad;
    _zkf_txn #(.W(2 * WFULL + 2), .STAGE_OUTPUT(STAGE_OUTPUT)) u_txn (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in_ready(in_ready), .accept(accept),
        .res_valid(res_valid), .res({res_r0, res_r1, res_quad}),
        .out_valid(out_valid), .out_ready(out_ready), .out({r0, r1, quadrant})
    );

    // The datapaths' tag spaces overlap, so each drives and sees the shared resources only in its own transactions.
    reg  vec_r;
    wire own_vec = (MODE == 2) ? vec_r : (MODE == 1);
    always @(posedge clk) begin
        if (accept) vec_r <= vectoring;
    end

    wire             si_valid, si_vec;
    wire [WFULL-1:0] si_a, si_b;
    zkf_pipe #(.W(2 * WFULL + 1), .N(STAGE_INPUT)) u_input_pipe (
        .clk(clk), .rst(rst), .in_valid(accept), .in({(MODE == 2) ? vectoring : (MODE == 1), a, b}),
        .out_valid(si_valid), .out({si_vec, si_a, si_b})
    );

    wire                 rot_start, rot_pm_iv, rot_pk_iv, rot_pk_sign, rot_pk_inf, rot_res;
    wire [WZ-1:0]        rot_z0;
    wire                 vec_start, vec_pm_iv, vec_pk_iv, vec_pk_sign, vec_res;
    wire [WX-1:0]        vec_x0, vec_y0;
    wire [WSB_PM-1:0]    rot_pm_sb, vec_pm_sb;
    wire [WA_PM-1:0]     rot_pm_a, vec_pm_a;
    wire [WB_PM-1:0]     rot_pm_b, vec_pm_b;
    wire [WEU_PK-1:0]    rot_pk_exp, vec_pk_exp;
    wire [WMAG_PK-1:0]   rot_pk_mag, vec_pk_mag;
    wire [WSB_PK-1:0]    rot_pk_sb, vec_pk_sb;
    wire [WFULL-1:0]     rot_sin, rot_cos, vec_theta, vec_mag;
    wire [1:0]           rot_quad;

    wire                 ce_done, ce_zdone, pm_valid, pk_valid;
    wire signed [WX-1:0] ce_xn, ce_yn;
    wire signed [WZ-1:0] ce_zn;
    wire [WCONST2PI-1:0] ce_const2pi;
    wire [WINVTAU-1:0]   ce_inv_tau;
    wire [WKINV_MAG-1:0] ce_kinv_mag;
    wire [WSB_PM-1:0]    pm_sb;
    wire [WA_PM+WB_PM-1:0] pm_p;
    wire [WFULL-1:0]     pk_y;
    wire [WSB_PK-1:0]    pk_sb;

    generate
        if (MODE != 1) begin : g_rot
            localparam integer WP         = WA_ROT + WB_ROT;
            // Magnitudes are at scale 2**-XF (the corrections and the exact +1) or 2**-CONST2PI_S (the const2pi
            // products); the packer adds the bias itself only at MODE 0.
            localparam integer BIAS_F     = (MODE == 0) ? 0 : BIAS;
            localparam integer EONE_XF    = WMAG_PK - 1 - XF + BIAS_F;
            localparam integer EONE_S     = WMAG_PK - 1 - CONST2PI_S + BIAS_F;
            localparam integer PHI_TRUNC  = (WPHI_NAT - WPHI > 0) ? (WPHI_NAT - WPHI) : 0;
            localparam integer PHI_S      = XF - PHI_TRUNC;     // scale of the narrowed phi
            localparam integer XK_TRUNC   = WX - WXC;
            localparam integer CORR_SHIFT = XF - XK_TRUNC - PHI_TRUNC;
            localparam integer SH_BASE    = BIAS - GUARD_FF - 1;
            localparam integer WLSH       = $clog2(WFF + 1);
            localparam integer WSH        = $clog2((1 << (WEXP - 1)) + GUARD_FF + 2) + 2;
            localparam signed [WSH-1:0] SH_BASE_S = SH_BASE[WSH-1:0];
            localparam signed [WSH:0]   SH_HI_S   = SH_BASE + WFF;

            // The shift-amount clamps compare exp against constants, in parallel with the subtract, not after it.
            wire [WEXP-1:0]       e0   = si_a[WFULL-2:WFRAC];
            wire signed [WSH-1:0] sh0  = $signed({{(WSH-WEXP){1'b0}}, e0}) - SH_BASE_S;
            wire                  e_lo = $signed({1'b0, e0}) < SH_BASE_S;
            reg              r1_valid;
            reg              r1_sign, r1_is_inf, r1_tiny;
            reg [WMAN-1:0]   r1_sig;
            reg [WLSH-1:0]   r1_lshamt;
            reg signed [WE-1:0] r1_e;
            always @(posedge clk) begin
                if (rst) r1_valid <= 1'b0;
                else     r1_valid <= si_valid & ~si_vec;
                r1_sign    <= si_a[WFULL-1];
                r1_is_inf  <= &e0;
                r1_tiny    <= e_lo;
                r1_sig     <= (~|e0) ? {WMAN{1'b0}} : {1'b1, si_a[WFRAC-1:0]};
                casez ({e_lo, $signed({1'b0, e0}) > SH_HI_S})
                    2'b1?:   r1_lshamt <= {WLSH{1'b0}};
                    2'b01:   r1_lshamt <= WFF[WLSH-1:0];
                    default: r1_lshamt <= sh0[WLSH-1:0];
                endcase
                r1_e       <= $signed({1'b0, e0}) - $signed(BIAS[WE-1:0]);
            end

            wire [WFF-1:0] frac_pos = {{(WFF-WMAN){1'b0}}, r1_sig} << r1_lshamt;
            reg              r2_valid;
            reg              r2_sign, r2_is_inf, r2_tiny;
            reg [1:0]        r2_quad;
            reg [WT-1:0]     r2_t;
            reg signed [WE-1:0] r2_e;
            always @(posedge clk) begin
                if (rst) r2_valid <= 1'b0;
                else     r2_valid <= r1_valid;
                r2_sign   <= r1_sign;
                r2_is_inf <= r1_is_inf;
                r2_tiny   <= r1_tiny;
                r2_quad   <= r1_is_inf ? 2'b00 : frac_pos[WFF-1:WFF-2];
                r2_t      <= frac_pos[WT-1:0];
                r2_e      <= r1_e;
            end

            // The complement's +1 keeps the small reflected sine accurate near the fold boundary; its WT-wide carry
            // chain is the front end's longest, hence the fold register. tzero is reduced here, off the shift's cone.
            wire                 oct_flip_c = r2_t > {1'b1, {(WT-1){1'b0}}};
            reg                  fr_valid, fr_octflip, fr_tiny, fr_tzero, fr_sign, fr_inf;
            reg [WT-1:0]         fr_tpw;
            reg [1:0]            fr_quad;
            reg signed [WE-1:0]  fr_e;
            always @(posedge clk) begin
                if (rst) fr_valid <= 1'b0;
                else     fr_valid <= r2_valid;
                fr_octflip <= oct_flip_c; fr_tpw <= oct_flip_c ? (~r2_t + 1'b1) : r2_t; fr_tiny <= r2_tiny;
                fr_tzero <= ~|r2_t; fr_sign <= r2_sign; fr_inf <= r2_is_inf; fr_quad <= r2_quad;
                fr_e <= r2_e;
            end

            assign rot_start = fr_valid;
            assign rot_z0    = {{(WZ-WT-GUARD_ZF){1'b0}}, fr_tpw, {GUARD_ZF{1'b0}}};
            // The transaction's metadata, held for the back end. A slice reduction, not `fr_tpw < (1 << TSA_BITS)`:
            // the unsized shift overflows once TSA_BITS >= 32.
            reg signed [WE-1:0]  e_o;
            reg [1:0]            quad_o;
            reg                  swap_o, tzero_o, tiny_o, sa_o, inf_o, sign_o;
            always @(posedge clk) begin
                if (fr_valid) begin
                    {e_o, quad_o, swap_o, tzero_o, tiny_o, inf_o, sign_o} <=
                        {fr_e, fr_quad, fr_quad[0] ^ fr_octflip, fr_tzero, fr_tiny, fr_inf, fr_sign};
                    sa_o <= fr_tiny | (~|fr_tpw[WT-1:TSA_BITS]);
                end
            end
            wire cd_done  = ce_done & ~own_vec;
            wire cd_zdone = ce_zdone & ~own_vec;

            // BYP is issued at rot_start, while the multiplier idles.
            // S waits in P_S for this transaction's PHI product, which the decoupled engine returns during the CORDIC.
            localparam [1:0] P_IDLE = 2'd0, P_S = 2'd1, P_C = 2'd2;
            localparam [1:0] BYP_TAG = 2'd0, PHI_TAG = 2'd1, S_TAG = 2'd2, C_TAG = 2'd3;

            reg [1:0]            mphase;
            reg signed [WP-1:0]  tprod_r;
            reg signed [WP-1:0]  corr_s_r;
            reg [WMAG_ROT-1:0]   bypass_mag_r;
            reg                  phi_seen;
            reg signed [WX-1:0]  e_xn, e_yn;

            wire signed [WCP-1:0] cphi_op   = $signed(ce_zn[WCP-1:0]);
            wire signed [WPHI-1:0] phi      = tprod_r >>> ((CONST2PI_S + ZF) - PHI_S);
            wire signed [WXC-1:0]  xc       = e_xn >>> XK_TRUNC;
            wire signed [WXC-1:0]  yc       = e_yn >>> XK_TRUNC;
            // rot_start and cd_zdone never coincide.
            assign rot_pm_iv = ((mphase == P_IDLE) && (rot_start | cd_zdone)) | ((mphase == P_S) && phi_seen)
                             | (mphase == P_C);
            assign rot_pm_sb = (mphase != P_IDLE) ? ((mphase == P_C) ? C_TAG : S_TAG)
                             : cd_zdone           ? PHI_TAG : BYP_TAG;
            assign rot_pm_a  = (mphase == P_IDLE) ? $signed({{(WA_ROT-WCONST2PI){1'b0}}, ce_const2pi})
                             : (mphase == P_C)    ? $signed({{(WA_ROT-WXC){yc[WXC-1]}}, yc})
                             :                      $signed({{(WA_ROT-WXC){xc[WXC-1]}}, xc});
            assign rot_pm_b  = (mphase != P_IDLE) ? $signed({{(WB_ROT-WPHI){phi[WPHI-1]}}, phi})
                             : cd_zdone           ? $signed({{(WB_ROT-WCP){cphi_op[WCP-1]}}, cphi_op})
                             :                      $signed({{(WB_ROT-WOP){1'b0}}, fr_tpw[WOP-1:0]});
            wire                  mul_valid = pm_valid & ~own_vec;
            wire signed [WP-1:0]  pmul_p    = pm_p[WP-1:0];    // exact: the product fits its native width

            reg signed [WX-1:0]  b2_sin, b2_cos;
            reg                  b2_valid;
            always @(posedge clk) begin
                if (rst) begin
                    mphase   <= P_IDLE;
                    b2_valid <= 1'b0;
                    phi_seen <= 1'b0;
                end else begin
                    b2_valid <= mul_valid && (pm_sb == C_TAG);
                    if (rot_start) phi_seen <= 1'b0;
                    else if (mul_valid && (pm_sb == PHI_TAG)) phi_seen <= 1'b1;
                    case (mphase)
                        P_IDLE:  if (cd_done) mphase <= P_S;
                        P_S:     if (phi_seen) mphase <= P_C;
                        default: mphase <= P_IDLE;
                    endcase
                end
            end
            always @(posedge clk) begin
                if (cd_done) begin
                    e_xn <= ce_xn; e_yn <= ce_yn;
                end
                if (mul_valid) begin
                    case (pm_sb)
                        BYP_TAG: bypass_mag_r <= pmul_p;   // non-negative
                        PHI_TAG: tprod_r      <= pmul_p;
                        S_TAG:   corr_s_r     <= pmul_p;
                        default: ;
                    endcase
                end
                if (mul_valid && (pm_sb == C_TAG)) begin
                    b2_sin <= e_yn + (corr_s_r >>> CORR_SHIFT);
                    b2_cos <= e_xn - (pmul_p   >>> CORR_SHIFT);
                end
            end

            localparam signed [WEU_PK-1:0] EONE_XF_S    = EONE_XF;
            localparam signed [WEU_PK-1:0] EONE_S_WFRAC = EONE_S - WFRAC;     // tiny bypass: const2pi*sig
            localparam signed [WEU_PK-1:0] EONE_S_ZFT   = EONE_S - (WT + 2);  // TSA bypass: const2pi*t'
            wire [WMAG_ROT-1:0] sin_tp_mag = sa_o ? bypass_mag_r : {{(WMAG_ROT-XF-1){1'b0}}, b2_sin[XF:0]};
            wire [WMAG_ROT-1:0] cos_tp_mag = sa_o ? {{(WMAG_ROT-XF-1){1'b0}}, 1'b1, {XF{1'b0}}}
                                                  : {{(WMAG_ROT-XF-1){1'b0}}, b2_cos[XF:0]};
            reg signed [WEU_PK-1:0]  sin_tp_exp;
            always @* begin
                casez ({sa_o, tiny_o})
                    2'b0?:   sin_tp_exp = EONE_XF_S;
                    2'b11:   sin_tp_exp = e_o + EONE_S_WFRAC;
                    default: sin_tp_exp = EONE_S_ZFT;
                endcase
            end

            // SIN, then COS a cycle later from the same b2 state: one payload bank serves both.
            reg                      sh_valid, sh_is_cos, sh_sgn;
            reg [WMAG_ROT-1:0]       sh_mag;
            reg signed [WEU_PK-1:0]  sh_exp;
            reg [1:0]                sh_quad;
            always @(posedge clk) begin
                if (rst) sh_valid <= 1'b0;
                else     sh_valid <= b2_valid | (sh_valid & ~sh_is_cos);

                if (b2_valid) begin
                    sh_is_cos <= 1'b0;
                    sh_sgn    <= quad_o[1] ^ sign_o;
                    sh_mag    <= swap_o ? cos_tp_mag : sin_tp_mag;
                    sh_exp    <= swap_o ? EONE_XF_S  : sin_tp_exp;
                    casez ({inf_o, sign_o, tzero_o})
                        3'b1??:  sh_quad <= 2'b00;
                        3'b00?:  sh_quad <= quad_o;
                        3'b011:  sh_quad <= 2'd0 - quad_o;
                        default: sh_quad <= 2'd3 - quad_o;
                    endcase
                end else if (sh_valid && !sh_is_cos) begin
                    sh_is_cos <= 1'b1;
                    sh_sgn    <= inf_o ? sign_o : (quad_o[1] ^ quad_o[0]);
                    sh_mag    <= swap_o ? sin_tp_mag : cos_tp_mag;
                    sh_exp    <= swap_o ? sin_tp_exp : EONE_XF_S;
                end
            end

`ifdef SIMULATION
            always @(posedge clk) begin
                if (!rst && b2_valid && sh_valid)
                    $fatal(1, "%m: shared back-end collision -- new payload before prior pair issued");
            end
`endif

            assign rot_pk_iv   = sh_valid;
            assign rot_pk_sign = sh_sgn;
            assign rot_pk_inf  = inf_o;
            assign rot_pk_exp  = sh_exp;
            assign rot_pk_mag  = sh_mag;                // zero-extends at MODE 2; the offsets read WMAG_PK
            assign rot_pk_sb   = {sh_is_cos, sh_quad};
            wire             be_ov = pk_valid & ~own_vec;
            reg [WFULL-1:0]  sin_num_r;
            always @(posedge clk) begin
                if (be_ov && !pk_sb[2]) sin_num_r <= pk_y;
            end

            assign rot_res  = be_ov & pk_sb[2];
            assign rot_sin  = sin_num_r;
            assign rot_cos  = pk_y;
            assign rot_quad = pk_sb[1:0];
        end else begin : g_no_rot
            assign {rot_start, rot_z0, rot_pm_iv, rot_pm_sb, rot_pm_a, rot_pm_b, rot_pk_iv, rot_pk_sign,
                    rot_pk_inf, rot_pk_exp, rot_pk_mag, rot_pk_sb, rot_res, rot_sin, rot_cos, rot_quad} = 0;
        end

        if (MODE != 0) begin : g_vec
            localparam integer GUARD_DIV = 8;                  // zkf_trig.py's GUARD_DIV
            localparam integer WCNT      = $clog2(STEPS + 1);
            localparam integer WPMUL     = WA_MUL + WKINV_MAG;
            localparam signed [WZ+1:0] QUARTER = {{(WZ+2-(ZF-1)){1'b0}}, 1'b1, {(ZF-2){1'b0}}};
            localparam signed [WZ+1:0] HALF    = {{(WZ+2-ZF){1'b0}}, 1'b1, {(ZF-1){1'b0}}};
            localparam [WEXP-1:0] TURN8_X1 = BIAS - 3;
            localparam [WEXP-1:0] TURN8_X2 = BIAS - 2;
            localparam [WEXP-1:0] TURN8_X4 = BIAS - 1;

            // |y| > |x| from half-width compares of {exp, frac}, whose unsigned order is the magnitude order.
            localparam integer WK  = WFULL - 1;
            localparam integer WKL = WK / 2;
            localparam integer WKH = WK - WKL;
            wire [WK-1:0]    f0_keyx = si_b[WK-1:0];
            wire [WK-1:0]    f0_keyy = si_a[WK-1:0];
            reg              d0_valid, d0_swap;
            reg [WFULL-1:0]  d0_y, d0_x;
            reg              d0_xz, d0_xi, d0_yz, d0_yi;
            wire             d0_sx = d0_x[WFULL-1];
            wire             d0_sy = d0_y[WFULL-1];
            // D0 loads only its own transaction, so D, a function of it, holds the metadata until the next one.
            always @(posedge clk) begin
                if (rst) d0_valid <= 1'b0; else d0_valid <= si_valid & si_vec;
                if (si_valid & si_vec) begin
                    d0_swap <= (f0_keyy[WK-1 -: WKH] > f0_keyx[WK-1 -: WKH]) |
                               ((f0_keyy[WK-1 -: WKH] == f0_keyx[WK-1 -: WKH]) & (f0_keyy[WKL-1:0] > f0_keyx[WKL-1:0]));
                    d0_y <= si_a; d0_x <= si_b;
                    d0_xz <= ~|si_b[WFULL-2:WFRAC]; d0_xi <= &si_b[WFULL-2:WFRAC];
                    d0_yz <= ~|si_a[WFULL-2:WFRAC]; d0_yi <= &si_a[WFULL-2:WFRAC];
                end
            end

            wire [WMAN-1:0]  f1_sigx    = {1'b1, d0_x[WFRAC-1:0]};
            wire [WMAN-1:0]  f1_sigy    = {1'b1, d0_y[WFRAC-1:0]};
            wire             f1_special = d0_xz | d0_xi | d0_yz | d0_yi;
            wire [WMAN-1:0]  f1_den_sig = d0_swap ? f1_sigy : f1_sigx;
            wire [WMAN-1:0]  f1_num_sig = d0_swap ? f1_sigx : f1_sigy;
            wire [WEXP-1:0]  f1_xe      = d0_x[WFULL-2:WFRAC];
            wire [WEXP-1:0]  f1_ye      = d0_y[WFULL-2:WFRAC];
            wire [WEXP-1:0]  f1_den_exp = d0_swap ? f1_ye : f1_xe;   // biased: so is the vectoring packer's exponent
            // Both orderings of the exponent difference and their clamps form in parallel, so swap only selects late.
            localparam integer WSH  = $clog2(WX + XF + 1);
            localparam integer WCMP = (((WE + 1) > (WSH + 1)) ? (WE + 1) : (WSH + 1)) + 1;
            localparam signed [WCMP-1:0] CLAMP_C = WX + XF;
            localparam signed [WCMP-1:0] TINY_C  = ZF - WMAN - GUARD_DIV;
            localparam [WSH-1:0] SHCLAMP = WX + XF;
            wire [WCMP-1:0]  f1_xe_c     = {{(WCMP - WEXP){1'b0}}, f1_xe};
            wire [WCMP-1:0]  f1_ye_c     = {{(WCMP - WEXP){1'b0}}, f1_ye};
            wire [WCMP-1:0]  f1_shift_xy = f1_xe_c - f1_ye_c;
            wire [WCMP-1:0]  f1_shift_yx = f1_ye_c - f1_xe_c;
            wire [WSH-1:0]   f1_shamt_xy = (f1_shift_xy > CLAMP_C) ? SHCLAMP : f1_shift_xy[WSH-1:0];
            wire [WSH-1:0]   f1_shamt_yx = (f1_shift_yx > CLAMP_C) ? SHCLAMP : f1_shift_yx[WSH-1:0];
            // The bypass initial remainder, precomputed off the divider's arm cone.
            wire [WMAN:0]    f1_byp_diff = {1'b0, f1_num_sig} - {1'b0, f1_den_sig};
            wire             f1_byp_ibit = ~f1_byp_diff[WMAN];

            // Axis specials: the nonzero operand is the larger, so the magnitude is the ordered denominator. turn8
            // waits for the output, off the D-register cone.
            reg                  d_valid;
            reg [WMAN-1:0]       d_den_sig, d_num_sig;
            reg [WSH-1:0]        d_shamt;
            reg                  d_swap, d_sx, d_sy, d_special, d_bypass;
            reg signed [WE-1:0]  d_eden;
            reg signed [WE:0]    d_eydiff;
            reg [WFULL-1:0]      d_sp_mag;
            reg [2:0]            d_spk;
            reg                  d_sp_sign;
            reg [WMAN-1:0]       d_byp_irem;
            reg                  d_byp_ibit;
            always @(posedge clk) begin
                if (rst) d_valid <= 1'b0; else d_valid <= d0_valid;
                d_den_sig <= f1_den_sig; d_num_sig <= f1_num_sig; d_shamt <= d0_swap ? f1_shamt_yx : f1_shamt_xy;
                d_swap <= d0_swap; d_sx <= d0_sx; d_sy <= d0_sy; d_special <= f1_special;
                d_bypass <= ~d0_swap & ~d0_sx & ~f1_special & (f1_shift_xy > TINY_C);
                d_eden <= $signed({1'b0, f1_den_exp}); d_eydiff <= $signed({2'b00, f1_ye}) - $signed({2'b00, f1_xe});
                d_sp_sign <= d0_yz ? 1'b0 : d0_sy;
                d_sp_mag <= {1'b0, f1_den_exp, f1_den_sig[WFRAC-1:0]};
                casez ({d0_xi, d0_yi, d0_xz, d0_yz})
                    4'b11??: begin d_spk <= d0_sx ? 3'd3 : 3'd1; d_sp_mag <= {1'b0, {WEXP{1'b1}}, {WFRAC{1'b0}}}; end
                    4'b01??: begin d_spk <= 3'd2;                d_sp_mag <= {1'b0, {WEXP{1'b1}}, {WFRAC{1'b0}}}; end
                    4'b10??: begin d_spk <= d0_sx ? 3'd4 : 3'd0; d_sp_mag <= {1'b0, {WEXP{1'b1}}, {WFRAC{1'b0}}}; end
                    4'b0011: begin d_spk <= 3'd0;                d_sp_mag <= {WFULL{1'b0}};                   end
                    4'b0001: d_spk <= d0_sx ? 3'd4 : 3'd0;
                    4'b0010: d_spk <= 3'd2;
                    default: d_spk <= 3'd0;
                endcase
                d_byp_irem <= f1_byp_ibit ? f1_byp_diff[WMAN-1:0] : f1_num_sig;
                d_byp_ibit <= f1_byp_ibit;
            end

            // The 1/4 pre-scaled significands top out at bit XF-2, within WX.
            reg                  f2_valid;
            reg signed [WX-1:0]  f2_x0, f2_y0;
            always @(posedge clk) begin
                if (rst) f2_valid <= 1'b0; else f2_valid <= d_valid;
                f2_x0 <= {{(WX-WMAN){1'b0}}, d_den_sig} << (XF - WFRAC - 2);
                f2_y0 <= ({{(WX-WMAN){1'b0}}, d_num_sig} << (XF - WFRAC - 2)) >> d_shamt;
            end
            assign vec_start = f2_valid;
            assign vec_x0    = f2_x0;
            assign vec_y0    = f2_y0;
            wire cd_done = ce_done & own_vec;

            // Specials divide garbage; the output masks it.
            wire signed [WX-1:0] be_ykabs = ce_yn[WX-1] ? -ce_yn : ce_yn;
            reg                  dv_valid, dv_run, mag_issue;
            reg signed [WX-1:0]  dv_xn;
            reg signed [WZ-1:0]  dv_zn;
            reg                  dv_yneg;
            reg [WCNT-1:0]       dv_cnt;
            reg [WX-1:0]         dv_rem, dv_den;
            reg [WX+1:0]         dv_den3;
            reg [WQUO-1:0]       dv_quo;
            wire [WX-1:0] den_arm = d_bypass ? {{(WX-WMAN){1'b0}}, d_den_sig} : ce_xn;
            wire [WX-1:0] step_rem_next;
            wire [1:0]      step_digit;
            _zkf_div_radix4_step #(.WMAN(WX)) u_step (
                .den(dv_den), .den3(dv_den3), .rem(dv_rem),
                .rem_next(step_rem_next), .digit(step_digit)
            );
            always @(posedge clk) begin
                if (rst) begin
                    dv_valid <= 1'b0;
                    dv_run   <= 1'b0;
                    mag_issue <= 1'b0;
                end else begin
                    dv_valid  <= 1'b0;
                    mag_issue <= cd_done;
                    if (cd_done) begin
                        dv_run   <= 1'b1;
                    end else if (dv_run) begin
                        if (dv_cnt == (STEPS - 1)) begin
                            dv_run   <= 1'b0;
                            dv_valid <= 1'b1;
                        end
                    end
                end
            end
            always @(posedge clk) begin
                if (cd_done) begin
                    dv_cnt <= {WCNT{1'b0}};
                    dv_rem <= d_bypass ? {{(WX-WMAN){1'b0}}, d_byp_irem} : be_ykabs;
                    dv_den <= den_arm;
                    dv_quo <= {{(WQUO-1){1'b0}}, d_bypass & d_byp_ibit};
                    dv_den3 <= {1'b0, den_arm, 1'b0} + {2'b00, den_arm};
                end else if (dv_run) begin
                    dv_rem <= step_rem_next;
                    dv_quo <= {dv_quo[WQUO-3:0], step_digit};
                    dv_cnt <= dv_cnt + 1'b1;
                end
            end

            // The engine holds these too; the copies keep the back-end's loads off its full-rate recurrence.
            always @(posedge clk) begin
                if (cd_done) begin
                    dv_xn   <= ce_xn;
                    dv_zn   <= ce_zn;
                    dv_yneg <= ce_yn[WX-1];
                end
            end

            // x_K > 0 for every non-special transaction, so the residual divisor needs no zero-guard; the stepper needs
            // rem < den, i.e. |y_K| < x_K.
`ifdef SIMULATION
            always @(posedge clk) begin
                if (!rst && cd_done && !d_special && ce_xn[WX-1])
                    $fatal(1, "%m: residual divisor x_K sign bit set for a non-special transaction");
                if (!rst && cd_done && !d_special && (ce_xn == {WX{1'b0}}))
                    $fatal(1, "%m: residual divisor x_K == 0 for a non-special transaction");
                if (!rst && cd_done && !d_special && !d_bypass && ({1'b0, be_ykabs} >= {1'b0, ce_xn}))
                    $fatal(1, "%m: residual |y_K| >= x_K at arm -- radix-4 divide precondition violated");
            end
`endif

            // MAG and QT are issued STEPS cycles apart, so they never collide in the multiplier.
            localparam [0:0] TAG_MAG = 1'b0, TAG_QT = 1'b1;
            assign vec_pm_iv = mag_issue | dv_valid;
            assign vec_pm_sb = dv_valid ? TAG_QT : TAG_MAG;
            assign vec_pm_a  = dv_valid ? {{(WA_MUL-WQUO){1'b0}}, dv_quo} : dv_xn[WA_MUL-1:0];  // zero-extends (MODE 2)
            assign vec_pm_b  = dv_valid ? ce_inv_tau : ce_kinv_mag;
            wire             pmul_ov      = pm_valid & own_vec;
            wire [WMAG_VEC-1:0] pmul_p    = {{(WMAG_VEC-WPMUL){1'b0}}, pm_p[WPMUL-1:0]};  // pad if theta dominates
            wire             mag_ov = pmul_ov && (pm_sb[0] == TAG_MAG);

            // theta = unmap_const -/+ (z_K +/- res_delta), split so that P2 adds z_K and B2 only the late product term
            // res_delta, whose sign folds swap^sx with y_K < 0.
            wire signed [WZ+1:0] zn_ext      = $signed(dv_zn);                          // z_K can be slightly < 0
            reg  signed [WZ+1:0] unmap_const;
            always @* begin
                casez ({d_swap, d_sx})
                    2'b1?:   unmap_const = QUARTER;
                    2'b01:   unmap_const = HALF;
                    default: unmap_const = {(WZ+2){1'b0}};
                endcase
            end
            wire                 un_neg_a0   = d_swap ^ d_sx;
            reg                      p2_valid, p2_bypass, p2_sub_delta;
            reg signed [WZ+1:0]      p2_un_base;
            reg [WZ+1:0]             p2_res_delta;
            reg [WMAG_VEC-1:0]       p2_byp_tmag;
            reg signed [WEU_PK-1:0]  p2_texp;
            always @(posedge clk) begin
                if (rst) p2_valid <= 1'b0; else p2_valid <= pmul_ov && (pm_sb[0] == TAG_QT);
                p2_bypass    <= d_bypass;
                p2_sub_delta <= un_neg_a0 ^ dv_yneg;
                p2_un_base   <= un_neg_a0 ? (unmap_const - zn_ext) : (unmap_const + zn_ext);
                p2_res_delta <= pmul_p >> (F + INVTAU_S - ZF);                        // >= 0 at every table WMAN
                p2_byp_tmag  <= pmul_p | {{(WMAG_VEC-1){1'b0}}, |dv_rem};
                // Prebiased, so the packer skips its bias add.
                p2_texp      <= (d_bypass ? ((WMAG_PK - 1) - F - INVTAU_S + d_eydiff) : ((WMAG_PK - 1) - ZF)) + BIAS;
            end
            wire signed [WZ+1:0] p2_un_tmag = p2_sub_delta ? (p2_un_base - $signed(p2_res_delta))
                                                           : (p2_un_base + $signed(p2_res_delta));

            reg                      b2_valid;
            reg [WMAG_VEC-1:0]       b2_tmag;
            always @(posedge clk) begin
                if (rst) b2_valid <= 1'b0; else b2_valid <= p2_valid;
                b2_tmag     <= p2_bypass ? p2_byp_tmag : {{(WMAG_VEC-(WZ+2)){1'b0}}, p2_un_tmag[WZ+1:0]};
            end

            // The magnitude is packed first and held in mag_num_r until theta emerges; specials override both results
            // from the held dv_* registers.
            localparam signed [WEU_PK-1:0] MAG_EXP_OFFS = (WMAG_PK - 1) - (XF + KINV_S) + 2;  // +2: the 1/4 pre-scale
`ifdef SIMULATION
            always @(posedge clk) begin
                if (!rst && mag_ov && b2_valid)
                    $fatal(1, "%m: shared back-end collision -- MAG and THETA issued together");
            end
`endif
            assign vec_pk_iv   = mag_ov | b2_valid;
            assign vec_pk_sign = b2_valid & d_sy;
            assign vec_pk_exp  = b2_valid ? p2_texp  : $signed({{(WEU_PK-WE){d_eden[WE-1]}}, d_eden}) + MAG_EXP_OFFS;
            assign vec_pk_mag  = b2_valid ? b2_tmag  : pmul_p;       // zero-extends at MODE 2
            assign vec_pk_sb   = b2_valid;
            wire             be_ov   = pk_valid & own_vec;
            reg [WFULL-1:0]  mag_num_r;
            always @(posedge clk) begin
                if (be_ov && !pk_sb[0]) mag_num_r <= pk_y;
            end
            // turn8, d_spk/8 of a turn, selects constants only: no carry chain on the output cone. -1/2 folds to the
            // canonical +1/2 (turn8's k == 4); |theta| <= 1/2, so only 1/2 has TURN8_X4's exponent.
            reg [WFULL-1:0] turn8;
            always @* begin
                case (d_spk)
                    3'd1:    turn8 = {d_sp_sign, TURN8_X1, {WFRAC{1'b0}}};
                    3'd2:    turn8 = {d_sp_sign, TURN8_X2, {WFRAC{1'b0}}};
                    3'd3:    turn8 = {d_sp_sign, TURN8_X2, 1'b1, {(WFRAC-1){1'b0}}};
                    3'd4:    turn8 = {1'b0,      TURN8_X4, {WFRAC{1'b0}}};
                    default: turn8 = {WFULL{1'b0}};   // 0 is +0; >= 5 never occurs
                endcase
            end
            assign vec_res   = be_ov & pk_sb[0];
            assign vec_theta = d_special ? turn8
                                         : {pk_y[WFULL-1] & (pk_y[WFULL-2:WFRAC] != TURN8_X4), pk_y[WFULL-2:0]};
            assign vec_mag   = d_special ? d_sp_mag : mag_num_r;
        end else begin : g_no_vec
            assign {vec_start, vec_x0, vec_y0, vec_pm_iv, vec_pm_sb, vec_pm_a, vec_pm_b, vec_pk_iv, vec_pk_sign,
                    vec_pk_exp, vec_pk_mag, vec_pk_sb, vec_res, vec_theta, vec_mag} = 0;
        end
    endgenerate

    // Intentional: an in-range WMAN without a generated _zkf_cordic_m<WMAN> fails elaboration, prompting generation.
    `define ZKF_CORDIC_TABLE(W) end else if (WMAN == W) begin : g_m``W \
        _zkf_cordic_m``W #( \
            .MODE(MODE), .UNROLL100(UNROLL100), .PARALLEL(PARALLEL), \
            .EXPECT_WMAN(WMAN), .EXPECT_N(N), .EXPECT_XF(XF), .EXPECT_WX(WX), .EXPECT_WT(WT), \
            .EXPECT_ZF(ZF), .EXPECT_WZ(WZ), .EXPECT_WCONST2PI(WCONST2PI), .EXPECT_CONST2PI_S(CONST2PI_S), \
            .EXPECT_WINVTAU(WINVTAU), .EXPECT_INVTAU_S(INVTAU_S), \
            .EXPECT_WKINV_MAG(WKINV_MAG), .EXPECT_KINV_S(KINV_S) \
        ) u_cordic ( \
            .clk(clk), .rst(rst), .start(rot_start | vec_start), .vectoring(own_vec), \
            .x0(vec_x0), .y0(vec_y0), .z0(own_vec ? {WZ{1'b0}} : rot_z0), \
            .busy(), .done(ce_done), .z_done(ce_zdone), .xn(ce_xn), .yn(ce_yn), .zn(ce_zn), \
            .const2pi(ce_const2pi), .inv_tau(ce_inv_tau), .kinv_mag(ce_kinv_mag));
    generate
        if (1'b0) begin : g_none
        `ZKF_CORDIC_TABLE(11)
        `ZKF_CORDIC_TABLE(12)
        `ZKF_CORDIC_TABLE(13)
        `ZKF_CORDIC_TABLE(14)
        `ZKF_CORDIC_TABLE(15)
        `ZKF_CORDIC_TABLE(16)
        `ZKF_CORDIC_TABLE(17)
        `ZKF_CORDIC_TABLE(18)
        `ZKF_CORDIC_TABLE(19)
        `ZKF_CORDIC_TABLE(20)
        `ZKF_CORDIC_TABLE(21)
        `ZKF_CORDIC_TABLE(22)
        `ZKF_CORDIC_TABLE(23)
        `ZKF_CORDIC_TABLE(24)
        `ZKF_CORDIC_TABLE(25)
        `ZKF_CORDIC_TABLE(26)
        `ZKF_CORDIC_TABLE(27)
        `ZKF_CORDIC_TABLE(28)
        `ZKF_CORDIC_TABLE(29)
        `ZKF_CORDIC_TABLE(30)
        `ZKF_CORDIC_TABLE(31)
        `ZKF_CORDIC_TABLE(32)
        `ZKF_CORDIC_TABLE(33)
        `ZKF_CORDIC_TABLE(34)
        `ZKF_CORDIC_TABLE(35)
        `ZKF_CORDIC_TABLE(36)
        `ZKF_CORDIC_TABLE(37)
        `ZKF_CORDIC_TABLE(38)
        `ZKF_CORDIC_TABLE(39)
        `ZKF_CORDIC_TABLE(40)
        `ZKF_CORDIC_TABLE(41)
        `ZKF_CORDIC_TABLE(42)
        `ZKF_CORDIC_TABLE(43)
        `ZKF_CORDIC_TABLE(44)
        `ZKF_CORDIC_TABLE(45)
        `ZKF_CORDIC_TABLE(46)
        `ZKF_CORDIC_TABLE(47)
        `ZKF_CORDIC_TABLE(48)
        `ZKF_CORDIC_TABLE(49)
        `ZKF_CORDIC_TABLE(50)
        `ZKF_CORDIC_TABLE(51)
        `ZKF_CORDIC_TABLE(52)
        `ZKF_CORDIC_TABLE(53)
        end else begin : g_unsupported
            _zkf_invalid_wman_out_of_range u_invalid();
        end
    endgenerate
    `undef ZKF_CORDIC_TABLE
    _zkf_pmul #(
        .WA(WA_PM), .WB(WB_PM), .A_SIGNED(MODE != 1), .B_SIGNED(MODE != 1), .WSB(WSB_PM),
        .WMULTIPLIER(WMULTIPLIER), .STAGE_PRODUCT(STAGE_PRODUCT)
    ) u_pmul (
        .clk(clk), .rst(rst), .in_valid(rot_pm_iv | vec_pm_iv), .sb_in(own_vec ? vec_pm_sb : rot_pm_sb),
        .a(own_vec ? vec_pm_a : rot_pm_a), .b(own_vec ? vec_pm_b : rot_pm_b),
        .out_valid(pm_valid), .sb_out(pm_sb), .p(pm_p)
    );
    _zkf_fixed_to_float #(
        .WEXP(WEXP), .WMAN(WMAN), .WMAG(WMAG_PK), .WEU(WEU_PK), .EXP_IS_BIASED(MODE != 0),
        .ASSUME_NO_OVERFLOW(MODE == 0), .SATURATE_ROUND_CARRY(MODE != 0), .WSB(WSB_PK),
        .STAGE_NORMALIZE(STAGE_NORMALIZE), .STAGE_PACK((STAGE_PACK == 2) ? 1 : STAGE_PACK),
        .STAGE_OUTPUT(STAGE_PACK == 2)
    ) u_f2f (
        .clk(clk), .rst(rst),
        .in_valid(rot_pk_iv | vec_pk_iv), .sign(own_vec ? vec_pk_sign : rot_pk_sign), .force_zero(1'b0),
        .force_inf(~own_vec & rot_pk_inf), .exp_offset(own_vec ? vec_pk_exp : rot_pk_exp),
        .mag(own_vec ? vec_pk_mag : rot_pk_mag), .sb_in(own_vec ? vec_pk_sb : rot_pk_sb),
        .out_valid(pk_valid), .y(pk_y), .sb_out(pk_sb)
    );

    assign res_valid = rot_res | vec_res;
    assign res_r0    = own_vec ? vec_theta : rot_sin;
    assign res_r1    = own_vec ? vec_mag   : rot_cos;
    assign res_quad  = own_vec ? 2'b00     : rot_quad;

`ifdef SIMULATION
    always @(posedge clk) begin
        if (!rst && (own_vec ? (rot_start | rot_pm_iv | rot_pk_iv) : (vec_start | vec_pm_iv | vec_pk_iv)))
            $fatal(1, "%m: the datapath not owning the transaction drives a shared resource");
    end
`endif
endmodule

`default_nettype wire
