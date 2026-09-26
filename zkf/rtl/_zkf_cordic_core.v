/// Folded (iterative) CORDIC engine shared by the ZKF trigonometric operators. One (x, y, z) datapath is reused over
/// several cycles, a configurable number of iterations unrolled per cycle, instead of an N-stage pipeline -- so the
/// area is a single datapath at the cost of an initiation interval equal to the latency.
///
/// UNROLL100 is the latency knob (iterations per cycle x100); pick the largest that closes timings:
///     50 = one iteration per two cycles (split shift/add to halve the per-iteration combinational path, at 2N cycles);
///     100 = one iteration per cycle;
///     200/300/400 = 2/3/4 iterations per cycle (fewer cycles, longer path).
///
/// MODE selects the trajectory; the x/y/z update is otherwise identical:
///
///   MODE = 0 (ROTATION):  sigma_i = (z_i >= 0) ? +1 : -1   -- drives the angle z to 0; (x, y) rotates by z0.
///                         Used by zkf_sincos with (x0, y0) = (1/gain, 0) so (xn, yn) ~ (cos z0, sin z0) and
///                         zn is the small residual _zkf_cordic_unit finishes with one linear rotation.
///
///   MODE = 1 (VECTORING): sigma_i = (y_i >= 0) ? -1 : +1   -- drives y to 0; zn = z0 + atan2(y0, x0), xn ~ |(x0,y0)|.
///                         Used by zkf_atan2.
///
///   MODE = 2 (RUNTIME):   either of the above per transaction, vectoring iff `vectoring` is high at `start`.
///
/// Each iteration: x' = x -/+ (y >>> i); y' = y +/- (x >>> i); z' = z -/+ L[i]. The shift `>>> i` truncates toward
/// -inf (matches the Python model's `>> i`). In the fold the shift amount i is the running iteration index, so it is a
/// variable (barrel) shift and L[i] is a variable index into the flat LUT bus -- unlike the pipelined CORDIC's
/// per-stage constant shifts. Each update is one controlled add/sub (a + (b ^ {W{sub}}) + sub: one carry chain).
///
/// Structure: a single x/y rotator (fast = U iters/cycle, or pipe = one iter / two cycles) that consumes a sigma
/// stream, plus -- only when PARALLEL is set, for rotation transactions -- a separate z-engine that produces that
/// stream ahead of time. The sigma sequence is identical either way; "coupled" (lock-step) and "decoupled" differ only
/// in HOW sigma reaches the rotator: an inline combinational tap off the in-step z-chain / y (coupled), or a registered
/// read from sig_mem fed by the ahead-running z-engine (decoupled). So the rotator is written once; the z handling is
/// the only thing that varies. Decoupling lets sincos start its residual-angle correction during the CORDIC.
///
/// PARALLEL only helps -- and is only legal -- with the half-rate (pipe) rotator: the z-recurrence is one narrow add,
/// so its fast rate is one iteration/cycle, which laps a half-rate x/y but merely ties a full-rate one. So a full-rate
/// rotator stays lock-step; the default should not be changed except for testing. At MODE=2 the z-engine also serves
/// vectoring transactions, stepping in lock-step.
///
/// Handshake: assert `start` for one cycle with x0/y0/z0 valid; `busy` is high while iterating; `done` pulses for one
/// cycle with xn/yn/zn valid. `start` is ignored while busy. Reset clears the FSM.

`default_nettype none

module _zkf_cordic_core #(
    parameter integer N           = 14,  // iterations
    parameter integer UNROLL100   = 100,
    parameter integer MODE        = 0,   // 0 = rotation, 1 = vectoring, 2 = runtime
    parameter integer PARALLEL    = (MODE != 1) && (UNROLL100 < 100),  // rotation: run the z-path ahead
    parameter integer WX          = 32,  // signed x/y width
    parameter integer WZ          = 32   // signed angle width
) (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,
    input  wire                 vectoring,
    input  wire signed [WX-1:0] x0,
    input  wire signed [WX-1:0] y0,
    input  wire signed [WZ-1:0] z0,
    input  wire [(((N > 0) ? N : 1)*WZ)-1:0] lut,  // L[i] (unsigned) at bits [i*WZ +: WZ]
    output wire                 busy,
    output wire                 done,        // pulses with xn/yn valid
    output wire                 z_done,      // decoupled rotation: pulses with zn valid, ahead of `done` (else == done)
    output wire signed [WX-1:0] xn,
    output wire signed [WX-1:0] yn,
    output wire signed [WZ-1:0] zn
);
    localparam integer N_EFF = (N > 0) ? N : 1;  // Keeps invalid N structurally well-formed until validation fires.
    localparam integer U    = (UNROLL100 < 100) ? 1 : (UNROLL100 / 100);
    localparam integer PIPE = UNROLL100 < 100;
    localparam integer DECOUPLE = (MODE != 1) && (PARALLEL != 0);
    localparam integer WI   = $clog2(N_EFF + U + 1);  // holds i_r + U / zi_r + 1; idle-lane indices may wrap

    generate
        if ((UNROLL100 != 50) && ((UNROLL100 < 100) || ((UNROLL100 % 100) != 0))) begin : g_invalid_unroll100
            _zkf_invalid_unroll100 u_invalid();
        end
        if (N <= 0) begin : g_invalid_n
            _zkf_invalid_cordic_n_must_be_positive u_invalid();
        end
        if ((MODE != 0) && (MODE != 1) && (MODE != 2)) begin : g_invalid_mode
            _zkf_invalid_cordic_mode u_invalid();
        end
        if (DECOUPLE && (PIPE == 0)) begin : g_invalid_decouple
            _zkf_decouple_needs_half_rate u_invalid();
        end
        if ((MODE == 1) && (PARALLEL != 0)) begin : g_invalid_parallel
            _zkf_parallel_needs_rotation_mode u_invalid();
        end
    endgenerate

    reg signed [WX-1:0] x_r;
    reg signed [WX-1:0] y_r;                   // y datapath (feeds yn, covered end-to-end)
    reg signed [WZ-1:0] z_r;
    reg        [WI-1:0] i_r;                   // base iteration index for this cycle
    reg                 run_r, done_r;
    reg                 vec_r;
    reg     [N_EFF-1:0] sig_mem; // sigma replay: written by the decoupled z-engine, read by the rotator.
    wire                adv;     // i_r advances by U: every cycle at full rate, on the add phase at half rate

    // i_r advances unconditionally so latency stays constant data-independent.
    wire last = (i_r + U[WI-1:0]) >= N_EFF[WI-1:0];
    always @(posedge clk) begin
        if (rst) begin
            run_r  <= 1'b0;
            done_r <= 1'b0;
        end else begin
            done_r <= run_r && adv && last;
            if (!run_r)           run_r <= start;
            else if (adv && last) run_r <= 1'b0;
        end
    end
    always @(posedge clk) begin
        if (!run_r) begin
            if (start) begin
                i_r <= {WI{1'b0}}; vec_r <= vectoring;
            end
        end else if (adv) begin
            i_r <= i_r + U[WI-1:0];
        end
    end

    // Sliced at compile-time offsets: a runtime lut[idx*WZ +: WZ] makes synthesis build the idx*WZ multiply (Diamond's
    // critical path on the wide vectoring engine). Zero entries past N-1 keep every read in bounds without a clamp: the
    // fast rotator's post-run idle reads reach N + 2U - 2, the decoupled prefetch N.
    localparam integer LUT_HI = N_EFF + 2*U - 1;
    wire [WZ-1:0] lut_a [0:LUT_HI];
    genvar gl;
    generate
        for (gl = 0; gl < N_EFF; gl = gl + 1) begin : g_lut_unpack
            assign lut_a[gl] = lut[gl*WZ +: WZ];   // constant (elaboration-time) offset -- no runtime multiply
        end
        for (gl = N_EFF; gl <= LUT_HI; gl = gl + 1) begin : g_lut_sentinel
            assign lut_a[gl] = {WZ{1'b0}};         // don't-care sentinels for terminal (en=0 / prefetch) indices
        end
    endgenerate

    generate
        // Rotator: the x/y datapath, shared by every mode. Per lane the sigma sign comes from sig_mem (decoupled) or,
        // in lock-step, the inline z-chain (rotation) / y (vectoring); the inline z-chain advances z_r in g_zadv. The
        // x/y add/sub, the FSM, and the handshake below the sigma source are identical across modes.
        if (PIPE == 0) begin : g_fast
            // U iterations per cycle. Combinational chain from the registered state, starting at index i_r. Iterations
            // whose index reaches N pass through unchanged (the last cycle may be a partial group if N % U != 0).
            // The x/y load rides the adders (all lanes masked while idle), so x_r/y_r's D is a bare sum. z keeps its
            // load and lane muxes: with a bare-sum D, Diamond LSE retimes z_r into its adder, onto the sigma path.
            wire signed [WX-1:0] cx [0:U];
            wire signed [WX-1:0] cy [0:U];
            wire signed [WZ-1:0] cz [0:U];     // inline z-chain (the full-rate rotator is always lock-step)
            assign cx[0] = run_r ? x_r : x0;
            assign cy[0] = run_r ? y_r : y0;
            assign cz[0] = z_r;

            genvar u;
            for (u = 0; u < U; u = u + 1) begin : g_unroll
                wire signed [WX-1:0] uy    = (u == 0) ? y_r : cy[u];
                wire [WI-1:0]        idx   = i_r + u[WI-1:0];
                wire                 en_z  = (idx < N_EFF[WI-1:0]);
                wire                 en    = run_r && en_z;
                wire signed [WX-1:0] ysh   = uy >>> idx;
                wire signed [WX-1:0] xsh   = ((u == 0) ? x_r : cx[u]) >>> idx;
                wire [WZ-1:0]        li    = lut_a[idx];
                wire                 neg;               // true => sigma = -1
                if (MODE == 0) begin : g_sig
                    assign neg = cz[u][WZ-1];
                end else if (MODE == 1) begin : g_sig
                    assign neg = ~uy[WX-1];
                end else begin : g_sig
                    assign neg = vec_r ? ~uy[WX-1] : cz[u][WZ-1];
                end
                wire                 sub_x = en & ~neg; // x subtracts ysh when sigma = +1
                wire                 sub_y = en &  neg; // y subtracts xsh when sigma = -1
                wire                 sub_z = ~neg;      // z subtracts li  when sigma = +1
                wire signed [WZ-1:0] nz    = cz[u] + ($signed({1'b0, li}) ^ {WZ{sub_z}}) + {{(WZ-1){1'b0}}, sub_z};
                assign cx[u+1] = cx[u] + ((ysh & {WX{en}}) ^ {WX{sub_x}}) + {{(WX-1){1'b0}}, sub_x};
                assign cy[u+1] = cy[u] + ((xsh & {WX{en}}) ^ {WX{sub_y}}) + {{(WX-1){1'b0}}, sub_y};
                assign cz[u+1] = en_z ? nz : cz[u];
            end

            assign adv = 1'b1;
            always @(posedge clk) begin
                if (run_r || start) begin
                    x_r <= cx[U]; y_r <= cy[U];
                end
                if (run_r)      z_r <= cz[U];
                else if (start) z_r <= z0;
            end
            assign z_done = done_r;  // lock-step: zn lands coincident with done
        end else begin : g_pipe
            // One iteration per 2 cycles for wide datapaths: phase 0 registers the shifted operands and the sampled
            // sigma sign; phase 1 applies the add/sub. Splitting the long shift -> wide-add cone across a register
            // closes timing, at 2*N cycles.
            reg                 phase_r;           // 0 = shift/sample, 1 = add/advance
            reg signed [WX-1:0] xsh_r;             // x>>>i sampled in phase 0
            reg signed [WX-1:0] ysh_r;             // y>>>i sampled in phase 0
            reg                 neg_r;             // sigma sign sampled in phase 0 (true => sigma = -1)
            wire [WI-1:0]        idx = i_r;
            wire signed [WX-1:0] xsh = x_r >>> idx;
            wire signed [WX-1:0] ysh = y_r >>> idx;
            wire                 neg;
            wire                 sub_x = ~neg_r;
            wire                 sub_y =  neg_r;
            wire signed [WX-1:0] nx = x_r + (ysh_r ^ {WX{sub_x}}) + {{(WX-1){1'b0}}, sub_x};
            wire signed [WX-1:0] ny = y_r + (xsh_r ^ {WX{sub_y}}) + {{(WX-1){1'b0}}, sub_y};
            if (DECOUPLE && (MODE == 0)) begin : g_sig
                assign neg = sig_mem[idx];                  // sigma replayed from the ahead-running z-engine
            end else if (DECOUPLE) begin : g_sig
                assign neg = vec_r ? ~y_r[WX-1] : sig_mem[idx];
            end else if (MODE == 0) begin : g_sig
                assign neg = z_r[WZ-1];
            end else if (MODE == 1) begin : g_sig
                assign neg = ~y_r[WX-1];
            end else begin : g_sig
                assign neg = vec_r ? ~y_r[WX-1] : z_r[WZ-1];
            end
            assign adv = phase_r;
            always @(posedge clk) begin
                if (rst)        phase_r <= 1'b0;
                else if (run_r) phase_r <= ~phase_r;
            end
            always @(posedge clk) begin
                if (!run_r) begin
                    if (start) begin
                        x_r <= x0; y_r <= y0;
                    end
                end else if (phase_r == 1'b0) begin
                    xsh_r <= xsh; ysh_r <= ysh; neg_r <= neg;
                end else begin
                    x_r <= nx; y_r <= ny;
                end
            end

            // Decoupled z-engine (PARALLEL): for rotation it runs the sigma recurrence at full rate AHEAD of the
            // rotator, buffering sigma into sig_mem and exposing zn and z_done early. sig_mem[0] is preloaded at start,
            // so the rotator needs no head-start. L[zi_r] is pre-fetched into li_r, lifting the wide lut[] index mux
            // out of the WZ-wide add cone.
            if (DECOUPLE) begin : g_sigma
                reg        [WI-1:0]  zi_r;              // z-iteration index for this cycle
                reg                  z_run_r, z_dn_r;
                reg        [WZ-1:0]  li_r;
                wire                 zvec = (MODE == 2) && vec_r;
                wire                 zneg = z_r[WZ-1];  // true => sigma = -1
                wire                 zsub = zvec ? ~neg_r : ~zneg;  // z subtracts L[i] when sigma = +1
                wire        [WI-1:0] zi_nxt = zi_r + 1'b1;
                wire                 z_last = (zi_nxt >= N_EFF[WI-1:0]);
                wire                 z_load = start && !run_r;
                always @(posedge clk) begin
                    if (rst) begin
                        z_run_r <= 1'b0;
                        z_dn_r  <= 1'b0;
                    end else begin
                        z_dn_r <= z_run_r && z_last;
                        if (!z_run_r)    z_run_r <= z_load && ((MODE == 0) || !vectoring);
                        else if (z_last) z_run_r <= 1'b0;
                    end
                end
                always @(posedge clk) begin
                    if (z_run_r || (zvec && run_r && phase_r)) begin
                        z_r           <= z_r + ($signed({1'b0, li_r}) ^ {WZ{zsub}}) + {{(WZ-1){1'b0}}, zsub};
                        zi_r          <= zi_nxt;
                        sig_mem[zi_r] <= zneg;
                        li_r          <= lut_a[zi_nxt];         // don't-care past N, then unused
                    end else if (z_load) begin
                        z_r        <= z0;
                        zi_r       <= {WI{1'b0}};
                        sig_mem[0] <= z0[WZ-1];
                        li_r       <= lut_a[0];
                    end
                end
                assign z_done = z_dn_r | (zvec & done_r);
            end

            if (!DECOUPLE) begin : g_zadv                  // lock-step z-path: sample L (phase 0), add to z_r (phase 1)
                reg        [WZ-1:0] li_r;
                wire       [WZ-1:0] li    = lut_a[idx];
                wire                sub_z = ~neg_r;
                wire signed [WZ-1:0] nz   = z_r + ($signed({1'b0, li_r}) ^ {WZ{sub_z}}) + {{(WZ-1){1'b0}}, sub_z};
                always @(posedge clk) begin
                    if (!run_r) begin
                        if (start) z_r <= z0;
                    end else if (phase_r == 1'b0) begin
                        li_r <= li;
                    end else begin
                        z_r <= nz;
                    end
                end
                assign z_done = done_r;
            end
        end
    endgenerate

    assign busy   = run_r;
    assign done   = done_r;
    assign xn     = x_r;
    assign yn     = y_r;
    assign zn     = z_r;
endmodule

`default_nettype wire
