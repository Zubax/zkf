// Iterative (single-transaction) sine and cosine of a phase in turns (with the reduced quadrant exposed):
//   sin = sin(2*pi*x), cos = cos(2*pi*x)
//
// This is NOT a throughput-1 pipeline. A transaction is accepted when `in_ready` is high; the module then runs for a
// fixed data-invariant latency and holds `out_valid` with a stable result until `out_ready` accepts it.
// LATENCY is accept-to-out_valid latency; with out_ready high, in_ready reasserts one cycle after retirement.
// Faithful rounding. Behavior:
//
//   x finite : sin = sin(2*pi*x), cos = cos(2*pi*x); quadrant = floor(frac(x)*4) mod 4
//              Exact boundaries take the upper quadrant: 0->0, 1/4->1, 1/2->2, 3/4->3); exact-zero outputs are +0.
//   x = +inf : sin = +inf, cos = +inf, quadrant = 0.
//   x = -inf : sin = -inf, cos = -inf, quadrant = 0.
//
// Algorithm (mixed CORDIC; the engine is in _zkf_cordic_core):
//
//  1. Reduce x mod 1 to the WFF-bit fraction frac(x) (WFF = WMAN + GUARD_FF): top 2 bits = |x| quadrant, low
//     WT = WFF-2 bits = quadrant-local coordinate t (local angle (pi/2)*t). Reduce |x| and use the sin sign flip /
//     quadrant reflection for x < 0. Fold the half-quadrant symmetry to bring the local angle theta' into [0, pi/4].
//
//  2. Run N CORDIC rotation iterations (rotation mode) on the folded engine to get (cos theta', sin theta') ~ at
//     scale 2**-XF and the small residual angle z_K.
//
//  3. Finish with ONE linear rotation by the residual: sin = y_K + x_K*phi, cos = x_K - y_K*phi, phi = 2*pi*z_K. The
//     correction multiplies (the operator's only ones) use the per-WMAN 2*pi constant. Tiny / below-TSA angles take
//     the linear small-angle bypass (sin = 2*pi*theta'_turns, cos = 1) from the same 2*pi constant.
//
//  4. Unmap the octant and |x| quadrant; one shared _zkf_fixed_to_float back-end renormalizes and rounds sin first,
//     then cos one cycle later. The packed sin is held until cos emerges so the outputs remain paired.
//
// Tuning knobs:
//
// WMULTIPLIER: Optional DSP multiplier argument width hint if wide multiplication is used.
//     By default (zero), the multiplication module will split arguments symmetrically, which is not always optimal.
//     If nonzero, the multiplier derives the minimal (possibly asymmetric) slice grid so each slice fits a
//     WMULTIPLIER-bit tile. This can significantly reduce DSP usage and may improve f_max.
//
// UNROLL100={50,100,200,...}: CORDIC iterations per engine cycle x100. Values <100 split operations across cycles.
//     Choose the maximum value that closes timings. It is forwarded to _zkf_cordic_core; refer there for details.
//
// STAGE_INPUT: Latch the input x before the decode; values >1 add dummy stages for routing-congested designs.
//     Value = latency cycle cost.
//
// STAGE_PRODUCT: Pipeline depth of the shared correction multiply and its default symmetric split.
//     See _zkf_pmul for details. Each unit costs up to two latency cycles (two dependent products).
//
// STAGE_NORMALIZE: Internal normshift barriers in the shared _zkf_fixed_to_float. Value = latency cycle cost.
//
// STAGE_PACK={0,1,2}: Register the _zkf_pack input (1), also its output (2), refer there. Value = latency cycle cost.
//
// STAGE_OUTPUT={0,1}: Register the results ahead of the out_ready hold (+1 cycle).

`default_nettype none

module zkf_sincos #(
    parameter WEXP            = 6,
    parameter WMAN            = 18,     // significand precision including the hidden bit
    parameter WMULTIPLIER     = 0,
    parameter UNROLL100       = 100,    // choose maximum value that closes timings
    parameter STAGE_INPUT     = 0,
    parameter STAGE_PRODUCT   = 0,
    parameter STAGE_NORMALIZE = 0,
    parameter STAGE_PACK      = 0,
    parameter STAGE_OUTPUT    = 0,
    parameter PARALLEL        = UNROLL100 < 100,   // Testing-only knob, DO NOT override.
    parameter LATENCY         = 0
) (
    input  wire                 clk,
    input  wire                 rst,

    input  wire                 in_valid,    // start a transaction (sampled only when in_ready)
    output wire                 in_ready,    // high when idle and able to accept a transaction
    input  wire [WEXP+WMAN-1:0] x,

    output wire                 out_valid,   // result is ready; held until out_ready (back-pressure)
    input  wire                 out_ready,   // consumer accepts the result on a cycle where out_valid & out_ready
    output wire [WEXP+WMAN-1:0] sin,
    output wire [WEXP+WMAN-1:0] cos,
    output wire [1:0]           quadrant
);
    _zkf_cordic_unit #(
        .WEXP(WEXP), .WMAN(WMAN), .MODE(0), .WMULTIPLIER(WMULTIPLIER), .UNROLL100(UNROLL100),
        .STAGE_INPUT(STAGE_INPUT), .STAGE_PRODUCT(STAGE_PRODUCT), .STAGE_NORMALIZE(STAGE_NORMALIZE),
        .STAGE_PACK(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT), .PARALLEL(PARALLEL), .LATENCY_ROTATION(LATENCY)
    ) u_unit (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in_ready(in_ready), .vectoring(1'b0), .a(x),
        .b({(WEXP + WMAN){1'b0}}), .out_valid(out_valid), .out_ready(out_ready), .r0(sin), .r1(cos), .quadrant(quadrant)
    );
endmodule

`default_nettype wire
