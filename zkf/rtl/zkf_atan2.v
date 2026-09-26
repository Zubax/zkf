// Iterative (single-transaction) two-argument arctangent and vector magnitude:
//   theta = atan2(y, x) in turns, range (-0.5, 0.5];   mag = hypot(y, x) = sqrt(x*x + y*y)
//
// This is NOT a throughput-1 pipeline. A transaction is accepted when `in_ready` is high; the module then runs for a
// fixed data-invariant latency and holds `out_valid` with a stable result until `out_ready` accepts it.
// LATENCY is accept-to-out_valid latency; with out_ready high, in_ready reasserts one cycle after retirement.
// Faithful rounding, magnitude overflow excepted. WEXP >= 5.
//
// Behavior (no NaN, only +0; tiny negative results flush to +0):
//   theta = atan2(y, x) / (2*pi)   [turns]; mag = hypot(y, x). Axis/diagonal specials:
//     y=+0,x>0 -> +0 ; y=+0,x<0 -> 1/2 ; y>0,x=0 -> 1/4 ; y<0,x=0 -> -1/4 ; y=0,x=0 -> +0 (mag +0).
//     |y|=inf,x finite -> +-1/4 ; x=+inf,y finite -> +-0 -> +0 ; x=-inf,y finite -> +1/2 ;
//     (inf,inf) -> +-1/8 (x>0) / +-3/8 (x<0).  mag = +inf whenever any input is inf.
// The magnitude's overflow behavior is held by SATURATE_ROUND_CARRY=1 on the shared back-end -- the magnitude error is
// two-sided at every GUARD_ITER_ATAN2, so a deeper CORDIC is no substitute. The exponent routes are held only by
// margin (the pre-rounding error stays well under the 0.5 ULP that would trip them), so widening the datapath error
// threatens that contract, not just the ULP figure. For theta, see GUARD_DIV.
//
// Algorithm (vectoring mixed CORDIC; the engine is in _zkf_cordic_core):
//
//  1. Decode (y, x); order den=max(|x|,|y|), num=min; align num down to den's binade. The vector is seeded into the
//     engine pre-scaled by 1/4 (den in [0.25, 0.5)*2**XF) so the CORDIC magnitude growth x_K = gain*hypot stays
//     inside the shared engine's signed width WX (which sincos sizes for a ~1*2**XF rotated magnitude).
//     a0 = atan(num/den).
//
//  2. Run N vectoring iterations driving y -> 0: z_K ~= a0 (turns, 2**-ZF), x_K ~= gain*hypot, y_K ~= 0.
//
//  3. Finish the small angle with ONE division: a0 = z_K + (y_K/x_K)*INV_TAU (the vectoring analogue of the sincos
//     linear-rotation multiply). A small-ratio bypass handles the near-+x-axis corner where theta underflows the
//     fixed-turns accumulator: theta = (|y|/|x|)*INV_TAU computed directly as a float (single rounding).
//
//  4. Octant/quadrant unmap from (sign x, sign y, swap) places theta in (-0.5, 0.5]; the magnitude descales x_K by
//     1/gain (== the per-WMAN KINV) and carries the den binade. One shared _zkf_fixed_to_float back-end
//     renormalizes + rounds both results (magnitude then theta of the single in-flight transaction, time-multiplexed).
//
// Arithmetic reuse: ONE folded radix-4 divider (the _zkf_div_core primitive _zkf_div_radix4_step,
// STEPS = ceil(XF/2) cycles, 2 quotient bits each) computes Q = floor(num*2**F/den) + sticky for BOTH the residual
// (num=|y_K|, den=x_K) and the bypass (num=sig_y, den=sig_x); they are mutually exclusive per transaction. ONE shared
// _zkf_pmul computes the magnitude product x_K*KINV (issued DURING the divide, so it costs no latency) and the
// post-divide Q*INV_TAU (the residual correction, and the bypass theta -- single-rounded via the divide sticky).
//
// Tuning knobs follow zkf_sincos.

`default_nettype none

module zkf_atan2 #(
    parameter WEXP            = 6,      // exponent field width
    parameter WMAN            = 18,     // significand precision including the hidden bit
    parameter WMULTIPLIER     = 0,      // optional native DSP tile operand width for tighter optimization
    parameter UNROLL100       = 100,
    parameter STAGE_INPUT     = 0,
    parameter STAGE_PRODUCT   = 0,      // shared-multiplier pipeline depth (1+STAGE_PRODUCT cycles)
    parameter STAGE_NORMALIZE = 0,
    parameter STAGE_PACK      = 0,
    parameter STAGE_OUTPUT    = 0,
    parameter LATENCY         = 0
) (
    input  wire                 clk,
    input  wire                 rst,

    input  wire                 in_valid,    // start a transaction (sampled only when in_ready)
    output wire                 in_ready,    // high when idle and able to accept a transaction
    input  wire [WEXP+WMAN-1:0] y,           // first argument (numerator / opposite)
    input  wire [WEXP+WMAN-1:0] x,           // second argument (denominator / adjacent)

    output wire                 out_valid,   // result is ready; held until out_ready (back-pressure)
    input  wire                 out_ready,   // consumer accepts the result on a cycle where out_valid & out_ready
    output wire [WEXP+WMAN-1:0] theta,       // atan2(y, x) in turns, range (-0.5, 0.5]
    output wire [WEXP+WMAN-1:0] mag          // hypot(y, x)
);
    _zkf_cordic_unit #(
        .WEXP(WEXP), .WMAN(WMAN), .MODE(1), .WMULTIPLIER(WMULTIPLIER), .UNROLL100(UNROLL100),
        .STAGE_INPUT(STAGE_INPUT), .STAGE_PRODUCT(STAGE_PRODUCT), .STAGE_NORMALIZE(STAGE_NORMALIZE),
        .STAGE_PACK(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT), .LATENCY_VECTORING(LATENCY)
    ) u_unit (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in_ready(in_ready), .vectoring(1'b1), .a(y), .b(x),
        .out_valid(out_valid), .out_ready(out_ready), .r0(theta), .r1(mag), .quadrant()
    );
endmodule

`default_nettype wire
