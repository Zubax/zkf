/// Iterative (single-transaction) CORDIC with the mode chosen per transaction:
/// - sine and cosine of a phase expressed in turns (2pi radians per unit);
/// - two-argument arctangent in turns and the vector magnitude.
/// One engine serves both modes, so a design needing both functions carries one datapath instead of separate zkf_sincos
/// and zkf_atan2. The two modes have distinct latencies, hence we have separate LATENCY_ROTATION and LATENCY_VECTORING,
/// unlike most other modules in the ZKF library.
///
///   vectoring = 0:  a = x in turns, b ignored   ->  r0 = sin(2*pi*x), r1 = cos(2*pi*x), quadrant as zkf_sincos
///   vectoring = 1:  a = y, b = x                ->  r0 = atan2(y, x) in turns, r1 = hypot(y, x), quadrant = 0
///
/// This is NOT a throughput-1 pipeline. A transaction is accepted when `in_ready` is high, which samples `vectoring`,
/// `a` and `b`; the module then runs for its mode's latency and holds `out_valid` with a stable result until
/// `out_ready` accepts it. With out_ready high, in_ready reasserts one cycle after retirement.
///
/// Results, accuracy and special cases are exactly zkf_sincos's (rotation) or zkf_atan2's (vectoring) at the same
/// parameters. Requires WEXP >= 5.
///
/// Tuning knobs:
///
/// WMULTIPLIER: Optional DSP multiplier argument width hint for the shared multiplier. By default (zero), wide
///     products are split symmetrically, which is not always optimal. If nonzero, the minimal (possibly asymmetric)
///     slice grid is derived so each slice fits a WMULTIPLIER-bit tile; this can reduce DSP usage and improve f_max.
///
/// UNROLL100={50,100,200,...}: CORDIC iterations per engine cycle x100. Values <100 split operations across cycles.
///     Choose the maximum value that closes timings.
///
/// STAGE_INPUT: Register the inputs before the decode. Value = latency cycle cost.
///
/// STAGE_PRODUCT: Pipeline depth of the shared multiplier. Each unit costs one cycle in vectoring and up to two in
///     rotation, which issues two dependent products.
///
/// STAGE_NORMALIZE={0,1,2}: Register barriers inside the shared normalizer. Value = latency cycle cost.
///
/// STAGE_PACK={0,1,2}: 1 registers the shared rounder's input, 2 also its rounded output. Value = latency cycle cost.
///
/// STAGE_OUTPUT={0,1}: Register the results ahead of the out_ready hold (+1 cycle).
///
/// LATENCY_ROTATION, LATENCY_VECTORING: Each mode's LATENCY: zkf_sincos's and zkf_atan2's, respectively, at the same
///     parameters, so refer there for details. Each mode's initiation interval is its latency + 1.

`default_nettype none

module zkf_cordic #(
    parameter WEXP            = 6,      // exponent field width
    parameter WMAN            = 18,     // significand precision including the hidden bit
    parameter WMULTIPLIER     = 0,
    parameter UNROLL100       = 100,
    parameter STAGE_INPUT     = 0,
    parameter STAGE_PRODUCT   = 0,
    parameter STAGE_NORMALIZE = 0,
    parameter STAGE_PACK      = 0,
    parameter STAGE_OUTPUT    = 0,
    parameter LATENCY_ROTATION  = 0,
    parameter LATENCY_VECTORING = 0
) (
    input  wire                 clk,
    input  wire                 rst,

    input  wire                 in_valid,    // start a transaction (sampled only when in_ready)
    output wire                 in_ready,    // high when idle and able to accept a transaction
    input  wire                 vectoring,
    input  wire [WEXP+WMAN-1:0] a,
    input  wire [WEXP+WMAN-1:0] b,

    output wire                 out_valid,   // result is ready; held until out_ready (back-pressure)
    input  wire                 out_ready,   // consumer accepts the result on a cycle where out_valid & out_ready
    output wire [WEXP+WMAN-1:0] r0,
    output wire [WEXP+WMAN-1:0] r1,
    output wire [1:0]           quadrant
);
    _zkf_cordic_unit #(
        .WEXP(WEXP), .WMAN(WMAN), .MODE(2), .WMULTIPLIER(WMULTIPLIER), .UNROLL100(UNROLL100),
        .STAGE_INPUT(STAGE_INPUT), .STAGE_PRODUCT(STAGE_PRODUCT), .STAGE_NORMALIZE(STAGE_NORMALIZE),
        .STAGE_PACK(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT), .LATENCY_ROTATION(LATENCY_ROTATION),
        .LATENCY_VECTORING(LATENCY_VECTORING)
    ) u_unit (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in_ready(in_ready), .vectoring(vectoring), .a(a), .b(b),
        .out_valid(out_valid), .out_ready(out_ready), .r0(r0), .r1(r1), .quadrant(quadrant)
    );
endmodule

`default_nettype wire
