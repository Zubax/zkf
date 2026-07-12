/// Streamed cast from Zubax Kulibin float to signed two's-complement integer with saturation.
/// +inf saturates to 2^(WINT-1)-1, -inf saturates to -2^(WINT-1), finite overflows saturate to the same bounds,
/// zero produces zero, and finite in-range values are round-to-nearest, ties-to-even.
/// The pipeline is a fixed 4 + STAGE_INPUT stages: to_int deliberately exposes no STAGE_NORMALIZE/PACK/OUTPUT tuning
/// knobs (unlike zkf_from_int) because the cast is cheap, and it needs no wide-WEXP portability guard of its own --
/// the underlying _zkf_to_fixpoint already rejects the unportable WEXP >= 31 range.

`default_nettype none

module zkf_to_int #(
    parameter WEXP        = 6,
    parameter WMAN        = 18,
    parameter WINT        = 32,
    parameter STAGE_INPUT = 0,  // number of input register stages (>=0); +STAGE_INPUT cycles
    parameter LATENCY     = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] a,

    output wire                   out_valid,
    output wire signed [WINT-1:0] y
);
    localparam LATENCY_REF = 4 + STAGE_INPUT;
    generate
        if ((WEXP < 2) || (WMAN < 4) || (WINT < 2)) begin : g_invalid
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    // -- Float -> unsigned WINT-bit magnitude reduction. _zkf_to_fixpoint owns the decode, the folded-constant
    // shift predicates, the right/left barrel shifters, and the two register stages (S1 capturing decode+clamps,
    // S2 capturing post-shift magnitude+GRS+specials). WI=WINT, FF=0 selects the to_int layout: mag = integer
    // magnitude, guard = bit just below the integer LSB, lost_sticky = combined round|sticky tail.
    wire             s2_valid;
    wire [WINT-1:0]  s2_mag;
    wire             s2_guard;
    wire             s2_lost_sticky;
    wire             s2_sign;
    wire             s2_is_inf;
    wire             s2_is_zero;
    wire             s2_oor;
    _zkf_to_fixpoint #(
        .WEXP(WEXP), .WMAN(WMAN),
        .WI(WINT), .FF(0),
        .STAGE_INPUT(STAGE_INPUT)
    ) u_to_fixpoint (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .a(a),
        .out_valid(s2_valid),
        .mag(s2_mag), .guard(s2_guard), .lost_sticky(s2_lost_sticky),
        .sign(s2_sign), .is_inf(s2_is_inf), .is_zero(s2_is_zero), .oor(s2_oor)
    );
    // s2_is_inf and s2_is_zero ride the helper but are not consumed here; +-inf is already folded into s2_oor,
    // and zero produces mag=0 through the helper's right-shift saturation. Tie off explicitly to keep lint happy.
    wire _unused_to_int = &{1'b0, s2_is_inf, s2_is_zero, 1'b0};

    wire             round_increment = s2_guard & (s2_lost_sticky | s2_mag[0]);
    wire [WINT-1:0]  mag_rounded     = s2_mag + {{(WINT-1){1'b0}}, round_increment};

    // Predict saturation from the pre-rounded magnitude in parallel with the rounding carry chain.
    wire top_pre     =  s2_mag[WINT-1];
    wire low_all_set = &s2_mag[WINT-2:0];
    wire low_any_set = |s2_mag[WINT-2:0];

    wire overflow_pos = top_pre | (round_increment & low_all_set);
    wire overflow_neg = top_pre & (round_increment | low_any_set);
    wire overflow_now = s2_oor | (s2_sign ? overflow_neg : overflow_pos);

    // Saturation magnitudes (as unsigned WINT bits). INT_NEG_MAG (= 0x80..0) negates back to INT_MIN.
    localparam [WINT-1:0] INT_NEG_MAG = {1'b1, {(WINT-1){1'b0}}};
    localparam [WINT-1:0] INT_MAX     = {1'b0, {(WINT-1){1'b1}}};

    wire [WINT-1:0] mag_sat_overflow = s2_sign ? INT_NEG_MAG : INT_MAX;
    wire [WINT-1:0] mag_sat          = overflow_now ? mag_sat_overflow : mag_rounded;

    // -- Stage 3 register.
    reg            s3_valid;
    reg            s3_sign;
    reg [WINT-1:0] s3_mag_sat;

    always @(posedge clk) begin
        if (rst) begin
            s3_valid <= 1'b0;
        end else begin
            s3_valid <= s2_valid;
        end
        s3_sign    <= s2_sign;
        s3_mag_sat <= mag_sat;
    end

    // -- Stage 3 -> Stage 4: apply sign by two's-complement negation.
    wire [WINT-1:0] y_pre_unsigned = s3_sign ? (~s3_mag_sat + {{(WINT-1){1'b0}}, 1'b1}) : s3_mag_sat;

    // -- Stage 4 register (output).
    reg                   s4_valid;
    reg signed [WINT-1:0] s4_y;

    always @(posedge clk) begin
        if (rst) begin
            s4_valid <= 1'b0;
        end else begin
            s4_valid <= s3_valid;
        end
        s4_y <= $signed(y_pre_unsigned);
    end

    assign out_valid = s4_valid;
    assign y         = s4_y;
endmodule

`default_nettype wire
