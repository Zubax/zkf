/// Streamed saturating cast to signed two's-complement integer. round_mode: 0=RNTE, 1=floor, 2=ceil, 3=truncation.
/// The pipeline latency is 4 + STAGE_INPUT cycles.

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
    input wire           [1:0] round_mode,

    output wire                   out_valid,
    output wire signed [WINT-1:0] y
);
    localparam LATENCY_REF = 4 + STAGE_INPUT;
    localparam ROUND_NEAREST_EVEN = 2'd0;
    localparam ROUND_FLOOR        = 2'd1;
    localparam ROUND_CEIL         = 2'd2;
    localparam ROUND_TRUNC        = 2'd3;
    generate
        if ((WEXP < 2) || (WMAN < 4) || (WINT < 2)) begin : g_invalid
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    reg [2:0] round_control_in;
    always @* begin
        case (round_mode)
            ROUND_NEAREST_EVEN: round_control_in = 3'b001;
            ROUND_FLOOR:        round_control_in = 3'b010;
            ROUND_CEIL:         round_control_in = 3'b100;
            ROUND_TRUNC:        round_control_in = 3'b000;
            default:            round_control_in = 3'b000;
        endcase
    end

    wire [2:0] round_control;
    zkf_pipe #(.W(3), .N(STAGE_INPUT + 2)) u_round_control_pipe (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in(round_control_in),
        .out_valid(), .out(round_control)
    );

    wire             s2_valid;
    wire [WINT-1:0]  s2_mag;
    wire             s2_guard;
    wire             s2_lost_sticky;
    wire             s2_sign;
    wire             s2_oor;
    _zkf_to_fixpoint #(.WEXP(WEXP), .WMAN(WMAN), .WI(WINT), .FF(0), .STAGE_INPUT(STAGE_INPUT)) u_to_fixpoint (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .a(a),
        .out_valid(s2_valid),
        .mag(s2_mag), .guard(s2_guard), .lost_sticky(s2_lost_sticky),
        .sign(s2_sign), .is_inf(), .is_zero(), .oor(s2_oor)
    );

    wire discarded        =  s2_guard | s2_lost_sticky;
    wire increment_rnte   =  s2_guard & (s2_lost_sticky | s2_mag[0]);
    wire increment_floor  =  s2_sign  & discarded;
    wire increment_ceil   = ~s2_sign  & discarded;
    wire round_increment  = (round_control[0] & increment_rnte) |
                            (round_control[1] & increment_floor) |
                            (round_control[2] & increment_ceil);
    wire [WINT-1:0] mag_rounded = s2_mag + {{(WINT-1){1'b0}}, round_increment};

    wire top_pre     =  s2_mag[WINT-1];
    wire low_all_set = &s2_mag[WINT-2:0];

    wire overflow_pos = top_pre | (round_increment & low_all_set);
    wire clamp_pos = s2_oor | overflow_pos;
    wire clamp_neg = s2_oor | top_pre;
    wire [WINT-2:0] mag_sat_pos_low = mag_rounded[WINT-2:0] | {(WINT-1){clamp_pos}};
    wire [WINT-2:0] mag_sat_neg_low = mag_rounded[WINT-2:0] & {(WINT-1){~clamp_neg}};
    wire [WINT-1:0] mag_sat_pos = {1'b0, mag_sat_pos_low};
    wire [WINT-1:0] mag_sat_neg = {mag_rounded[WINT-1] | clamp_neg, mag_sat_neg_low};
    wire [WINT-1:0] mag_sat = s2_sign ? mag_sat_neg : mag_sat_pos;

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

    wire [WINT-1:0] y_pre_unsigned = s3_sign ? (~s3_mag_sat + {{(WINT-1){1'b0}}, 1'b1}) : s3_mag_sat;

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
