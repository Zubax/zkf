/// Extracts the raw unbiased exponent as an integer, including limit cases of zero (-bias) and infinity (bias+1).
/// In principle follows C++'s std::ilogb() modulo edge case handling.
/// Result and zero/infinity ignore sign and fraction; negative reports the raw sign bit, including negative zero.
/// WINT >= WEXP+1. Outputs are meaningful with out_valid; II=1; latency is 1 + STAGE_INPUT cycles.

`default_nettype none

module zkf_ilog2 #(
    parameter WEXP        = 6,
    parameter WMAN        = 18,
    parameter WINT        = 32,
    parameter STAGE_INPUT = 0,
    parameter LATENCY     = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] a,  // the input float to extract the exponent from

    output reg                    out_valid,
    output wire signed [WINT-1:0] y,        // unbiased exponent of abs(y); well-defined for any inputs
    output reg                    zero,     // argument is zero (exp=-bias)
    output reg                    infinity, // argument is a positive or negative infinity (exp=bias+1)
    output reg                    negative  // argument sign bit
);
    localparam LATENCY_REF = 1 + STAGE_INPUT;
    localparam signed [WEXP:0] BIAS = {2'b00, {(WEXP-1){1'b1}}};
    generate
        if ((WEXP < 2) || (WMAN < 4)) begin : g_invalid_wm
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        if (WINT < WEXP + 1) begin : g_invalid_wint
            _zkf_invalid_ilog2_wint u_invalid();
        end
        if (STAGE_INPUT < 0) begin : g_invalid_stage_input
            _zkf_invalid_stage_input u_invalid();
        end
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    wire in_valid_q;
    wire [WEXP:0] a_q;
    zkf_pipe #(.W(WEXP + 1), .N(STAGE_INPUT)) u_input_pipe (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in(a[WEXP+WMAN-1:WMAN-1]),
        .out_valid(in_valid_q), .out(a_q)
    );

    reg signed [WEXP:0] result;
    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0;
        end else begin
            out_valid <= in_valid_q;
        end
        result   <= $signed({1'b0, a_q[WEXP-1:0]}) - BIAS;
        zero     <= ~|a_q[WEXP-1:0];
        infinity <=  &a_q[WEXP-1:0];
        negative <=   a_q[WEXP];
    end

    assign y = result;
endmodule

`default_nettype wire
