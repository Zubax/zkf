// Streamed Zubax Kulibin float adder.
// Subtraction a - b is achieved by inverting the sign bit of b.
//
// LATENCY = 4 + STAGE_*

`default_nettype none

module zkf_add #(
    parameter WEXP            = 6,    // exponent field width
    parameter WMAN            = 18,   // significand precision including the hidden bit
    parameter STAGE_INPUT     = 0,    // number of input register stages
    parameter STAGE_DECODE    = 0,    // registers decoded operands and their magnitude comp ahead of operand selection
    parameter STAGE_ALIGN     = 0,    // see _zkf_add_core
    parameter STAGE_NORMALIZE = 0,    // see _zkf_add_core
    parameter STAGE_PACK      = 0,    // see _zkf_add_core
    parameter STAGE_OUTPUT    = 0,    // see _zkf_add_core
    parameter LATENCY         = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] a,
    input wire [WEXP+WMAN-1:0] b,

    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    localparam LATENCY_REF = 4 + STAGE_INPUT + STAGE_DECODE + STAGE_ALIGN + STAGE_NORMALIZE + STAGE_PACK + STAGE_OUTPUT;
    generate
        if ((WEXP < 2) || (WMAN < 4)) begin : g_invalid_wman
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        if ((STAGE_DECODE != 0) && (STAGE_DECODE != 1)) begin : g_invalid_stage_decode
            _zkf_invalid_stage_decode u_invalid();
        end
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;

    wire             in_valid_q;
    wire [WFULL-1:0] a_q;
    wire [WFULL-1:0] b_q;
    zkf_pipe #(.W(2*WFULL), .N(STAGE_INPUT)) u_input_pipe (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in({b, a}),
        .out_valid(in_valid_q), .out({b_q, a_q})
    );

    // Exponent-zero operands are zero regardless of sign and fraction. Zero and non-finite operands enter the datapath
    // as zero magnitudes: a zero leaves the other operand unchanged, a non-finite one is overridden further on.
    wire [WEXP-1:0] a_exp    = a_q[WFULL-2:WFRAC];
    wire [WEXP-1:0] b_exp    = b_q[WFULL-2:WFRAC];
    wire            a_finite = (|a_exp) && !(&a_exp);
    wire            b_finite = (|b_exp) && !(&b_exp);
    // One unsigned compare of {exponent, fraction} orders finite operands by magnitude, so the larger one also has
    // the larger or equal exponent; the order of anything else does not matter.
    wire            a_ge_b;
    _zkf_add_ge #(.W(WFULL-1)) u_mag_ge (.a(a_q[WFULL-2:0]), .b(b_q[WFULL-2:0]), .ge(a_ge_b));

    wire            d_valid;
    wire            d_a_sign, d_b_sign, d_a_inf, d_b_inf, d_a_ge_b;
    wire [WEXP-1:0] d_a_exp, d_b_exp;
    wire [WMAN-1:0] d_a_sig, d_b_sig;
    zkf_pipe #(.W(5 + 2*WEXP + 2*WMAN), .N(STAGE_DECODE)) u_decode_pipe (
        .clk(clk), .rst(rst), .in_valid(in_valid_q),
        .in({
            a_q[WFULL-1], b_q[WFULL-1], &a_exp, &b_exp, a_ge_b,
            a_finite ? a_exp : {WEXP{1'b0}}, b_finite ? b_exp : {WEXP{1'b0}},
            a_finite ? {1'b1, a_q[WFRAC-1:0]} : {WMAN{1'b0}}, b_finite ? {1'b1, b_q[WFRAC-1:0]} : {WMAN{1'b0}}
        }),
        .out_valid(d_valid),
        .out({d_a_sign, d_b_sign, d_a_inf, d_b_inf, d_a_ge_b, d_a_exp, d_b_exp, d_a_sig, d_b_sig})
    );

    wire            same_sign = ~(d_a_sign ^ d_b_sign);
    wire [WEXP-1:0] large_exp = d_a_ge_b ? d_a_exp : d_b_exp;
    wire [WEXP-1:0] small_exp = d_a_ge_b ? d_b_exp : d_a_exp;
    _zkf_add_core #(
        .WEXP(WEXP), .WMAN(WMAN), .STAGE_ALIGN(STAGE_ALIGN), .STAGE_NORMALIZE(STAGE_NORMALIZE),
        .STAGE_PACK(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT)
    ) u_core (
        .clk(clk), .rst(rst),
        .in_valid(d_valid),
        .finite_sign(d_a_ge_b ? d_a_sign : d_b_sign),
        .inf_sign((d_a_inf & d_a_sign) | (d_b_inf & d_b_sign)),
        .same_sign(same_sign),
        .force_zero(d_a_inf && d_b_inf && !same_sign),
        .force_inf(d_a_inf || d_b_inf),
        .exp({1'b0, large_exp}),
        .shamt(large_exp - small_exp),
        .larger(d_a_ge_b ? d_a_sig : d_b_sig),
        .smaller(d_a_ge_b ? d_b_sig : d_a_sig),
        .out_valid(out_valid), .y(y)
    );
endmodule


// Compare unsigned values through an explicit carry-chain-friendly subtraction; enables much better timings than
// the ordinary comparison operator. Related:
// https://stackoverflow.com/questions/60844496/does-subtraction-need-less-resource-than-comparison-symbol-in-verilog
module _zkf_add_ge #(parameter W = 18) (input wire [W-1:0] a, input wire [W-1:0] b, output wire ge);
    wire [W:0] diff = {1'b0, a} - {1'b0, b};
    assign ge = !diff[W];
endmodule


`default_nettype wire
