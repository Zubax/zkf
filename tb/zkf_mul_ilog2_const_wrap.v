/// Testbench harness for zkf_mul_ilog2_const. Each output port is the result of a separate DUT instance with a
/// distinct K. The set is chosen to span identity (K=0), small shifts (K=±1), midrange shifts, and the boundary
/// values K=EXP_MAX_FINITE-1 and K=-EXP_MAX_FINITE. The K values are derived from WEXP so that every instance stays
/// inside the allowed parameter range, and so the same wrap module can be elaborated at every (WEXP, WMAN) the test
/// matrix uses.
///
/// At WEXP=2 the valid K range collapses to {-2, -1, 0, 1}, which causes the midrange instances to coincide with the
/// identity and unit-shift instances. The Python test computes the same K values and checks every port, so the
/// duplication is harmless.

`default_nettype none

module zkf_mul_ilog2_const_wrap #(
    parameter WEXP         = 6,
    parameter WMAN         = 18,
    parameter STAGE_INPUT  = 0,
    parameter STAGE_DECODE = 0
) (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 in_valid,
    input  wire [WEXP+WMAN-1:0] a,
    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y_k0,
    output wire [WEXP+WMAN-1:0] y_kp1,
    output wire [WEXP+WMAN-1:0] y_kn1,
    output wire [WEXP+WMAN-1:0] y_kp_mid,
    output wire [WEXP+WMAN-1:0] y_kn_mid,
    output wire [WEXP+WMAN-1:0] y_kp_max,
    output wire [WEXP+WMAN-1:0] y_kn_max
);
    localparam EXP_MAX_FINITE = (1 << WEXP) - 2;
    localparam K_MAX_POS      = EXP_MAX_FINITE - 1;
    localparam K_MAX_NEG      = -EXP_MAX_FINITE;
    localparam K_MID_POS      = (EXP_MAX_FINITE - 1) / 2;
    localparam K_MID_NEG      = -((EXP_MAX_FINITE - 1) / 2);

    wire ov_k0, ov_kp1, ov_kn1, ov_kp_mid, ov_kn_mid, ov_kp_max, ov_kn_max;

    zkf_mul_ilog2_const #(
        .WEXP(WEXP), .WMAN(WMAN), .K(0), .STAGE_INPUT(STAGE_INPUT), .STAGE_DECODE(STAGE_DECODE)
    ) u_k0 (
        .clk(clk), .rst(rst), .in_valid(in_valid), .a(a),
        .out_valid(ov_k0), .y(y_k0)
    );
    zkf_mul_ilog2_const #(
        .WEXP(WEXP), .WMAN(WMAN), .K(1), .STAGE_INPUT(STAGE_INPUT), .STAGE_DECODE(STAGE_DECODE)
    ) u_kp1 (
        .clk(clk), .rst(rst), .in_valid(in_valid), .a(a),
        .out_valid(ov_kp1), .y(y_kp1)
    );
    zkf_mul_ilog2_const #(
        .WEXP(WEXP), .WMAN(WMAN), .K(-1), .STAGE_INPUT(STAGE_INPUT), .STAGE_DECODE(STAGE_DECODE)
    ) u_kn1 (
        .clk(clk), .rst(rst), .in_valid(in_valid), .a(a),
        .out_valid(ov_kn1), .y(y_kn1)
    );
    zkf_mul_ilog2_const #(
        .WEXP(WEXP), .WMAN(WMAN), .K(K_MID_POS), .STAGE_INPUT(STAGE_INPUT), .STAGE_DECODE(STAGE_DECODE)
    ) u_kp_mid (
        .clk(clk), .rst(rst), .in_valid(in_valid), .a(a),
        .out_valid(ov_kp_mid), .y(y_kp_mid)
    );
    zkf_mul_ilog2_const #(
        .WEXP(WEXP), .WMAN(WMAN), .K(K_MID_NEG), .STAGE_INPUT(STAGE_INPUT), .STAGE_DECODE(STAGE_DECODE)
    ) u_kn_mid (
        .clk(clk), .rst(rst), .in_valid(in_valid), .a(a),
        .out_valid(ov_kn_mid), .y(y_kn_mid)
    );
    zkf_mul_ilog2_const #(
        .WEXP(WEXP), .WMAN(WMAN), .K(K_MAX_POS), .STAGE_INPUT(STAGE_INPUT), .STAGE_DECODE(STAGE_DECODE)
    ) u_kp_max (
        .clk(clk), .rst(rst), .in_valid(in_valid), .a(a),
        .out_valid(ov_kp_max), .y(y_kp_max)
    );
    zkf_mul_ilog2_const #(
        .WEXP(WEXP), .WMAN(WMAN), .K(K_MAX_NEG), .STAGE_INPUT(STAGE_INPUT), .STAGE_DECODE(STAGE_DECODE)
    ) u_kn_max (
        .clk(clk), .rst(rst), .in_valid(in_valid), .a(a),
        .out_valid(ov_kn_max), .y(y_kn_max)
    );

    // All instances share the same pipeline depth, so the common out_valid is whichever flag.
    assign out_valid = ov_k0;
endmodule

`default_nettype wire
