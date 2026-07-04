/// Formal harness: _zkf_pack DUT vs zkf_pack_ref, for either STAGE_OUTPUT.
/// Single-pulse: drive arbitrary inputs at cycle 1 with rst=0/in_valid=1, else in_valid=0.
/// STAGE_OUTPUT=1: result is registered, valid at cycle 2 (compare against the shadowed inputs).
/// STAGE_OUTPUT=0: result is combinational, valid at cycle 1 (compare against the live inputs).

`default_nettype none

module zkf_pack_eq #(parameter WEXP = 6, parameter WMAN = 18, parameter WEXP_UNBIASED = WEXP + 2,
                     parameter STAGE_OUTPUT = 0) (
    input wire clk,
    input wire rst,
    input wire in_valid,
    input wire                            sign,
    input wire                            force_zero,
    input wire                            force_inf,
    input wire signed [WEXP_UNBIASED-1:0] exp_unbiased,
    input wire                 [WMAN-1:0] significand,
    input wire                            guard,
    input wire                            round_bit,
    input wire                            sticky
);
    localparam WFULL    = WEXP + WMAN;
    localparam T_RESULT = 1 + STAGE_OUTPUT;  // combinational: result at cycle 1; registered: at cycle 2

    reg [3:0] cycle = 4'd0;
    always @(posedge clk) cycle <= (cycle == 4'd15) ? cycle : cycle + 4'd1;

    always @(*) begin
        if (cycle == 4'd0) begin
            assume(rst == 1'b1);
            assume(in_valid == 1'b0);
        end else if (cycle == 4'd1) begin
            assume(rst == 1'b0);
            assume(in_valid == 1'b1);
        end else begin
            assume(rst == 1'b0);
            assume(in_valid == 1'b0);
        end
    end

    // Shadow latches.
    reg                            sh_sign;
    reg                            sh_force_zero;
    reg                            sh_force_inf;
    reg signed [WEXP_UNBIASED-1:0] sh_exp_unbiased;
    reg                 [WMAN-1:0] sh_significand;
    reg                            sh_guard;
    reg                            sh_round_bit;
    reg                            sh_sticky;
    always @(posedge clk) if (cycle == 4'd1) begin
        sh_sign         <= sign;
        sh_force_zero   <= force_zero;
        sh_force_inf    <= force_inf;
        sh_exp_unbiased <= exp_unbiased;
        sh_significand  <= significand;
        sh_guard        <= guard;
        sh_round_bit    <= round_bit;
        sh_sticky       <= sticky;
    end

    // DUT.
    wire             dut_valid;
    wire [WFULL-1:0] dut_y;
    _zkf_pack #(.WEXP(WEXP), .WMAN(WMAN), .WEXP_UNBIASED(WEXP_UNBIASED), .STAGE_OUTPUT(STAGE_OUTPUT)) u_dut (
        .clk(clk), .rst(rst), .in_valid(in_valid),
        .sign(sign), .force_zero(force_zero), .force_inf(force_inf),
        .exp_unbiased(exp_unbiased), .significand(significand),
        .guard(guard), .round(round_bit), .sticky(sticky),
        .out_valid(dut_valid), .y(dut_y)
    );

    // Reference operands: registered output compares against the shadow (result one cycle late); combinational
    // output compares against the live inputs (result in the input's own cycle).
    wire                            r_sign         = STAGE_OUTPUT ? sh_sign         : sign;
    wire                            r_force_zero   = STAGE_OUTPUT ? sh_force_zero   : force_zero;
    wire                            r_force_inf    = STAGE_OUTPUT ? sh_force_inf    : force_inf;
    wire signed [WEXP_UNBIASED-1:0] r_exp_unbiased = STAGE_OUTPUT ? sh_exp_unbiased : exp_unbiased;
    wire                 [WMAN-1:0] r_significand  = STAGE_OUTPUT ? sh_significand  : significand;
    wire                            r_guard        = STAGE_OUTPUT ? sh_guard        : guard;
    wire                            r_round_bit    = STAGE_OUTPUT ? sh_round_bit    : round_bit;
    wire                            r_sticky       = STAGE_OUTPUT ? sh_sticky       : sticky;

    // Reference.
    wire [WFULL-1:0] ref_y;
    zkf_pack_ref #(.WEXP(WEXP), .WMAN(WMAN), .WEXP_UNBIASED(WEXP_UNBIASED)) u_ref (
        .sign(r_sign), .force_zero(r_force_zero), .force_inf(r_force_inf),
        .exp_unbiased(r_exp_unbiased), .significand(r_significand),
        .guard(r_guard), .round_bit(r_round_bit), .sticky(r_sticky),
        .y(ref_y)
    );

    always @(posedge clk) begin
        if (cycle == T_RESULT) begin
            assert(dut_valid == 1'b1);
            assert(dut_y == ref_y);
        end
        // The registered output is silent the cycle before the result; the combinational output is valid at cycle 1.
        if (STAGE_OUTPUT != 0 && cycle == 4'd1) assert(dut_valid == 1'b0);
    end
endmodule

`default_nettype wire
