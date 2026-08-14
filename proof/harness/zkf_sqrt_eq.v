/// Formal harness: zkf_sqrt DUT vs zkf_sqrt_ref.
/// LATENCY is supplied by run_proofs.py from the shared Python model.

`default_nettype none

module zkf_sqrt_eq #(parameter WEXP = 4, parameter WMAN = 6, parameter STAGE_INPUT = 0,
                     parameter STAGE_PACK = 0, parameter STAGE_OUTPUT = 0, parameter LATENCY = 0) (
    input wire clk,
    input wire rst,
    input wire in_valid,
    input wire [WEXP+WMAN-1:0] x
);
    localparam WFULL            = WEXP + WMAN;
    localparam integer T_RESULT = 1 + LATENCY;
    localparam integer CYCLE_W  = (T_RESULT < 2) ? 2 : $clog2(T_RESULT + 2);

    reg [CYCLE_W-1:0] cycle = {CYCLE_W{1'b0}};
    always @(posedge clk) if (cycle < T_RESULT + 1) cycle <= cycle + 1'b1;

    always @(*) begin
        if (cycle == 0) begin
            assume(rst == 1'b1);
            assume(in_valid == 1'b0);
        end else if (cycle == 1) begin
            assume(rst == 1'b0);
            assume(in_valid == 1'b1);
        end else begin
            assume(rst == 1'b0);
            assume(in_valid == 1'b0);
        end
    end

    reg [WFULL-1:0] x_shadow;
    always @(posedge clk) if (cycle == 1) x_shadow <= x;

    wire             dut_valid;
    wire [WFULL-1:0] dut_y;
    wire             dut_de;
    zkf_sqrt #(
        .WEXP(WEXP),
        .WMAN(WMAN),
        .STAGE_INPUT(STAGE_INPUT),
        .STAGE_PACK(STAGE_PACK),
        .STAGE_OUTPUT(STAGE_OUTPUT),
        .LATENCY(LATENCY)
    ) u_dut (
        .clk(clk), .rst(rst), .in_valid(in_valid),
        .x(x),
        .out_valid(dut_valid), .y(dut_y), .domain_error(dut_de)
    );

    wire [WFULL-1:0]      ref_y;
    wire                  ref_de;
    wire [2*WMAN-1:0]     ref_rad;
    wire [WMAN-1:0]       ref_lo;
    wire signed [WEXP+1:0] ref_exp;
    zkf_sqrt_ref #(.WEXP(WEXP), .WMAN(WMAN)) u_ref (
        .x(x_shadow),
        .y(ref_y), .domain_error(ref_de),
        .dbg_y(ref_rad), .dbg_lo(ref_lo), .dbg_exp(ref_exp)
    );

    // Sanity on the reference's unrounded root: lo^2 <= y < (lo+1)^2 (lo widened by one bit before +1).
    wire [WMAN:0]     lo_ext   = {1'b0, ref_lo};
    wire [WMAN:0]     lo_p1    = lo_ext + {{WMAN{1'b0}}, 1'b1};
    wire [2*WMAN+1:0] lo_sq    = lo_ext * lo_ext;
    wire [2*WMAN+1:0] lo_p1_sq = lo_p1 * lo_p1;
    wire [2*WMAN+1:0] rad_ext  = {2'b00, ref_rad};

    // For normal inputs the reference's biased result exponent must be a finite normal value, proving its
    // WEXP-bit pack truncation safe (and the DUT's ASSUME_NO_OVERFLOW claim along with it).
    wire x_sh_normal = (|x_shadow[WFULL-2:WMAN-1]) && !(&x_shadow[WFULL-2:WMAN-1]);
    wire signed [WEXP+1:0] exp_max_finite = {2'b00, {(WEXP-1){1'b1}}, 1'b0};

    always @(posedge clk) begin
        if (cycle == T_RESULT) begin
            assert(dut_valid == 1'b1);
            assert(dut_y == ref_y);
            assert(dut_de == ref_de);
            assert(lo_sq <= rad_ext);
            assert(rad_ext < lo_p1_sq);
            if (x_sh_normal) begin
                assert(ref_exp >= 1);
                assert(ref_exp <= exp_max_finite);
            end
        end
        if (cycle >= 1 && cycle < T_RESULT) assert(dut_valid == 1'b0);
    end
endmodule

`default_nettype wire
