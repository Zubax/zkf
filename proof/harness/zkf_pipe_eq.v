/// Formal harness: zkf_pipe DUT.
/// Spec: at every cycle after the initial reset, out_valid mirrors shadow_valid[N-1] and (when valid) out
/// equals shadow_data[N-1]. Both pipelines are driven by the same (in, in_valid, rst) sequence so their
/// internal state must remain equal for all time.
///
/// BMC strategy: assume rst=1 at cycle 0 and rst=0 thereafter. With matching reset-gated shadow registers
/// the data-pipe registers also align (their inputs are identical) so the equivalence holds from cycle 1 on.

`default_nettype none

module zkf_pipe_eq #(parameter W = 24, parameter N = 4) (
    input wire clk,
    input wire rst,
    input wire in_valid,
    input wire [W-1:0] in
);
    reg [3:0] cycle = 4'd0;
    always @(posedge clk) cycle <= (cycle == 4'd15) ? cycle : cycle + 4'd1;

    // Drive rst=1 at cycle 0 only; thereafter rst=0. in / in_valid remain free.
    always @(*) begin
        if (cycle == 4'd0) assume(rst == 1'b1);
        else               assume(rst == 1'b0);
    end

    // DUT.
    wire         dut_valid;
    wire [W-1:0] dut_out;
    zkf_pipe #(.W(W), .N(N)) u_dut (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in(in),
        .out_valid(dut_valid), .out(dut_out)
    );

    // Shadow data pipeline. Initialize to zero so basecase has a known starting state.
    reg [W-1:0] shadow_data [0:N-1];
    integer i;
    initial begin
        for (i = 0; i < N; i = i + 1) shadow_data[i] = {W{1'b0}};
    end
    always @(posedge clk) begin
        shadow_data[0] <= in;
        for (i = 1; i < N; i = i + 1) shadow_data[i] <= shadow_data[i-1];
    end

    // Shadow validity pipeline.
    reg [N-1:0] shadow_valid = {N{1'b0}};
    always @(posedge clk) begin
        if (rst) begin
            shadow_valid <= {N{1'b0}};
        end else begin
            shadow_valid[0] <= in_valid;
            for (i = 1; i < N; i = i + 1) shadow_valid[i] <= shadow_valid[i-1];
        end
    end

    // Validity must match every cycle after the initial reset has been observed.
    // Data is meaningful only when valid is asserted; data registers free-run with arbitrary initial
    // values, so the data assertion is only meaningful once valid propagates a real input through.
    always @(*) begin
        if (cycle >= 4'd1) begin
            assert(dut_valid == shadow_valid[N-1]);
            if (shadow_valid[N-1]) begin
                assert(dut_out == shadow_data[N-1]);
            end
        end
    end
endmodule

`default_nettype wire
