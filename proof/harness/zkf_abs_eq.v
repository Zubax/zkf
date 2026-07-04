/// Formal harness: zkf_abs DUT vs. direct bit-twiddle spec.

`default_nettype none

module zkf_abs_eq #(parameter WEXP = 6, parameter WMAN = 18) (
    input wire [WEXP+WMAN-1:0] x
);
    localparam WFULL = WEXP + WMAN;

    wire [WFULL-1:0] y_dut;
    zkf_abs #(.WEXP(WEXP), .WMAN(WMAN)) u_dut (.x(x), .y(y_dut));

    wire [WFULL-1:0] y_ref = {1'b0, x[WFULL-2:0]};

    always @(*) begin
        assert(y_dut == y_ref);
        assert(y_dut[WFULL-1] == 1'b0);
        // Idempotence: applying abs to the output yields the same value.
        assert(y_dut == {1'b0, y_dut[WFULL-2:0]});
    end
endmodule

`default_nettype wire
