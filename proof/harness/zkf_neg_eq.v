/// Formal harness: zkf_neg DUT vs. direct bit-twiddle spec.

`default_nettype none

module zkf_neg_eq #(parameter WEXP = 6, parameter WMAN = 18) (
    input wire [WEXP+WMAN-1:0] x
);
    localparam WFULL = WEXP + WMAN;

    wire [WFULL-1:0] y_dut;
    zkf_neg #(.WEXP(WEXP), .WMAN(WMAN)) u_dut (.x(x), .y(y_dut));

    wire [WFULL-1:0] y_ref = {~x[WFULL-1], x[WFULL-2:0]};

    // Involution: zkf_neg(zkf_neg(x)) == x as a bit-pattern.
    wire [WFULL-1:0] y_dut_twice;
    zkf_neg #(.WEXP(WEXP), .WMAN(WMAN)) u_dut_twice (.x(y_dut), .y(y_dut_twice));

    always @(*) begin
        assert(y_dut == y_ref);
        assert(y_dut[WFULL-2:0] == x[WFULL-2:0]);
        assert(y_dut_twice == x);
    end
endmodule

`default_nettype wire
