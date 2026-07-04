/// Formal harness: zkf_is_finite DUT vs. direct bit-twiddle spec.

`default_nettype none

module zkf_is_finite_eq #(parameter WEXP = 6, parameter WMAN = 18) (
    input wire [WEXP+WMAN-1:0] x
);
    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;

    wire y_dut;
    zkf_is_finite #(.WEXP(WEXP), .WMAN(WMAN)) u_dut (.x(x), .y(y_dut));

    wire y_ref = ~(&x[WFULL-2:WFRAC]);

    // x is infinity iff exponent field is all-ones; otherwise finite.
    wire is_inf_explicit = (x[WFULL-2:WFRAC] == {WEXP{1'b1}});

    always @(*) begin
        assert(y_dut == y_ref);
        assert(y_dut != is_inf_explicit);
        // Spec one more way: zero patterns are also finite.
        if (x[WFULL-2:WFRAC] == {WEXP{1'b0}}) assert(y_dut == 1'b1);
    end
endmodule

`default_nettype wire
