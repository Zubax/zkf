/// Formal harness: zkf_saturate DUT vs. explicit case-analysis spec.

`default_nettype none

module zkf_saturate_eq #(parameter WEXP = 6, parameter WMAN = 18) (
    input wire [WEXP+WMAN-1:0] x
);
    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;

    wire [WFULL-1:0] y_dut;
    zkf_saturate #(.WEXP(WEXP), .WMAN(WMAN)) u_dut (.x(x), .y(y_dut));

    wire x_inf  = (x[WFULL-2:WFRAC] == {WEXP{1'b1}});
    wire x_sign = x[WFULL-1];

    // Saturated value: same sign, exponent = EXP_INF - 1 (all-ones with LSB cleared), fraction = all-ones.
    wire [WFULL-1:0] sat_value = {x_sign, {(WEXP-1){1'b1}}, 1'b0, {WFRAC{1'b1}}};
    wire [WFULL-1:0] y_ref     = x_inf ? sat_value : x;

    // Saturate composed with is_finite: any input becomes finite afterwards.
    wire y_dut_finite_check;
    zkf_is_finite #(.WEXP(WEXP), .WMAN(WMAN)) u_is_finite (.x(y_dut), .y(y_dut_finite_check));

    // Idempotence: saturating again is a no-op.
    wire [WFULL-1:0] y_dut_twice;
    zkf_saturate #(.WEXP(WEXP), .WMAN(WMAN)) u_dut_twice (.x(y_dut), .y(y_dut_twice));

    always @(*) begin
        assert(y_dut == y_ref);
        assert(y_dut_finite_check == 1'b1);
        assert(y_dut_twice == y_dut);
        // Finite inputs are returned unchanged.
        if (!x_inf) assert(y_dut == x);
    end
endmodule

`default_nettype wire
