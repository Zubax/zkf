/// Formal harness: _zkf_div_radix4_step DUT vs zkf_div_radix4_step_ref.
/// Combinational, BMC depth 1. Tests with the divider invariants (den normalized, rem < den).

`default_nettype none

module zkf_div_radix4_step_eq #(parameter WMAN = 18) (
    input wire [WMAN-1:0] den,
    input wire [WMAN-1:0] rem
);
    localparam WREM4 = WMAN + 2;

    // Compute den3 from den exactly as _zkf_div_core does (3*den).
    wire [WREM4-1:0] den3 = {1'b0, den, 1'b0} + {2'b00, den};

    // DUT.
    wire [WMAN-1:0] dut_rem_next;
    wire      [1:0] dut_digit;
    _zkf_div_radix4_step #(.WMAN(WMAN)) u_dut (
        .den(den), .den3(den3), .rem(rem),
        .rem_next(dut_rem_next), .digit(dut_digit)
    );

    // Reference.
    wire [WMAN-1:0] ref_rem_next;
    wire      [1:0] ref_digit;
    zkf_div_radix4_step_ref #(.WMAN(WMAN)) u_ref (
        .den(den), .rem(rem),
        .rem_next(ref_rem_next), .digit(ref_digit)
    );

    // Sanity check: den3 we computed equals 3*den in the reference's wider arithmetic.
    wire [WREM4-1:0] den3_expected = {2'b00, den} + {1'b0, den, 1'b0};

    always @(*) begin
        // Operand invariants from the divider context: den is normalized (MSB = 1) and rem < den.
        assume(den[WMAN-1] == 1'b1);
        assume(rem < den);

        // Equivalence.
        assert(dut_digit    == ref_digit);
        assert(dut_rem_next == ref_rem_next);

        // Side invariants from the spec.
        assert(den3 == den3_expected);
        // After one step, the remainder still fits below den (invariant for the chain).
        assert(dut_rem_next < den);
    end
endmodule

`default_nettype wire
