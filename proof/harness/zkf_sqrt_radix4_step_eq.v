/// Formal harness: _zkf_sqrt_radix4_step DUT vs zkf_sqrt_radix4_step_ref, combinational, BMC depth 1.
/// One checker per prefix width: the widths used by the end-to-end proofs (WQ=1, 3, 5: WMAN=6/7 stages,
/// WQ=1 being the constant-folded decode-stage first digit) plus the late-stage widths WQ=17 (w6m18
/// production) and WQ=51 (WMAN=53 class), catching unsized or signed constant mistakes and borrow
/// extraction past 32 bits. Each checker derives M = 3Q+1 exactly as
/// the pipeline maintains it (the step contract), assumes the normalized-prefix and remainder invariants,
/// and asserts equivalence plus the recurrence identity and the 0 <= rem' <= 2Q' invariant.

`default_nettype none

module zkf_sqrt_step_check #(parameter WQ = 3) (
    input wire [WQ-1:0] q,
    input wire [WQ:0]   rem,
    input wire [3:0]    n4
);
    // M = 3q+1, the invariant the pipeline maintains for the DUT's m input.
    wire [WQ+1:0] m = {1'b0, q, 1'b0} + {2'b00, q} + {{(WQ+1){1'b0}}, 1'b1};

    wire [1:0]    dut_digit;
    wire [WQ+2:0] dut_rem_next;
    wire [WQ+3:0] dut_m_next;
    _zkf_sqrt_radix4_step #(.WQ(WQ)) u_dut (
        .q(q), .m(m), .rem(rem), .n4(n4),
        .digit(dut_digit), .rem_next(dut_rem_next), .m_next(dut_m_next)
    );

    wire [1:0]    ref_digit;
    wire [WQ+2:0] ref_rem_next;
    wire [WQ+3:0] ref_m_next;
    zkf_sqrt_radix4_step_ref #(.WQ(WQ)) u_ref (
        .q(q), .rem(rem), .n4(n4),
        .digit(ref_digit), .rem_next(ref_rem_next), .m_next(ref_m_next)
    );

    // Widened views for the recurrence-identity and invariant checks.
    localparam WW = WQ + 6;
    wire [WW-1:0]  minuend    = {{(WW-WQ-5){1'b0}}, rem, n4};          // 16*rem + n4
    wire [WW-1:0]  qe         = {{(WW-WQ){1'b0}}, q};
    wire [WW-1:0]  de         = {{(WW-2){1'b0}}, dut_digit};
    wire [WW-1:0]  subtrahend = de * (8 * qe + de);                    // digit*(8q + digit)
    wire [WQ+1:0]  q_next     = {q, dut_digit};
    wire [WW-1:0]  rem_n_ext  = {{(WW-WQ-3){1'b0}}, dut_rem_next};
    wire [WW-1:0]  q_next2    = {{(WW-WQ-3){1'b0}}, q_next, 1'b0};     // 2*q_next
    wire [WQ+3:0]  m_next_dir = 3 * {2'b00, q_next} + 1;               // 3Q'+1, direct (not via the ref)

    always @(*) begin
        // Operand invariants from the recurrence context: normalized prefix, 0 <= rem <= 2q.
        assume(q[WQ-1] == 1'b1);
        assume({1'b0, rem} <= {1'b0, q, 1'b0});

        // Equivalence.
        assert(dut_digit    == ref_digit);
        assert(dut_rem_next == ref_rem_next);
        assert(dut_m_next   == ref_m_next);

        // Remainder recurrence identity, maximal-digit selection, and invariant preservation.
        assert(minuend == subtrahend + rem_n_ext);
        assert(rem_n_ext <= q_next2);                                  // rem' <= 2q' (maximality: d+1 overshoots)
        assert(dut_m_next == m_next_dir);
    end
endmodule


module zkf_sqrt_radix4_step_eq (
    input wire [0:0]  q1,  input wire [1:0]  rem1,  input wire [3:0] n4_1,
    input wire [2:0]  q3,  input wire [3:0]  rem3,  input wire [3:0] n4_3,
    input wire [4:0]  q5,  input wire [5:0]  rem5,  input wire [3:0] n4_5,
    input wire [16:0] q17, input wire [17:0] rem17, input wire [3:0] n4_17,
    input wire [50:0] q51, input wire [51:0] rem51, input wire [3:0] n4_51
);
    // WQ=1 is the constant-operand instantiation folded into the decode stage (q==1 forced by the assume).
    zkf_sqrt_step_check #(.WQ(1))  u_w1  (.q(q1),  .rem(rem1),  .n4(n4_1));
    zkf_sqrt_step_check #(.WQ(3))  u_w3  (.q(q3),  .rem(rem3),  .n4(n4_3));
    zkf_sqrt_step_check #(.WQ(5))  u_w5  (.q(q5),  .rem(rem5),  .n4(n4_5));
    zkf_sqrt_step_check #(.WQ(17)) u_w17 (.q(q17), .rem(rem17), .n4(n4_17));
    zkf_sqrt_step_check #(.WQ(51)) u_w51 (.q(q51), .rem(rem51), .n4(n4_51));
endmodule

`default_nettype wire
