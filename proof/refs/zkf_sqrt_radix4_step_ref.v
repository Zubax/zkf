/// Combinational reference for the radix-4 restoring square-root step.
/// Spec: pick the largest digit d in {0,1,2,3} such that d*(8q+d) <= 16*rem + n4; output
/// rem_next = 16*rem + n4 - d*(8q+d) and m_next = 3*{q,d} + 1.
/// This reference is structurally different from the DUT (priority-encode via explicit
/// constant-multiply comparisons; m_next recomputed from the new prefix, not maintained incrementally).

`default_nettype none

module zkf_sqrt_radix4_step_ref #(parameter WQ = 3) (
    input  wire [WQ-1:0] q,
    input  wire [WQ:0]   rem,
    input  wire [3:0]    n4,
    output reg  [1:0]    digit,
    output reg  [WQ+2:0] rem_next,
    output reg  [WQ+3:0] m_next
);
    localparam WW = WQ + 6;

    wire [WW-1:0] qe   = {{(WW-WQ){1'b0}}, q};
    wire [WW-1:0] mini = {{(WW-WQ-5){1'b0}}, rem, n4};
    wire [WW-1:0] s1   = 8 * qe + 1;
    wire [WW-1:0] s2   = 16 * qe + 4;
    wire [WW-1:0] s3   = 24 * qe + 9;

    reg [WW-1:0]   rem_ext;
    reg [WQ+1:0]   q_next;
    reg [WQ+3:0]   m_wide;

    always @(*) begin
        if (mini >= s3) begin
            digit   = 2'd3;
            rem_ext = mini - s3;
        end else if (mini >= s2) begin
            digit   = 2'd2;
            rem_ext = mini - s2;
        end else if (mini >= s1) begin
            digit   = 2'd1;
            rem_ext = mini - s1;
        end else begin
            digit   = 2'd0;
            rem_ext = mini;
        end
        rem_next = rem_ext[WQ+2:0];
        q_next   = {q, digit};
        m_wide   = 3 * {2'b00, q_next} + 1;
        m_next   = m_wide;
    end
endmodule

`default_nettype wire
