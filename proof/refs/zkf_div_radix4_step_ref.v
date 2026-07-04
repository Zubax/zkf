/// Combinational reference for the radix-4 restoring-divide step.
/// Spec: pick the largest digit in {0,1,2,3} such that digit*den <= 4*rem; output rem_next = 4*rem - digit*den.
/// This reference is structurally different from the DUT (priority-encode via explicit comparisons,
/// without parallel candidate-subtract via inverted-borrow flags).

`default_nettype none

module zkf_div_radix4_step_ref #(parameter WMAN = 18) (
    input  wire [WMAN-1:0] den,
    input  wire [WMAN-1:0] rem,
    output reg  [WMAN-1:0] rem_next,
    output reg       [1:0] digit
);
    localparam WREM4 = WMAN + 2;

    wire [WREM4-1:0] rem4    = {rem, 2'b00};
    wire [WREM4-1:0] den1    = {2'b00, den};
    wire [WREM4-1:0] den2    = {1'b0, den, 1'b0};
    wire [WREM4-1:0] den3    = den2 + den1;

    reg [WREM4-1:0] rem_next_ext;

    always @(*) begin
        if (rem4 >= den3) begin
            digit        = 2'd3;
            rem_next_ext = rem4 - den3;
        end else if (rem4 >= den2) begin
            digit        = 2'd2;
            rem_next_ext = rem4 - den2;
        end else if (rem4 >= den1) begin
            digit        = 2'd1;
            rem_next_ext = rem4 - den1;
        end else begin
            digit        = 2'd0;
            rem_next_ext = rem4;
        end
        rem_next = rem_next_ext[WMAN-1:0];
    end
endmodule

`default_nettype wire
