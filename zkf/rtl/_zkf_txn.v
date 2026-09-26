/// Single-transaction handshake of the iterative operators. `res_valid` is a one-cycle strobe with `res`.

`default_nettype none

module _zkf_txn #(
    parameter integer W            = 1,
    parameter integer STAGE_OUTPUT = 0
) (
    input  wire         clk,
    input  wire         rst,
    input  wire         in_valid,
    output wire         in_ready,
    output wire         accept,
    input  wire         res_valid,
    input  wire [W-1:0] res,
    output wire         out_valid,
    input  wire         out_ready,
    output wire [W-1:0] out
);
    reg busy;
    assign in_ready = ~busy;
    assign accept   = in_valid & in_ready;

    generate
        if ((STAGE_OUTPUT != 0) && (STAGE_OUTPUT != 1)) begin : g_invalid_stage_output
            _zkf_invalid_stage_output u_invalid();
        end
        if (STAGE_OUTPUT == 0) begin : g_out_comb
            reg         pending;
            reg [W-1:0] hold;
            always @(posedge clk) begin
                if (rst)                         pending <= 1'b0;
                else if (res_valid & ~out_ready) pending <= 1'b1;
                else if (pending & out_ready)    pending <= 1'b0;
                if (res_valid & ~out_ready) hold <= res;
            end
            assign out_valid = res_valid | pending;
            assign out       = pending ? hold : res;
        end else begin : g_out_reg
            reg         r_valid;
            reg [W-1:0] r_out;
            always @(posedge clk) begin
                if (rst)            r_valid <= 1'b0;
                else if (res_valid) r_valid <= 1'b1;
                else if (out_ready) r_valid <= 1'b0;
                if (res_valid) r_out <= res;
            end
            assign out_valid = r_valid;
            assign out       = r_out;
        end
    endgenerate

    always @(posedge clk) begin
        if (rst)                        busy <= 1'b0;
        else if (accept)                busy <= 1'b1;
        else if (out_valid & out_ready) busy <= 1'b0;
    end
endmodule

`default_nettype wire
