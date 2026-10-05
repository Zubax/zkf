// One radix-4 digit of zkf_divsqrt's recurrence, whose header defines the state; also zkf_cordic's divider, for which
// any divisor works as long as w < d. The root's subtrahends get bit IQ (u/4) or IQ+2 (u) set, which is zero there; d
// takes the digit at IQ+1 and d3 adds (3*digit - 3) << (IQ+1). MODE as zkf_divsqrt's; at 2 the operation is chosen by
// sqrt, and inj gives the subtrahend bits at the injection points (F3 at IQ, F2 at IQ+2, F1 at IQ) so that the caller
// can register them; the other modes ignore it.

`default_nettype none

module _zkf_divsqrt_step #(
    parameter WW   = 20,   // width of w and d; d3 is two bits wider
    parameter IQ   = 0,
    parameter MODE = 2
) (
    input  wire          sqrt,
    input  wire [WW-1:0] w,
    input  wire [WW-1:0] d,
    input  wire [WW+1:0] d3,
    input  wire    [2:0] inj,
    output wire    [1:0] digit,
    output wire [WW-1:0] w_next,
    output wire [WW-1:0] d_next,
    output wire [WW+1:0] d3_next
);
    localparam WT = WW + 2;
    localparam LO = (MODE == 1) ? IQ : 0;   // a root-only build's subtrahends are zero below IQ
    localparam WH = WT - IQ - 1;            // the bits of d3 from u/2 up, the only ones the root changes
    wire [WT-1:0] m    = {w, 2'b00};
    wire [WT-1:0] f1, f2, f3;
    generate
        if (MODE == 0) begin : g_plain
            assign f1 = {2'b00, d};
            assign f2 = {1'b0, d, 1'b0};
            assign f3 = d3;
        end else begin : g_inject
            wire    [2:0] bits = (MODE == 1) ? 3'b111 : inj;
            wire [WT-1:0] at   = {{(WT-1){1'b0}}, 1'b1} << IQ;
            assign f1 = ({2'b00, d} & ~at) | (bits[0] ? at : {WT{1'b0}});
            assign f2 = ({1'b0, d, 1'b0} & ~(at << 2)) | (bits[1] ? (at << 2) : {WT{1'b0}});
            assign f3 = (d3 & ~at) | (bits[2] ? at : {WT{1'b0}});
        end
    endgenerate

    wire [WT-LO:0] t1 = {1'b0, m[WT-1:LO]} - {1'b0, f1[WT-1:LO]};
    wire [WT-LO:0] t2 = {1'b0, m[WT-1:LO]} - {1'b0, f2[WT-1:LO]};
    wire [WT-LO:0] t3 = {1'b0, m[WT-1:LO]} - {1'b0, f3[WT-1:LO]};
    wire ge1 = !t1[WT-LO];
    wire ge2 = !t2[WT-LO];
    wire ge3 = !t3[WT-LO];
    assign digit = {ge2, ge3 || (ge1 && !ge2)};

    wire [WT-1:0] r1, r2, r3;
    wire [WH-1:0] hi = d3[WT-1:IQ+1];
    generate
        if (LO > 0) begin : g_tail
            assign r1 = {t1[WT-LO-1:0], m[LO-1:0]};
            assign r2 = {t2[WT-LO-1:0], m[LO-1:0]};
            assign r3 = {t3[WT-LO-1:0], m[LO-1:0]};
        end else begin : g_full
            assign r1 = t1[WT-1:0];
            assign r2 = t2[WT-1:0];
            assign r3 = t3[WT-1:0];
        end
        if (MODE == 1) begin : g_by_digit
            // Selected by the digit, which lets Vivado fold the d3 candidates into one adder; LSE's fold puts that
            // adder behind the select, which ZKF_ATTRIBUTE_KEEP prevents.
            reg [WW-1:0] w_sel;
            reg [WH-1:0] hi_sel;
`ifdef ZKF_ATTRIBUTE_KEEP
            `ZKF_ATTRIBUTE_KEEP
`endif
            wire [WH-1:0] hi_m3 = hi - {{(WH-2){1'b0}}, 2'd3}, hi_p3 = hi + {{(WH-2){1'b0}}, 2'd3},
                          hi_p6 = hi + {{(WH-3){1'b0}}, 3'd6};
            always @* begin
                case (digit)
                    2'd0:    begin w_sel = m[WW-1:0];  hi_sel = hi_m3; end
                    2'd1:    begin w_sel = r1[WW-1:0]; hi_sel = hi;    end
                    2'd2:    begin w_sel = r2[WW-1:0]; hi_sel = hi_p3; end
                    default: begin w_sel = r3[WW-1:0]; hi_sel = hi_p6; end
                endcase
            end
            assign w_next  = w_sel;
            assign d3_next = {hi_sel, d3[IQ:0]};
            assign d_next  = d | ({{(WW-2){1'b0}}, digit} << (IQ + 1));
        end else if (MODE == 0) begin : g_by_priority
            // A priority select by the compares measures fastest for the divider alone.
            assign w_next  = ge3 ? r3[WW-1:0] : ge2 ? r2[WW-1:0] : ge1 ? r1[WW-1:0] : m[WW-1:0];
            assign d3_next = d3;
            assign d_next  = d;
        end else begin : g_by_compares
            // Selected by the compares, which are monotonic (ge3 implies ge2 implies ge1): a level shallower than by
            // the digit, where the divider and the root share the path.
            wire [WW-1:0] w_lo = ge1 ? r1[WW-1:0] : m[WW-1:0];
            wire [WW-1:0] w_hi = ge3 ? r3[WW-1:0] : r2[WW-1:0];
            assign w_next = ge2 ? w_hi : w_lo;
            // The root's candidates fall back to hi for the divider, so the select stays 4:1.
            wire [WH-1:0] hi_m3 = sqrt ? hi - {{(WH-2){1'b0}}, 2'd3} : hi;
            wire [WH-1:0] hi_p3 = sqrt ? hi + {{(WH-2){1'b0}}, 2'd3} : hi;
            wire [WH-1:0] hi_p6 = sqrt ? hi + {{(WH-3){1'b0}}, 3'd6} : hi;
            wire [WH-1:0] hi_lo = ge1 ? hi : hi_m3;
            wire [WH-1:0] hi_hi = ge3 ? hi_p6 : hi_p3;
            assign d3_next = {ge2 ? hi_hi : hi_lo, d3[IQ:0]};
            assign d_next  = d | ({{(WW-2){1'b0}}, digit & {2{sqrt}}} << (IQ + 1));
        end
    endgenerate
endmodule

`default_nettype wire
