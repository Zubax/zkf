// Streamed round to integer, delivered on the same cycle both as a float of the same format (y_float) and as a
// saturating signed two's-complement integer (y_int). An unused result may be left unconnected; its logic is pruned.
// round_mode: 0 = nearest, ties to even; 1 = floor; 2 = ceil; 3 = truncation.
//
// y_float: +-inf passes through as canonical inf; zero and any zero-magnitude result are canonical +0; a rounded value
// beyond the format is signed inf. y_int: +-inf and out-of-range values saturate.
//
// Everything the rounding decision does not influence is computed beside it, so the decision only drives the final
// select between the truncated and the incremented candidate of each result. The float increment is taken on the packed
// {exponent, fraction} with the discarded bits set, so its carry ripples into the exponent and lands on canonical inf
// by itself. The integer candidates are the truncated magnitude XOR sign and that plus one, which covers both the
// rounding increment and the negation with one incrementer.
//
// STAGE_INPUT: registers ahead of the decode.
// STAGE_SHIFT: register the fraction mask and the truncated integer magnitude ahead of the rounding decision;
//     the first stage to enable.
// STAGE_ROUND: register the rounding decision ahead of the increments; the second stage to enable.
// STAGE_OUTPUT: register the results.
// Each knob is a register count costing as many cycles; values above one add dummy stages. With all of them zero the
// module is combinational and clk/rst are ignored.

`default_nettype none

module zkf_rint #(
    parameter WEXP         = 6,
    parameter WMAN         = 18,
    parameter WINT         = 32,
    parameter STAGE_INPUT  = 0,
    parameter STAGE_SHIFT  = 0,
    parameter STAGE_ROUND  = 0,
    parameter STAGE_OUTPUT = 0,
    parameter LATENCY      = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] a,           // operand to round
    input wire           [1:0] round_mode,  // 0 = nearest, ties to even; 1 = floor; 2 = ceil; 3 = truncation

    output wire                   out_valid,
    output wire   [WEXP+WMAN-1:0] y_float,
    output wire signed [WINT-1:0] y_int
);
    localparam LATENCY_REF = STAGE_INPUT + STAGE_SHIFT + STAGE_ROUND + STAGE_OUTPUT;
    generate
        if ((WEXP < 2) || (WMAN < 4) || (WINT < 2)) begin : g_invalid
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        // Shift by WEXP >= 32 would overflow Verilog's integer constant arithmetic and yield tool-dependent values.
        if (WEXP >= 32) begin : g_invalid_wexp_too_wide
            _zkf_invalid_rint_wexp_too_wide_unportable u_invalid();
        end
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    localparam ROUND_NEAREST_EVEN = 2'd0;
    localparam ROUND_FLOOR        = 2'd1;
    localparam ROUND_CEIL         = 2'd2;
    localparam ROUND_TRUNC        = 2'd3;

    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;
    localparam WT    = WINT - 1;                        // magnitude bits of the integer result
    localparam WS    = (WT > 1) ? $clog2(WT) : 1;

    // KK is the exponent at which the significand LSB has unit weight; TOP is the largest exponent whose magnitude
    // still fits in WT bits.
    localparam integer BIAS = (1 << (WEXP - 1)) - 1;
    localparam integer KK   = WFRAC + BIAS;
    localparam integer TOP  = BIAS + WINT - 2;
    localparam         WPP  = $clog2(WFRAC + 1);
    localparam         WA   = $clog2(((KK > TOP) ? KK : TOP) + 1) + 1;  // signed; KK alone exceeds any exponent

    localparam        [WEXP-1:0] BIAS_VEC    = BIAS;
    localparam        [WEXP-1:0] BIAS_M1_VEC = BIAS - 1;
    localparam         [WPP-1:0] WFRAC_VEC   = WFRAC;
    localparam signed   [WA-1:0] KK_S        = KK;
    localparam signed   [WA-1:0] TOP_S       = TOP;

    wire             in_q_valid;
    wire [WFULL+1:0] in_q;
    zkf_pipe #(.W(WFULL + 2), .N(STAGE_INPUT)) u_input_pipe (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in({round_mode, a}),
        .out_valid(in_q_valid), .out(in_q)
    );

    // Decode: classification, the mask of the discarded fraction bits, and the truncated integer magnitude.
    wire             sign_in = in_q[WFULL-1];
    wire  [WEXP-1:0] exp_in  = in_q[WFULL-2:WFRAC];
    wire [WFRAC-1:0] frac_in = in_q[WFRAC-1:0];
    wire             is_inf  = &exp_in;
    wire             sub_one = exp_in < BIAS_VEC;       // |a| < 1, zero included: the result is 0 or +-1
    wire             special = is_inf | ~|exp_in;       // zero or inf: nothing to round

    wire signed [WA-1:0] exp_s = $signed({{(WA-WEXP){1'b0}}, exp_in});
    wire signed [WA-1:0] p_s   = KK_S - exp_s;          // number of fraction bits
    wire signed [WA-1:0] s_s   = TOP_S - exp_s;         // right shift that aligns the magnitude in WT bits
    wire                 oor   = is_inf | s_s[WA-1];    // inf or magnitude beyond WT bits
    reg        [WPP-1:0] pp;
    always @* begin
        case ({sub_one | is_inf, p_s <= 0})
            2'b00:   pp = p_s[WPP-1:0];
            2'b01:   pp = {WPP{1'b0}};
            default: pp = WFRAC_VEC;                    // inf discards its payload
        endcase
    end

    wire [WT-1:0] mag_aligned;
    generate
        if (WT == 1) begin : g_mag_unit
            assign mag_aligned = 1'b1;
        end else if (WMAN >= WT) begin : g_mag_narrow
            assign mag_aligned = {1'b1, frac_in[WFRAC-1 -: WT-1]};
        end else begin : g_mag_wide
            assign mag_aligned = {1'b1, frac_in, {(WT-WMAN){1'b0}}};
        end
    endgenerate
    // near: round to nearest; away: round away from zero whenever anything is discarded.
    reg near, away;
    always @* begin
        case (in_q[WFULL+1:WFULL])
            ROUND_NEAREST_EVEN: {near, away} = 2'b10;
            ROUND_FLOOR:        {near, away} = {1'b0, sign_in};
            ROUND_CEIL:         {near, away} = {1'b0, ~sign_in};
            ROUND_TRUNC:        {near, away} = 2'b00;
        endcase
    end

    localparam WSH = 6 + WEXP + (2 * WFRAC) + WT;
    wire           sh_valid;
    wire [WSH-1:0] sh;
    zkf_pipe #(.W(WSH), .N(STAGE_SHIFT)) u_shift_pipe (
        .clk(clk), .rst(rst),
        .in_valid(in_q_valid),
        // In order: a sub-one input rounds to +-1 (only if |a| > 0.5 when rounding to nearest); the two rounding rules
        // of the other inputs; a positive magnitude of all ones has no in-range successor; a sub-one float exponent is
        // replaced by BIAS-1 so that its incremented candidate is +-1; an out-of-range magnitude is forced to all ones,
        // which the XOR with the sign turns into the rail.
        .in({~special & sub_one & (away | (near & (exp_in == BIAS_M1_VEC) & (|frac_in))),
             ~special & ~sub_one & near,
             ~special & ~sub_one & away,
             ~oor & ~(~sign_in & ~|s_s & (&mag_aligned)),
             sign_in,
             sub_one,
             sub_one ? BIAS_M1_VEC : exp_in,
             frac_in, ~({WFRAC{1'b1}} << pp),
             ((mag_aligned >> s_s[WS-1:0]) & {WT{~sub_one}}) | {WT{oor}}}),
        .out_valid(sh_valid), .out(sh)
    );
    wire    [WT-1:0] trunc_s   = sh[0 +: WT];
    wire [WFRAC-1:0] fm        = sh[WT +: WFRAC];
    wire [WFRAC-1:0] frac      = sh[WT+WFRAC +: WFRAC];
    wire  [WEXP-1:0] exp_f     = sh[WT+(2*WFRAC) +: WEXP];
    wire             sub_one_s = sh[WSH-6];
    wire             sign_s    = sh[WSH-5];
    wire             has_next  = sh[WSH-4];
    wire             away_s    = sh[WSH-3];
    wire             near_s    = sh[WSH-2];
    wire             inc_sub   = sh[WSH-1];

    // Rounding decision.
    wire [WMAN-1:0] bit_pp   = {fm, 1'b1} & ~{1'b0, fm};                      // one-hot at the integer LSB
    wire            discards = |(frac & fm);
    wire            guard    = |(frac & fm & ~(fm >> 1));
    wire            odd_or_sticky = |({1'b1, frac} & (bit_pp | {2'b00, fm[WFRAC-1:1]}));
    wire            inc      = inc_sub | (away_s & discards) | (near_s & guard & odd_or_sticky);

    localparam WRND = 4 + WEXP + (2 * WFRAC) + WT;
    wire            rnd_valid;
    wire [WRND-1:0] rnd;
    zkf_pipe #(.W(WRND), .N(STAGE_ROUND)) u_round_pipe (
        .clk(clk), .rst(rst),
        .in_valid(sh_valid),
        .in({inc,
             (inc ^ sign_s) & has_next,
             sign_s,
             sub_one_s,
             exp_f,
             frac & ~fm,
             fm,
             trunc_s}),
        .out_valid(rnd_valid), .out(rnd)
    );
    wire    [WT-1:0] trunc_r   = rnd[0 +: WT];
    wire [WFRAC-1:0] fm_r      = rnd[WT +: WFRAC];
    wire [WFRAC-1:0] frac_r    = rnd[WT+WFRAC +: WFRAC];             // discarded bits cleared
    wire  [WEXP-1:0] exp_r     = rnd[WT+(2*WFRAC) +: WEXP];
    wire             sub_one_r = rnd[WRND-4];
    wire             sign_r    = rnd[WRND-3];
    wire             next_r    = rnd[WRND-2];
    wire             inc_r     = rnd[WRND-1];

    // Candidates and the final select.
    wire [WFULL-2:0] float_next = {exp_r, frac_r | fm_r} + {{(WFULL-2){1'b0}}, 1'b1};
    wire [WFULL-1:0] float_sel  = inc_r ? {sign_r, float_next} : ({sign_r, exp_r, frac_r} & {WFULL{~sub_one_r}});
    wire  [WINT-1:0] int_base   = {1'b0, trunc_r} ^ {WINT{sign_r}};
    wire  [WINT-1:0] int_sel    = next_r ? (int_base + {{WT{1'b0}}, 1'b1}) : int_base;

    wire [WFULL+WINT-1:0] out_q;
    zkf_pipe #(.W(WFULL + WINT), .N(STAGE_OUTPUT)) u_output_pipe (
        .clk(clk), .rst(rst),
        .in_valid(rnd_valid), .in({float_sel, int_sel}),
        .out_valid(out_valid), .out(out_q)
    );
    assign y_float = out_q[WINT +: WFULL];
    assign y_int   = $signed(out_q[WINT-1:0]);
endmodule

`default_nettype wire
