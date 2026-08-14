/// Combinational reference square root for formal equivalence proofs.
/// Implementation: special-case classification, then a radix-2 restoring integer square root of
/// y = sig << (WFRAC + r) (widened BEFORE the shift), then rounding at target precision by direct
/// midpoint-square comparison: round up iff 4*y > (2*lo+1)^2, ties-to-even applied directly.
/// Structurally different from the radix-4 chain in zkf_sqrt.v, including at the rounding boundary
/// (no QFRAC scaling, no remainder-vs-root guard rule).
/// The unrounded root lo and radicand y are exposed so the harness can assert lo^2 <= y < (lo+1)^2.

`default_nettype none

module zkf_sqrt_ref #(
    parameter WEXP = 6,
    parameter WMAN = 18
) (
    input  wire [WEXP+WMAN-1:0] x,
    output wire [WEXP+WMAN-1:0] y,
    output wire                 domain_error,
    output wire [2*WMAN-1:0]    dbg_y,
    output wire [WMAN-1:0]      dbg_lo,
    output wire signed [WEXP+1:0] dbg_exp
);
    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;
    localparam WE    = WEXP + 2;                       // signed exponent working width

    localparam signed [WE-1:0] BIAS = (1 << (WEXP - 1)) - 1;

    // Decode.
    wire             x_sign = x[WFULL-1];
    wire [WEXP-1:0]  x_exp  = x[WFULL-2:WFRAC];
    wire [WFRAC-1:0] x_frac = x[WFRAC-1:0];

    wire             x_zero   = ~|x_exp;
    wire             x_inf    =  &x_exp;
    wire             negative = x_sign && !x_zero;
    wire [WMAN-1:0]  sig      = {1'b1, x_frac};

    assign domain_error = negative;

    // Radicand y = sig << (WFRAC + r): sqrt(y) = sqrt(m * 2^r) * 2^WFRAC. Widen before the shift.
    wire signed [WE-1:0] e     = $signed({{(WE-WEXP){1'b0}}, x_exp}) - BIAS;
    wire                 r_par = e[0];                 // e mod 2 (two's-complement LSB is the parity)
    wire [2*WMAN-1:0]    sig_w = {{WMAN{1'b0}}, sig};
    wire [2*WMAN-1:0]    y_rad = r_par ? (sig_w << (WFRAC + 1)) : (sig_w << WFRAC);

    // Radix-2 restoring integer square root: WMAN result bits from the 2*WMAN-bit radicand.
    // Invariant after each bit: part_rem = (consumed radicand bits) - lo^2, 0 <= part_rem <= 2*lo.
    reg [WMAN-1:0] lo;
    reg [WMAN+3:0] part_rem;
    reg [WMAN+3:0] trial;
    reg [WMAN+3:0] sub;
    integer        i;
    always @(*) begin
        lo       = {WMAN{1'b0}};
        part_rem = {(WMAN+4){1'b0}};
        trial    = {(WMAN+4){1'b0}};
        sub      = {(WMAN+4){1'b0}};
        for (i = WMAN - 1; i >= 0; i = i - 1) begin
            trial = {part_rem[WMAN+1:0], y_rad[2*i +: 2]};      // 4*part_rem + next radicand bit pair
            sub   = {2'b00, lo, 2'b01};                         // 4*lo + 1
            if (trial >= sub) begin
                part_rem = trial - sub;
                lo       = {lo[WMAN-2:0], 1'b1};
            end else begin
                part_rem = trial;
                lo       = {lo[WMAN-2:0], 1'b0};
            end
        end
    end

    assign dbg_y  = y_rad;
    assign dbg_lo = lo;
    // dbg_exp is assigned below; the harness range-asserts it so the finite pack's truncation is proved safe.

    // Round to nearest by midpoint-square comparison: sqrt(y) > lo + 1/2 iff 4*y > (2*lo+1)^2.
    // Ties-to-even applied directly (a tie is structurally impossible; kept for independence).
    wire [2*WMAN+1:0] y4     = {y_rad, 2'b00};
    wire [WMAN:0]     mid    = {lo, 1'b1};             // 2*lo + 1
    wire [2*WMAN+1:0] mid_sq = mid * mid;
    wire              up     = (y4 > mid_sq) || ((y4 == mid_sq) && lo[0]);
    wire [WMAN:0]     rounded_ext = {1'b0, lo} + {{WMAN{1'b0}}, up};

    // Result exponent floor(e/2) = (e - r) / 2, re-biased; +1 on the (unreachable) rounding carry.
    // r_ext/carry_ext are declared signed so the >>> stays an arithmetic shift (a bare concatenation
    // operand would turn the whole expression unsigned).
    wire signed [WE-1:0] r_ext     = {{(WE-1){1'b0}}, r_par};
    wire signed [WE-1:0] carry_ext = {{(WE-1){1'b0}}, rounded_ext[WMAN]};
    wire signed [WE-1:0] e_half    = (e - r_ext) >>> 1;
    wire                 carry     = rounded_ext[WMAN];
    wire signed [WE-1:0] exp_out   = e_half + BIAS + carry_ext;
    assign dbg_exp = exp_out;
    wire [WMAN-1:0]      root_sig = carry ? rounded_ext[WMAN:1] : rounded_ext[WMAN-1:0];

    // Pack: specials first (zero beats sign, negative beats infinity), then the always-normal finite result.
    wire [WFULL-1:0] pos_inf = {1'b0, {WEXP{1'b1}}, {WFRAC{1'b0}}};
    wire [WFULL-1:0] neg_inf = {1'b1, {WEXP{1'b1}}, {WFRAC{1'b0}}};
    wire [WFULL-1:0] finite  = {1'b0, exp_out[WEXP-1:0], root_sig[WFRAC-1:0]};

    assign y = x_zero   ? {WFULL{1'b0}} :
               negative ? neg_inf :
               x_inf    ? pos_inf :
                          finite;
endmodule

`default_nettype wire
