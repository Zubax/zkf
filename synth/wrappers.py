"""
Verilog measurement-harness generators for the float synthesis suite.

Each write_*_wrapper emits a synthesis top that registers every DUT input and output, so the reported
f max is a register-to-register limit rather than ignoring primary I/O paths. The harness is identical
regardless of the target device/tool, so this module is device-independent. Not runnable on its own.
"""

from __future__ import annotations

from pathlib import Path

from common import SYNTH_REG_ATTR
from modules import MUL_ILOG2_CONST_K, ModuleSpec, effective_parallel, effective_stage_product_final, register_stages


def write_pack_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    wexp_unbiased = spec.wexp_unbiased or (spec.wexp + 2)
    path.write_text(f"""`default_nettype none

module {spec.top} (
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     in_valid,
    input  wire                     sign,
    input  wire                     force_zero,
    input  wire                     force_inf,
    input  wire signed [{wexp_unbiased - 1}:0] exp_unbiased,
    input  wire [{spec.wman - 1}:0] significand,
    input  wire                     guard,
    input  wire                     round,
    input  wire                     sticky,
    output wire                     out_valid,
    output wire [{wfull - 1}:0]     y
);
    // Measurement harness: put real registers on every DUT input and output so the timing report includes
    // paths that would otherwise be reported as unconstrained primary-input/primary-output delays.
    {SYNTH_REG_ATTR}
    reg                            r_in_valid;
    {SYNTH_REG_ATTR}
    reg                            r_sign;
    {SYNTH_REG_ATTR}
    reg                            r_force_zero;
    {SYNTH_REG_ATTR}
    reg                            r_force_inf;
    {SYNTH_REG_ATTR}
    reg signed [{wexp_unbiased - 1}:0] r_exp_unbiased;
    {SYNTH_REG_ATTR}
    reg                 [{spec.wman - 1}:0] r_significand;
    {SYNTH_REG_ATTR}
    reg                            r_guard;
    {SYNTH_REG_ATTR}
    reg                            r_round;
    {SYNTH_REG_ATTR}
    reg                            r_sticky;

    wire                 dut_out_valid;
    wire [{wfull - 1}:0] dut_y;

    {SYNTH_REG_ATTR}
    reg                 r_out_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_y;

    assign out_valid = r_out_valid;
    assign y         = r_y;

    _zkf_pack #(
        .WEXP({spec.wexp}),
        .WMAN({spec.wman}),
        .WEXP_UNBIASED({wexp_unbiased})
    ) dut (
        .clk(clk),
        .rst(rst),
        .in_valid(r_in_valid),
        .sign(r_sign),
        .force_zero(r_force_zero),
        .force_inf(r_force_inf),
        .exp_unbiased(r_exp_unbiased),
        .significand(r_significand),
        .guard(r_guard),
        .round(r_round),
        .sticky(r_sticky),
        .out_valid(dut_out_valid),
        .y(dut_y)
    );

    always @(posedge clk) begin
        if (rst) begin
            r_in_valid  <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid  <= in_valid;
            r_out_valid <= dut_out_valid;
        end

        r_sign          <= sign;
        r_force_zero    <= force_zero;
        r_force_inf     <= force_inf;
        r_exp_unbiased  <= exp_unbiased;
        r_significand   <= significand;
        r_guard         <= guard;
        r_round         <= round;
        r_sticky        <= sticky;
        r_y             <= dut_y;
    end
endmodule

`default_nettype wire
""")


def write_mul_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    path.write_text(f"""`default_nettype none

module {spec.top} (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 in_valid,
    input  wire [{wfull - 1}:0] a,
    input  wire [{wfull - 1}:0] b,
    output wire                 out_valid,
    output wire [{wfull - 1}:0] y
);
    // Measurement harness: put real registers on every DUT input and output so the timing report includes
    // paths that would otherwise be reported as unconstrained primary-input/primary-output delays.
    {SYNTH_REG_ATTR}
    reg                 r_in_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_a;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_b;

    wire                 dut_out_valid;
    wire [{wfull - 1}:0] dut_y;

    {SYNTH_REG_ATTR}
    reg                 r_out_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_y;

    assign out_valid = r_out_valid;
    assign y         = r_y;

    zkf_mul #(
        .WEXP({spec.wexp}),
        .WMAN({spec.wman}),
        .STAGE_INPUT({spec.stage_input}),
        .STAGE_PRODUCT({spec.stage_product}),
        .WMULTIPLIER({spec.wmultiplier}),
        .STAGE_PACK({spec.stage_pack}),
        .STAGE_OUTPUT({spec.stage_output}),
        .LATENCY({register_stages(spec)})
    ) dut (
        .clk(clk),
        .rst(rst),
        .in_valid(r_in_valid),
        .a(r_a),
        .b(r_b),
        .out_valid(dut_out_valid),
        .y(dut_y)
    );

    always @(posedge clk) begin
        if (rst) begin
            r_in_valid  <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid  <= in_valid;
            r_out_valid <= dut_out_valid;
        end

        r_a <= a;
        r_b <= b;
        r_y <= dut_y;
    end
endmodule

`default_nettype wire
""")


def write_add_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    path.write_text(f"""`default_nettype none

module {spec.top} (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 in_valid,
    input  wire [{wfull - 1}:0] a,
    input  wire [{wfull - 1}:0] b,
    output wire                 out_valid,
    output wire [{wfull - 1}:0] y
);
    // Measurement harness: put real registers on every DUT input and output so the timing report includes
    // paths that would otherwise be reported as unconstrained primary-input/primary-output delays.
    {SYNTH_REG_ATTR}
    reg                 r_in_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_a;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_b;

    wire                 dut_out_valid;
    wire [{wfull - 1}:0] dut_y;

    {SYNTH_REG_ATTR}
    reg                 r_out_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_y;

    assign out_valid = r_out_valid;
    assign y         = r_y;

    zkf_add #(
        .WEXP({spec.wexp}),
        .WMAN({spec.wman}),
        .STAGE_INPUT({spec.stage_input}),
        .STAGE_DECODE({spec.stage_decode}),
        .STAGE_ALIGN({spec.stage_align}),
        .STAGE_NORMALIZE({spec.stage_normalize}),
        .STAGE_PACK({spec.stage_pack}),
        .STAGE_OUTPUT({spec.stage_output}),
        .LATENCY({register_stages(spec)})
    ) dut (
        .clk(clk),
        .rst(rst),
        .in_valid(r_in_valid),
        .a(r_a),
        .b(r_b),
        .out_valid(dut_out_valid),
        .y(dut_y)
    );

    always @(posedge clk) begin
        if (rst) begin
            r_in_valid  <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid  <= in_valid;
            r_out_valid <= dut_out_valid;
        end

        r_a <= a;
        r_b <= b;
        r_y <= dut_y;
    end
endmodule

`default_nettype wire
""")


def write_addsub_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    path.write_text(f"""`default_nettype none

module {spec.top} (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 in_valid,
    input  wire [{wfull - 1}:0] a,
    input  wire [{wfull - 1}:0] b,
    input  wire                 op_sub,
    output wire                 out_valid,
    output wire [{wfull - 1}:0] y
);
    // Measurement harness: put real registers on every DUT input and output so the timing report includes
    // paths that would otherwise be reported as unconstrained primary-input/primary-output delays.
    {SYNTH_REG_ATTR}
    reg                 r_in_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_a;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_b;
    {SYNTH_REG_ATTR}
    reg                 r_op_sub;

    wire                 dut_out_valid;
    wire [{wfull - 1}:0] dut_y;

    {SYNTH_REG_ATTR}
    reg                 r_out_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_y;

    assign out_valid = r_out_valid;
    assign y         = r_y;

    zkf_addsub #(
        .WEXP({spec.wexp}),
        .WMAN({spec.wman}),
        .STAGE_INPUT({spec.stage_input}),
        .STAGE_DECODE({spec.stage_decode}),
        .STAGE_ALIGN({spec.stage_align}),
        .STAGE_NORMALIZE({spec.stage_normalize}),
        .STAGE_PACK({spec.stage_pack}),
        .STAGE_OUTPUT({spec.stage_output}),
        .LATENCY({register_stages(spec)})
    ) dut (
        .clk(clk),
        .rst(rst),
        .in_valid(r_in_valid),
        .a(r_a),
        .b(r_b),
        .op_sub(r_op_sub),
        .out_valid(dut_out_valid),
        .y(dut_y)
    );

    always @(posedge clk) begin
        if (rst) begin
            r_in_valid  <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid  <= in_valid;
            r_out_valid <= dut_out_valid;
        end

        r_a      <= a;
        r_b      <= b;
        r_op_sub <= op_sub;
        r_y      <= dut_y;
    end
endmodule

`default_nettype wire
""")


def write_fma_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    path.write_text(f"""`default_nettype none

module {spec.top} (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 in_valid,
    input  wire [{wfull - 1}:0] a,
    input  wire [{wfull - 1}:0] b,
    input  wire [{wfull - 1}:0] c,
    output wire                 out_valid,
    output wire [{wfull - 1}:0] y
);
    // Measurement harness: put real registers on every DUT input and output so the timing report includes
    // paths that would otherwise be reported as unconstrained primary-input/primary-output delays.
    {SYNTH_REG_ATTR}
    reg                 r_in_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_a;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_b;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_c;

    wire                 dut_out_valid;
    wire [{wfull - 1}:0] dut_y;

    {SYNTH_REG_ATTR}
    reg                 r_out_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_y;

    assign out_valid = r_out_valid;
    assign y         = r_y;

    zkf_fma #(
        .WEXP({spec.wexp}),
        .WMAN({spec.wman}),
        .STAGE_INPUT({spec.stage_input}),
        .STAGE_PRODUCT({spec.stage_product}),
        .WMULTIPLIER({spec.wmultiplier}),
        .STAGE_DECODE({spec.stage_decode}),
        .STAGE_ALIGN({spec.stage_align}),
        .STAGE_NORMALIZE({spec.stage_normalize}),
        .STAGE_PACK({spec.stage_pack}),
        .STAGE_OUTPUT({spec.stage_output}),
        .LATENCY({register_stages(spec)})
    ) dut (
        .clk(clk),
        .rst(rst),
        .in_valid(r_in_valid),
        .a(r_a),
        .b(r_b),
        .c(r_c),
        .out_valid(dut_out_valid),
        .y(dut_y)
    );

    always @(posedge clk) begin
        if (rst) begin
            r_in_valid  <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid  <= in_valid;
            r_out_valid <= dut_out_valid;
        end

        r_a <= a;
        r_b <= b;
        r_c <= c;
        r_y <= dut_y;
    end
endmodule

`default_nettype wire
""")


def write_div_core_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    wexp_unbiased = spec.wexp + 2
    path.write_text(f"""`default_nettype none

module {spec.top} (
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     in_valid,
    input  wire [{wfull - 1}:0]     a,
    input  wire [{wfull - 1}:0]     b,
    output wire                     out_valid,
    output wire                     sign,
    output wire                     force_zero,
    output wire                     force_inf,
    output wire signed [{wexp_unbiased - 1}:0] exp_biased,
    output wire [{spec.wman - 1}:0] significand,
    output wire                     guard,
    output wire                     round,
    output wire                     sticky,
    output wire                     div0,
    output wire [{spec.wman - 1}:0] partial_rem
);
    // Measurement harness: put real registers on every DUT input and output so the timing report includes
    // paths that would otherwise be reported as unconstrained primary-input/primary-output delays.
    {SYNTH_REG_ATTR}
    reg                 r_in_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_a;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_b;

    wire                            dut_out_valid;
    wire                            dut_sign;
    wire                            dut_force_zero;
    wire                            dut_force_inf;
    wire signed [{wexp_unbiased - 1}:0] dut_exp_biased;
    wire                 [{spec.wman - 1}:0] dut_significand;
    wire                            dut_guard;
    wire                            dut_round;
    wire                            dut_sticky;
    wire                            dut_div0;
    wire                 [{spec.wman - 1}:0] dut_partial_rem;

    {SYNTH_REG_ATTR}
    reg                            r_out_valid;
    {SYNTH_REG_ATTR}
    reg                            r_sign;
    {SYNTH_REG_ATTR}
    reg                            r_force_zero;
    {SYNTH_REG_ATTR}
    reg                            r_force_inf;
    {SYNTH_REG_ATTR}
    reg signed [{wexp_unbiased - 1}:0] r_exp_biased;
    {SYNTH_REG_ATTR}
    reg                 [{spec.wman - 1}:0] r_significand;
    {SYNTH_REG_ATTR}
    reg                            r_guard;
    {SYNTH_REG_ATTR}
    reg                            r_round;
    {SYNTH_REG_ATTR}
    reg                            r_sticky;
    {SYNTH_REG_ATTR}
    reg                            r_div0;
    {SYNTH_REG_ATTR}
    reg                 [{spec.wman - 1}:0] r_partial_rem;

    assign out_valid    = r_out_valid;
    assign sign         = r_sign;
    assign force_zero   = r_force_zero;
    assign force_inf    = r_force_inf;
    assign exp_biased   = r_exp_biased;
    assign significand  = r_significand;
    assign guard        = r_guard;
    assign round        = r_round;
    assign sticky       = r_sticky;
    assign div0         = r_div0;
    assign partial_rem  = r_partial_rem;

    _zkf_div_core #(
        .WEXP({spec.wexp}),
        .WMAN({spec.wman})
    ) dut (
        .clk(clk),
        .rst(rst),
        .in_valid(r_in_valid),
        .a(r_a),
        .b(r_b),
        .out_valid(dut_out_valid),
        .sign(dut_sign),
        .force_zero(dut_force_zero),
        .force_inf(dut_force_inf),
        .exp_biased(dut_exp_biased),
        .significand(dut_significand),
        .guard(dut_guard),
        .round(dut_round),
        .sticky(dut_sticky),
        .div0(dut_div0),
        .partial_rem(dut_partial_rem)
    );

    always @(posedge clk) begin
        if (rst) begin
            r_in_valid  <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid  <= in_valid;
            r_out_valid <= dut_out_valid;
        end

        r_a             <= a;
        r_b             <= b;
        r_sign          <= dut_sign;
        r_force_zero    <= dut_force_zero;
        r_force_inf     <= dut_force_inf;
        r_exp_biased    <= dut_exp_biased;
        r_significand   <= dut_significand;
        r_guard         <= dut_guard;
        r_round         <= dut_round;
        r_sticky        <= dut_sticky;
        r_div0          <= dut_div0;
        r_partial_rem   <= dut_partial_rem;
    end
endmodule

`default_nettype wire
""")


def write_div_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    path.write_text(f"""`default_nettype none

module {spec.top} (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 in_valid,
    input  wire [{wfull - 1}:0] a,
    input  wire [{wfull - 1}:0] b,
    output wire                 out_valid,
    output wire [{wfull - 1}:0] q,
    output wire                 div0
);
    // Measurement harness: put real registers on every DUT input and output so the timing report includes
    // paths that would otherwise be reported as unconstrained primary-input/primary-output delays.
    {SYNTH_REG_ATTR}
    reg                 r_in_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_a;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_b;

    wire                 dut_out_valid;
    wire [{wfull - 1}:0] dut_q;
    wire                 dut_div0;

    {SYNTH_REG_ATTR}
    reg                 r_out_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_q;
    {SYNTH_REG_ATTR}
    reg                 r_div0;

    assign out_valid = r_out_valid;
    assign q         = r_q;
    assign div0      = r_div0;

    zkf_div #(
        .WEXP({spec.wexp}),
        .WMAN({spec.wman}),
        .STAGE_INPUT({spec.stage_input}),
        .STAGE_PACK({spec.stage_pack}),
        .STAGE_OUTPUT({spec.stage_output}),
        .LATENCY({register_stages(spec)})
    ) dut (
        .clk(clk),
        .rst(rst),
        .in_valid(r_in_valid),
        .a(r_a),
        .b(r_b),
        .out_valid(dut_out_valid),
        .q(dut_q),
        .div0(dut_div0)
    );

    always @(posedge clk) begin
        if (rst) begin
            r_in_valid  <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid  <= in_valid;
            r_out_valid <= dut_out_valid;
        end

        r_a    <= a;
        r_b    <= b;
        r_q    <= dut_q;
        r_div0 <= dut_div0;
    end
endmodule

`default_nettype wire
""")


def write_cmp_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    path.write_text(f"""`default_nettype none

module {spec.top} (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 in_valid,
    input  wire [{wfull - 1}:0] a,
    input  wire [{wfull - 1}:0] b,
    output wire                 out_valid,
    output wire                 a_gt_b,
    output wire                 a_eq_b,
    output wire                 a_lt_b
);
    // Measurement harness: put real registers on every DUT input and output so the timing report includes
    // paths that would otherwise be reported as unconstrained primary-input/primary-output delays.
    {SYNTH_REG_ATTR}
    reg                 r_in_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_a;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_b;

    wire dut_out_valid;
    wire dut_a_gt_b;
    wire dut_a_eq_b;
    wire dut_a_lt_b;

    {SYNTH_REG_ATTR}
    reg r_out_valid;
    {SYNTH_REG_ATTR}
    reg r_a_gt_b;
    {SYNTH_REG_ATTR}
    reg r_a_eq_b;
    {SYNTH_REG_ATTR}
    reg r_a_lt_b;

    assign out_valid = r_out_valid;
    assign a_gt_b    = r_a_gt_b;
    assign a_eq_b    = r_a_eq_b;
    assign a_lt_b    = r_a_lt_b;

    zkf_cmp #(
        .WEXP({spec.wexp}),
        .WMAN({spec.wman}),
        .STAGE_INPUT({spec.stage_input}),
        .LATENCY({register_stages(spec)})
    ) dut (
        .clk(clk),
        .rst(rst),
        .in_valid(r_in_valid),
        .a(r_a),
        .b(r_b),
        .out_valid(dut_out_valid),
        .a_gt_b(dut_a_gt_b),
        .a_eq_b(dut_a_eq_b),
        .a_lt_b(dut_a_lt_b)
    );

    always @(posedge clk) begin
        if (rst) begin
            r_in_valid  <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid  <= in_valid;
            r_out_valid <= dut_out_valid;
        end

        r_a      <= a;
        r_b      <= b;
        r_a_gt_b <= dut_a_gt_b;
        r_a_eq_b <= dut_a_eq_b;
        r_a_lt_b <= dut_a_lt_b;
    end
endmodule

`default_nettype wire
""")


def write_sort_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    path.write_text(f"""`default_nettype none

module {spec.top} (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 in_valid,
    input  wire [{wfull - 1}:0] a,
    input  wire [{wfull - 1}:0] b,
    output wire                 out_valid,
    output wire [{wfull - 1}:0] min,
    output wire [{wfull - 1}:0] max
);
    // Measurement harness: put real registers on every DUT input and output so the timing report includes
    // paths that would otherwise be reported as unconstrained primary-input/primary-output delays.
    {SYNTH_REG_ATTR}
    reg                 r_in_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_a;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_b;

    wire                 dut_out_valid;
    wire [{wfull - 1}:0] dut_min;
    wire [{wfull - 1}:0] dut_max;

    {SYNTH_REG_ATTR}
    reg                 r_out_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_min;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_max;

    assign out_valid = r_out_valid;
    assign min       = r_min;
    assign max       = r_max;

    zkf_sort #(
        .WEXP({spec.wexp}),
        .WMAN({spec.wman}),
        .STAGE_INPUT({spec.stage_input}),
        .LATENCY({register_stages(spec)})
    ) dut (
        .clk(clk),
        .rst(rst),
        .in_valid(r_in_valid),
        .a(r_a),
        .b(r_b),
        .out_valid(dut_out_valid),
        .min(dut_min),
        .max(dut_max)
    );

    always @(posedge clk) begin
        if (rst) begin
            r_in_valid  <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid  <= in_valid;
            r_out_valid <= dut_out_valid;
        end

        r_a   <= a;
        r_b   <= b;
        r_min <= dut_min;
        r_max <= dut_max;
    end
endmodule

`default_nettype wire
""")


def write_mul_ilog2_const_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    path.write_text(f"""`default_nettype none

module {spec.top} (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 in_valid,
    input  wire [{wfull - 1}:0] a,
    output wire                 out_valid,
    output wire [{wfull - 1}:0] y
);
    // Measurement harness: put real registers on every DUT input and output so the timing report includes
    // paths that would otherwise be reported as unconstrained primary-input/primary-output delays.
    {SYNTH_REG_ATTR}
    reg                 r_in_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_a;

    wire                 dut_out_valid;
    wire [{wfull - 1}:0] dut_y;

    {SYNTH_REG_ATTR}
    reg                 r_out_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_y;

    assign out_valid = r_out_valid;
    assign y         = r_y;

    zkf_mul_ilog2_const #(
        .WEXP({spec.wexp}),
        .WMAN({spec.wman}),
        .K({MUL_ILOG2_CONST_K}),
        .STAGE_INPUT({spec.stage_input}),
        .STAGE_DECODE({spec.stage_decode}),
        .LATENCY({register_stages(spec)})
    ) dut (
        .clk(clk),
        .rst(rst),
        .in_valid(r_in_valid),
        .a(r_a),
        .out_valid(dut_out_valid),
        .y(dut_y)
    );

    always @(posedge clk) begin
        if (rst) begin
            r_in_valid  <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid  <= in_valid;
            r_out_valid <= dut_out_valid;
        end

        r_a <= a;
        r_y <= dut_y;
    end
endmodule

`default_nettype wire
""")


def write_mul_ilog2_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    wk = spec.wk or (spec.wexp + 1)
    path.write_text(f"""`default_nettype none

module {spec.top} (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 in_valid,
    input  wire [{wfull - 1}:0] a,
    input  wire signed [{wk - 1}:0] k,
    output wire                 out_valid,
    output wire [{wfull - 1}:0] y
);
    // Measurement harness: register every DUT input and output so the timing report includes the primary-I/O paths.
    {SYNTH_REG_ATTR}
    reg                 r_in_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_a;
    {SYNTH_REG_ATTR}
    reg signed [{wk - 1}:0] r_k;

    wire                 dut_out_valid;
    wire [{wfull - 1}:0] dut_y;

    {SYNTH_REG_ATTR}
    reg                 r_out_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_y;

    assign out_valid = r_out_valid;
    assign y         = r_y;

    zkf_mul_ilog2 #(
        .WEXP({spec.wexp}),
        .WMAN({spec.wman}),
        .WK({wk}),
        .STAGE_INPUT({spec.stage_input}),
        .STAGE_DECODE({spec.stage_decode}),
        .LATENCY({register_stages(spec)})
    ) dut (
        .clk(clk),
        .rst(rst),
        .in_valid(r_in_valid),
        .a(r_a),
        .k(r_k),
        .out_valid(dut_out_valid),
        .y(dut_y)
    );

    always @(posedge clk) begin
        if (rst) begin
            r_in_valid  <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid  <= in_valid;
            r_out_valid <= dut_out_valid;
        end

        r_a <= a;
        r_k <= k;
        r_y <= dut_y;
    end
endmodule

`default_nettype wire
""")


def write_from_int_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    wint = spec.wint
    path.write_text(f"""`default_nettype none

module {spec.top} (
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          in_valid,
    input  wire signed [{wint - 1}:0]    a,
    output wire                          out_valid,
    output wire [{wfull - 1}:0]          y
);
    // Measurement harness: register every DUT I/O so the timing report includes register-to-register paths only.
    {SYNTH_REG_ATTR}
    reg                       r_in_valid;
    {SYNTH_REG_ATTR}
    reg signed [{wint - 1}:0] r_a;

    wire                 dut_out_valid;
    wire [{wfull - 1}:0] dut_y;

    {SYNTH_REG_ATTR}
    reg                 r_out_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_y;

    assign out_valid = r_out_valid;
    assign y         = r_y;

    zkf_from_int #(
        .WEXP({spec.wexp}),
        .WMAN({spec.wman}),
        .WINT({wint}),
        .STAGE_INPUT({spec.stage_input}),
        .STAGE_NORMALIZE({spec.stage_normalize}),
        .STAGE_PACK({spec.stage_pack}),
        .STAGE_OUTPUT({spec.stage_output}),
        .LATENCY({register_stages(spec)})
    ) dut (
        .clk(clk),
        .rst(rst),
        .in_valid(r_in_valid),
        .a(r_a),
        .out_valid(dut_out_valid),
        .y(dut_y)
    );

    always @(posedge clk) begin
        if (rst) begin
            r_in_valid  <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid  <= in_valid;
            r_out_valid <= dut_out_valid;
        end

        r_a <= a;
        r_y <= dut_y;
    end
endmodule

`default_nettype wire
""")


def write_to_int_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    wint = spec.wint
    path.write_text(f"""`default_nettype none

module {spec.top} (
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          in_valid,
    input  wire [{wfull - 1}:0]          a,
    output wire                          out_valid,
    output wire signed [{wint - 1}:0]    y
);
    // Measurement harness: register every DUT I/O so the timing report includes register-to-register paths only.
    {SYNTH_REG_ATTR}
    reg                 r_in_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_a;

    wire                       dut_out_valid;
    wire signed [{wint - 1}:0] dut_y;

    {SYNTH_REG_ATTR}
    reg                       r_out_valid;
    {SYNTH_REG_ATTR}
    reg signed [{wint - 1}:0] r_y;

    assign out_valid = r_out_valid;
    assign y         = r_y;

    zkf_to_int #(
        .WEXP({spec.wexp}),
        .WMAN({spec.wman}),
        .WINT({wint}),
        .STAGE_INPUT({spec.stage_input}),
        .LATENCY({register_stages(spec)})
    ) dut (
        .clk(clk),
        .rst(rst),
        .in_valid(r_in_valid),
        .a(r_a),
        .out_valid(dut_out_valid),
        .y(dut_y)
    );

    always @(posedge clk) begin
        if (rst) begin
            r_in_valid  <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid  <= in_valid;
            r_out_valid <= dut_out_valid;
        end

        r_a <= a;
        r_y <= dut_y;
    end
endmodule

`default_nettype wire
""")


def write_resize_wrapper(spec: ModuleSpec, path: Path) -> None:
    in_wfull = spec.wexp_in + spec.wman_in
    out_wfull = spec.wexp_out + spec.wman_out
    path.write_text(f"""`default_nettype none

module {spec.top} (
    input  wire                       clk,
    input  wire                       rst,
    input  wire                       in_valid,
    input  wire [{in_wfull - 1}:0]    a,
    output wire                       out_valid,
    output wire [{out_wfull - 1}:0]   y
);
    // Measurement harness: register every DUT I/O so the timing report includes register-to-register paths only.
    {SYNTH_REG_ATTR}
    reg                    r_in_valid;
    {SYNTH_REG_ATTR}
    reg [{in_wfull - 1}:0] r_a;

    wire                     dut_out_valid;
    wire [{out_wfull - 1}:0] dut_y;

    {SYNTH_REG_ATTR}
    reg                     r_out_valid;
    {SYNTH_REG_ATTR}
    reg [{out_wfull - 1}:0] r_y;

    assign out_valid = r_out_valid;
    assign y         = r_y;

    zkf_resize #(
        .WEXP_IN({spec.wexp_in}),
        .WMAN_IN({spec.wman_in}),
        .WEXP_OUT({spec.wexp_out}),
        .WMAN_OUT({spec.wman_out}),
        .STAGE_INPUT({spec.stage_input}),
        .STAGE_OUTPUT({spec.stage_output}),
        .LATENCY({register_stages(spec)})
    ) dut (
        .clk(clk),
        .rst(rst),
        .in_valid(r_in_valid),
        .a(r_a),
        .out_valid(dut_out_valid),
        .y(dut_y)
    );

    always @(posedge clk) begin
        if (rst) begin
            r_in_valid  <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid  <= in_valid;
            r_out_valid <= dut_out_valid;
        end

        r_a <= a;
        r_y <= dut_y;
    end
endmodule

`default_nettype wire
""")


def write_round_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    path.write_text(f"""`default_nettype none

module {spec.top} (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 in_valid,
    input  wire [{wfull - 1}:0] a,
    input  wire           [1:0] round_mode,
    output wire                 out_valid,
    output wire [{wfull - 1}:0] y
);
    // Measurement harness: register every DUT I/O so the timing report includes register-to-register paths only.
    {SYNTH_REG_ATTR}
    reg                 r_in_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_a;
    {SYNTH_REG_ATTR}
    reg           [1:0] r_round_mode;

    wire                 dut_out_valid;
    wire [{wfull - 1}:0] dut_y;

    {SYNTH_REG_ATTR}
    reg                 r_out_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_y;

    assign out_valid = r_out_valid;
    assign y         = r_y;

    zkf_round #(
        .WEXP({spec.wexp}),
        .WMAN({spec.wman}),
        .STAGE_INPUT({spec.stage_input}),
        .STAGE_DECODE({spec.stage_decode}),
        .STAGE_PACK({spec.stage_pack}),
        .STAGE_OUTPUT({spec.stage_output}),
        .LATENCY({register_stages(spec)})
    ) dut (
        .clk(clk),
        .rst(rst),
        .in_valid(r_in_valid),
        .a(r_a),
        .round_mode(r_round_mode),
        .out_valid(dut_out_valid),
        .y(dut_y)
    );

    always @(posedge clk) begin
        if (rst) begin
            r_in_valid  <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid  <= in_valid;
            r_out_valid <= dut_out_valid;
        end

        r_a          <= a;
        r_round_mode <= round_mode;
        r_y          <= dut_y;
    end
endmodule

`default_nettype wire
""")


def write_exp2_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    path.write_text(f"""`default_nettype none

module {spec.top} (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 in_valid,
    input  wire [{wfull - 1}:0] x,
    output wire                 out_valid,
    output wire [{wfull - 1}:0] y
);
    // Measurement harness: register every DUT I/O so the timing report includes register-to-register paths only.
    {SYNTH_REG_ATTR}
    reg                 r_in_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_x;

    wire                 dut_out_valid;
    wire [{wfull - 1}:0] dut_y;

    {SYNTH_REG_ATTR}
    reg                 r_out_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_y;

    assign out_valid = r_out_valid;
    assign y         = r_y;

    zkf_exp2 #(
        .WEXP({spec.wexp}),
        .WMAN({spec.wman}),
        .STAGE_INPUT({spec.stage_input}),
        .STAGE_REDUCE({spec.stage_reduce}),
        .STAGE_PRODUCT({spec.stage_product}),
        .WMULTIPLIER({spec.wmultiplier}),
        .STAGE_PACK({spec.stage_pack}),
        .STAGE_OUTPUT({spec.stage_output}),
        .LATENCY({register_stages(spec)})
    ) dut (
        .clk(clk),
        .rst(rst),
        .in_valid(r_in_valid),
        .x(r_x),
        .out_valid(dut_out_valid),
        .y(dut_y)
    );

    always @(posedge clk) begin
        if (rst) begin
            r_in_valid  <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid  <= in_valid;
            r_out_valid <= dut_out_valid;
        end

        r_x <= x;
        r_y <= dut_y;
    end
endmodule

`default_nettype wire
""")


def write_log2_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    path.write_text(f"""`default_nettype none

module {spec.top} (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 in_valid,
    input  wire [{wfull - 1}:0] x,
    output wire                 out_valid,
    output wire [{wfull - 1}:0] y,
    output wire                 domain_error,
    output wire                 pole
);
    // Measurement harness: register every DUT I/O so the timing report includes register-to-register paths only.
    {SYNTH_REG_ATTR}
    reg                 r_in_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_x;

    wire                 dut_out_valid;
    wire [{wfull - 1}:0] dut_y;
    wire                 dut_domain_error;
    wire                 dut_pole;

    {SYNTH_REG_ATTR}
    reg                 r_out_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_y;
    {SYNTH_REG_ATTR}
    reg                 r_domain_error;
    {SYNTH_REG_ATTR}
    reg                 r_pole;

    assign out_valid    = r_out_valid;
    assign y            = r_y;
    assign domain_error = r_domain_error;
    assign pole         = r_pole;

    zkf_log2 #(
        .WEXP({spec.wexp}),
        .WMAN({spec.wman}),
        .STAGE_INPUT({spec.stage_input}),
        .STAGE_DECODE({spec.stage_decode}),
        .STAGE_PRODUCT({spec.stage_product}),
        .STAGE_PRODUCT_FINAL({effective_stage_product_final(spec)}),
        .WMULTIPLIER({spec.wmultiplier}),
        .STAGE_NORMALIZE({spec.stage_normalize}),
        .STAGE_NORMALIZE_OUTPUT({spec.stage_normalize_output}),
        .STAGE_PACK({spec.stage_pack}),
        .STAGE_OUTPUT({spec.stage_output}),
        .LATENCY({register_stages(spec)})
    ) dut (
        .clk(clk),
        .rst(rst),
        .in_valid(r_in_valid),
        .x(r_x),
        .out_valid(dut_out_valid),
        .y(dut_y),
        .domain_error(dut_domain_error),
        .pole(dut_pole)
    );

    always @(posedge clk) begin
        if (rst) begin
            r_in_valid  <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid  <= in_valid;
            r_out_valid <= dut_out_valid;
        end

        r_x            <= x;
        r_y            <= dut_y;
        r_domain_error <= dut_domain_error;
        r_pole         <= dut_pole;
    end
endmodule

`default_nettype wire
""")


def write_sincos_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    path.write_text(f"""`default_nettype none

module {spec.top} (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 in_valid,
    output wire                 in_ready,
    input  wire [{wfull - 1}:0] x,
    output wire                 out_valid,
    input  wire                 out_ready,
    output wire [{wfull - 1}:0] sin,
    output wire [{wfull - 1}:0] cos,
    output wire [1:0]           quadrant
);
    // Measurement harness: register every DUT I/O so the timing report includes register-to-register paths only.
    {SYNTH_REG_ATTR}
    reg                 r_in_valid;
    {SYNTH_REG_ATTR}
    reg                 r_out_ready;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_x;

    wire                 dut_in_ready;
    wire                 dut_out_valid;
    wire [{wfull - 1}:0] dut_sin;
    wire [{wfull - 1}:0] dut_cos;
    wire [1:0]           dut_quadrant;

    {SYNTH_REG_ATTR}
    reg                 r_in_ready;
    {SYNTH_REG_ATTR}
    reg                 r_out_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_sin;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_cos;
    {SYNTH_REG_ATTR}
    reg [1:0]           r_quadrant;

    assign in_ready  = r_in_ready;
    assign out_valid = r_out_valid;
    assign sin       = r_sin;
    assign cos       = r_cos;
    assign quadrant  = r_quadrant;

    zkf_sincos #(
        .WEXP({spec.wexp}),
        .WMAN({spec.wman}),
        .WMULTIPLIER({spec.wmultiplier}),
        .UNROLL100({spec.unroll100}),
        .PARALLEL({effective_parallel(spec)}),
        .STAGE_INPUT({spec.stage_input}),
        .STAGE_PRODUCT({spec.stage_product}),
        .STAGE_NORMALIZE({spec.stage_normalize}),
        .STAGE_PACK({spec.stage_pack}),
        .STAGE_OUTPUT({spec.stage_output}),
        .LATENCY({register_stages(spec)})
    ) dut (
        .clk(clk),
        .rst(rst),
        .in_valid(r_in_valid),
        .in_ready(dut_in_ready),
        .x(r_x),
        .out_valid(dut_out_valid),
        .out_ready(r_out_ready),
        .sin(dut_sin),
        .cos(dut_cos),
        .quadrant(dut_quadrant)
    );

    always @(posedge clk) begin
        if (rst) begin
            r_in_valid  <= 1'b0;
            r_in_ready  <= 1'b0;
            r_out_valid <= 1'b0;
            r_out_ready <= 1'b0;
        end else begin
            r_in_valid  <= in_valid;
            r_in_ready  <= dut_in_ready;
            r_out_valid <= dut_out_valid;
            r_out_ready <= out_ready;
        end

        r_x        <= x;
        r_sin      <= dut_sin;
        r_cos      <= dut_cos;
        r_quadrant <= dut_quadrant;
    end
endmodule

`default_nettype wire
""")


def write_atan2_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    path.write_text(f"""`default_nettype none

module {spec.top} (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 in_valid,
    output wire                 in_ready,
    input  wire [{wfull - 1}:0] y,
    input  wire [{wfull - 1}:0] x,
    output wire                 out_valid,
    input  wire                 out_ready,
    output wire [{wfull - 1}:0] theta,
    output wire [{wfull - 1}:0] mag
);
    // Measurement harness: register every DUT I/O so the timing report includes register-to-register paths only.
    {SYNTH_REG_ATTR}
    reg                 r_in_valid;
    {SYNTH_REG_ATTR}
    reg                 r_out_ready;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_y;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_x;

    wire                 dut_in_ready;
    wire                 dut_out_valid;
    wire [{wfull - 1}:0] dut_theta;
    wire [{wfull - 1}:0] dut_mag;

    {SYNTH_REG_ATTR}
    reg                 r_in_ready;
    {SYNTH_REG_ATTR}
    reg                 r_out_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_theta;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_mag;

    assign in_ready  = r_in_ready;
    assign out_valid = r_out_valid;
    assign theta     = r_theta;
    assign mag       = r_mag;

    zkf_atan2 #(
        .WEXP({spec.wexp}),
        .WMAN({spec.wman}),
        .WMULTIPLIER({spec.wmultiplier}),
        .UNROLL100({spec.unroll100}),
        .STAGE_INPUT({spec.stage_input}),
        .STAGE_PRODUCT({spec.stage_product}),
        .STAGE_NORMALIZE({spec.stage_normalize}),
        .STAGE_PACK({spec.stage_pack}),
        .STAGE_OUTPUT({spec.stage_output}),
        .LATENCY({register_stages(spec)})
    ) dut (
        .clk(clk),
        .rst(rst),
        .in_valid(r_in_valid),
        .in_ready(dut_in_ready),
        .y(r_y),
        .x(r_x),
        .out_valid(dut_out_valid),
        .out_ready(r_out_ready),
        .theta(dut_theta),
        .mag(dut_mag)
    );

    always @(posedge clk) begin
        if (rst) begin
            r_in_valid  <= 1'b0;
            r_in_ready  <= 1'b0;
            r_out_valid <= 1'b0;
            r_out_ready <= 1'b0;
        end else begin
            r_in_valid  <= in_valid;
            r_in_ready  <= dut_in_ready;
            r_out_valid <= dut_out_valid;
            r_out_ready <= out_ready;
        end

        r_y     <= y;
        r_x     <= x;
        r_theta <= dut_theta;
        r_mag   <= dut_mag;
    end
endmodule

`default_nettype wire
""")


def write_wrapper(spec: ModuleSpec, path: Path) -> None:
    if spec.kind == "pack":
        write_pack_wrapper(spec, path)
    elif spec.kind == "mul":
        write_mul_wrapper(spec, path)
    elif spec.kind == "add":
        write_add_wrapper(spec, path)
    elif spec.kind == "addsub":
        write_addsub_wrapper(spec, path)
    elif spec.kind == "fma":
        write_fma_wrapper(spec, path)
    elif spec.kind == "div_core":
        write_div_core_wrapper(spec, path)
    elif spec.kind == "div":
        write_div_wrapper(spec, path)
    elif spec.kind == "cmp":
        write_cmp_wrapper(spec, path)
    elif spec.kind == "sort":
        write_sort_wrapper(spec, path)
    elif spec.kind == "mul_ilog2_const":
        write_mul_ilog2_const_wrapper(spec, path)
    elif spec.kind == "mul_ilog2":
        write_mul_ilog2_wrapper(spec, path)
    elif spec.kind == "from_int":
        write_from_int_wrapper(spec, path)
    elif spec.kind == "to_int":
        write_to_int_wrapper(spec, path)
    elif spec.kind == "resize":
        write_resize_wrapper(spec, path)
    elif spec.kind == "round":
        write_round_wrapper(spec, path)
    elif spec.kind == "exp2":
        write_exp2_wrapper(spec, path)
    elif spec.kind == "log2":
        write_log2_wrapper(spec, path)
    elif spec.kind == "sincos":
        write_sincos_wrapper(spec, path)
    elif spec.kind == "atan2":
        write_atan2_wrapper(spec, path)
    else:
        raise ValueError(f"unsupported module kind: {spec.kind}")
