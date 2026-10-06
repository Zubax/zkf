"""
Verilog measurement-harness generators for the float synthesis suite.

Each write_*_wrapper emits a synthesis top that registers every DUT input and output, so the reported
f max is a register-to-register limit rather than ignoring primary I/O paths. The harness is identical
regardless of the target device/tool, so this module is device-independent. Not runnable on its own.
"""

from __future__ import annotations

from pathlib import Path

from common import SYNTH_REG_ATTR
from modules import ModuleSpec, model_for


def _verilog_params(spec: ModuleSpec) -> str:
    return model_for(spec).verilog_params.replace(", ", ",\n        ")


def write_pack_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    model = model_for(spec)
    wexp_unbiased = model.params["WEXP_UNBIASED"]
    params = model.verilog_params.replace(", ", ",\n        ")
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
        {params}
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
    params = _verilog_params(spec)
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
        {params}
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
    params = _verilog_params(spec)
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
        {params}
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
    params = _verilog_params(spec)
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
        {params}
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
    params = _verilog_params(spec)
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
        {params}
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


def write_cmp_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    params = _verilog_params(spec)
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
        {params}
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
    params = _verilog_params(spec)
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
        {params}
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


def write_mul_ilog2_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    model = model_for(spec)
    wk = model.params["WK"]
    params = model.verilog_params.replace(", ", ",\n        ")
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
        {params}
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
    params = _verilog_params(spec)
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
        {params}
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


def write_ilog2_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    wint = spec.wint
    params = _verilog_params(spec)
    path.write_text(f"""`default_nettype none

module {spec.top} (
    input wire clk,
    input wire rst,
    input wire in_valid,
    input wire [{wfull - 1}:0] a,
    output wire out_valid,
    output wire signed [{wint - 1}:0] y,
    output wire zero,
    output wire infinity,
    output wire negative
);
    {SYNTH_REG_ATTR}
    reg r_in_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_a;
    wire dut_out_valid;
    wire signed [{wint - 1}:0] dut_y;
    wire [2:0] dut_flags;
    {SYNTH_REG_ATTR}
    reg r_out_valid;
    {SYNTH_REG_ATTR}
    reg signed [{wint - 1}:0] r_y;
    {SYNTH_REG_ATTR}
    reg [2:0] r_flags;

    assign out_valid = r_out_valid;
    assign y = r_y;
    assign {{zero, infinity, negative}} = r_flags;

    zkf_ilog2 #(
        {params}
    ) dut (
        .clk(clk), .rst(rst), .in_valid(r_in_valid), .a(r_a),
        .out_valid(dut_out_valid), .y(dut_y),
        .zero(dut_flags[2]), .infinity(dut_flags[1]), .negative(dut_flags[0])
    );

    always @(posedge clk) begin
        if (rst) begin
            r_in_valid <= 1'b0;
            r_out_valid <= 1'b0;
        end else begin
            r_in_valid <= in_valid;
            r_out_valid <= dut_out_valid;
        end
        r_a <= a;
        r_y <= dut_y;
        r_flags <= dut_flags;
    end
endmodule

`default_nettype wire
""")


def write_resize_wrapper(spec: ModuleSpec, path: Path) -> None:
    in_wfull = spec.wexp_in + spec.wman_in
    out_wfull = spec.wexp_out + spec.wman_out
    params = _verilog_params(spec)
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
        {params}
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


def write_rint_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    params = _verilog_params(spec)
    path.write_text(f"""`default_nettype none

module {spec.top} (
    input  wire                        clk,
    input  wire                        rst,
    input  wire                        in_valid,
    input  wire [{wfull - 1}:0]        a,
    input  wire                  [1:0] round_mode,
    output wire                        out_valid,
    output wire [{wfull - 1}:0]        y_float,
    output wire signed [{spec.wint - 1}:0] y_int
);
    // Measurement harness: register every DUT I/O so the timing report includes register-to-register paths only.
    {SYNTH_REG_ATTR}
    reg                 r_in_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0] r_a;
    {SYNTH_REG_ATTR}
    reg           [1:0] r_round_mode;

    wire                        dut_out_valid;
    wire [{wfull - 1}:0]        dut_y_float;
    wire signed [{spec.wint - 1}:0] dut_y_int;

    {SYNTH_REG_ATTR}
    reg                        r_out_valid;
    {SYNTH_REG_ATTR}
    reg [{wfull - 1}:0]        r_y_float;
    {SYNTH_REG_ATTR}
    reg signed [{spec.wint - 1}:0] r_y_int;

    assign out_valid = r_out_valid;
    assign y_float   = r_y_float;
    assign y_int     = r_y_int;

    zkf_rint #(
        {params}
    ) dut (
        .clk(clk),
        .rst(rst),
        .in_valid(r_in_valid),
        .a(r_a),
        .round_mode(r_round_mode),
        .out_valid(dut_out_valid),
        .y_float(dut_y_float),
        .y_int(dut_y_int)
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
        r_y_float    <= dut_y_float;
        r_y_int      <= dut_y_int;
    end
endmodule

`default_nettype wire
""")


def write_exp2_wrapper(spec: ModuleSpec, path: Path) -> None:
    wfull = spec.wexp + spec.wman
    params = _verilog_params(spec)
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
        {params}
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
    params = _verilog_params(spec)
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
        {params}
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


def write_cordic_wrapper(spec: ModuleSpec, path: Path) -> None:
    # A fixed mode gets only the ports it uses, as an instance in a design would leave the others tied off or open.
    wfull = spec.wexp + spec.wman
    inputs = {0: ("a",), 1: ("a", "b"), 2: ("vectoring", "a", "b")}[spec.mode]
    outputs = ("r0", "r1") if spec.mode == 1 else ("r0", "r1", "quadrant")
    unused = {"vectoring": f"1'b{int(spec.mode == 1)}", "b": f"{{{wfull}{{1'b0}}}}", "quadrant": ""}  # "" = open
    widths = {"vectoring": "", "quadrant": "[1:0] "}
    w = {name: widths.get(name, f"[{wfull - 1}:0] ") for name in inputs + outputs}
    w.update({name: "" for name in ("in_valid", "in_ready", "out_valid", "out_ready")})
    head = ("in_valid", "in_ready", *inputs, "out_valid", "out_ready", *outputs)
    direction = {name: "output" if name in ("in_ready", "out_valid", *outputs) else "input " for name in head}
    captured = ("in_valid", "out_ready", *inputs)
    driven = ("in_ready", "out_valid", *outputs)
    control = (("in_valid", "in_valid"), ("in_ready", "dut_in_ready"), ("out_valid", "dut_out_valid"))
    control += (("out_ready", "out_ready"),)
    connections = {name: f"r_{name}" if name in captured else f"dut_{name}" for name in head}
    dut = ("in_valid", "in_ready", "vectoring", "a", "b", "out_valid", "out_ready", "r0", "r1", "quadrant")
    lines = ["`default_nettype none", "", f"module {spec.top} (", "    input  wire clk,", "    input  wire rst,"]
    lines += [f"    {direction[n]} wire {w[n]}{n}{',' if i < len(head) - 1 else ''}" for i, n in enumerate(head)]
    lines += [
        ");",
        "    // Measurement harness: register every DUT I/O so the timing report includes register-to-register",
    ]
    lines += ["    // paths only."]
    for name in captured:
        lines += [f"    {SYNTH_REG_ATTR}", f"    reg {w[name]}r_{name};"]
    lines += [f"    wire {w[name]}dut_{name};" for name in driven]
    for name in driven:
        lines += [f"    {SYNTH_REG_ATTR}", f"    reg {w[name]}r_{name};"]
    lines += [f"    assign {name} = r_{name};" for name in driven]
    lines += ["    zkf_cordic #(", f"        {_verilog_params(spec)}", "    ) dut (", "        .clk(clk),"]
    lines += ["        .rst(rst),"] + [
        f"        .{n}({connections[n] if n in connections else unused[n]})," for n in dut
    ]
    lines[-1] = lines[-1].rstrip(",")
    lines += ["    );", "    always @(posedge clk) begin", "        if (rst) begin"]
    lines += [f"            r_{name} <= 1'b0;" for name, _ in control] + ["        end else begin"]
    lines += [f"            r_{name} <= {source};" for name, source in control] + ["        end"]
    lines += [f"        r_{name} <= {name};" for name in inputs] + [
        f"        r_{name} <= dut_{name};" for name in outputs
    ]
    path.write_text("\n".join(lines + ["    end", "endmodule", "", "`default_nettype wire", ""]))


def write_divsqrt_wrapper(spec: ModuleSpec, path: Path) -> None:
    # A fixed mode gets only the inputs it uses, as an instance in a design would tie the others off.
    wfull = spec.wexp + spec.wman
    inputs = {0: ("a", "b"), 1: ("a",), 2: ("op_sqrt", "a", "b")}[spec.mode]
    outputs = ("y", "error")
    width = {name: f"[{wfull - 1}:0] " for name in ("a", "b", "y")}
    unused = {"op_sqrt": "1'b0", "b": f"{{{wfull}{{1'b0}}}}"}
    lines = ["`default_nettype none", "", f"module {spec.top} (", "    input  wire clk,", "    input  wire rst,"]
    lines += ["    input  wire in_valid,"] + [f"    input  wire {width.get(n, '')}{n}," for n in inputs]
    lines += ["    output wire out_valid,"] + [f"    output wire {width.get(n, '')}{n}," for n in outputs]
    lines[-1] = lines[-1].rstrip(",")
    lines += [
        ");",
        "    // Measurement harness: register every DUT I/O so the timing report includes register-to-register",
    ]
    lines += ["    // paths only."]
    for name in ("in_valid", *inputs, "out_valid", *outputs):
        lines += [f"    {SYNTH_REG_ATTR}", f"    reg {width.get(name, '')}r_{name};"]
    lines += [f"    wire {width.get(name, '')}dut_{name};" for name in ("out_valid", *outputs)]
    lines += [f"    assign {name} = r_{name};" for name in ("out_valid", *outputs)]
    lines += ["    zkf_divsqrt #(", f"        {_verilog_params(spec)}", "    ) dut (", "        .clk(clk),"]
    lines += ["        .rst(rst),", "        .in_valid(r_in_valid),"]
    for name in ("op_sqrt", "a", "b"):
        lines.append(f"        .{name}({f'r_{name}' if name in inputs else unused[name]}),")
    lines.append("        .out_valid(dut_out_valid),")
    lines += [f"        .{name}(dut_{name})," for name in outputs]
    lines[-1] = lines[-1].rstrip(",")
    lines += ["    );", "    always @(posedge clk) begin", "        if (rst) begin"]
    lines += ["            r_in_valid  <= 1'b0;", "            r_out_valid <= 1'b0;", "        end else begin"]
    lines += ["            r_in_valid  <= in_valid;", "            r_out_valid <= dut_out_valid;", "        end"]
    lines += [f"        r_{name} <= {name};" for name in inputs] + [
        f"        r_{name} <= dut_{name};" for name in outputs
    ]
    path.write_text("\n".join(lines + ["    end", "endmodule", "", "`default_nettype wire", ""]))


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
    elif spec.kind == "divsqrt":
        write_divsqrt_wrapper(spec, path)
    elif spec.kind == "cmp":
        write_cmp_wrapper(spec, path)
    elif spec.kind == "sort":
        write_sort_wrapper(spec, path)
    elif spec.kind == "ilog2":
        write_ilog2_wrapper(spec, path)
    elif spec.kind == "mul_ilog2":
        write_mul_ilog2_wrapper(spec, path)
    elif spec.kind == "from_int":
        write_from_int_wrapper(spec, path)
    elif spec.kind == "resize":
        write_resize_wrapper(spec, path)
    elif spec.kind == "rint":
        write_rint_wrapper(spec, path)
    elif spec.kind == "exp2":
        write_exp2_wrapper(spec, path)
    elif spec.kind == "log2":
        write_log2_wrapper(spec, path)
    elif spec.kind == "cordic":
        write_cordic_wrapper(spec, path)
    else:
        raise ValueError(f"unsupported module kind: {spec.kind}")
