// Bench top for test_cordic_modes: one table instance per MODE; WMAN (18 or 53) and the widths come from the matrix.

`default_nettype none

module _zkf_cordic_modes_tb #(
    parameter integer WMAN      = 18,
    parameter integer UNROLL100 = 100,
    parameter integer PARALLEL  = 0,
    parameter integer WX        = 2,
    parameter integer WZ        = 2
) (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,
    input  wire                 vectoring,
    input  wire signed [WX-1:0] x0,
    input  wire signed [WX-1:0] y0,
    input  wire signed [WZ-1:0] z0,
    output wire           [2:0] busy,
    output wire           [2:0] done,
    output wire           [2:0] z_done,
    output wire signed [WX-1:0] xn_rot, xn_vec, xn_run,
    output wire signed [WX-1:0] yn_rot, yn_vec, yn_run,
    output wire signed [WZ-1:0] zn_rot, zn_vec, zn_run
);
    wire signed [WX-1:0] xn [0:2];
    wire signed [WX-1:0] yn [0:2];
    wire signed [WZ-1:0] zn [0:2];
    assign {xn_run, xn_vec, xn_rot} = {xn[2], xn[1], xn[0]};
    assign {yn_run, yn_vec, yn_rot} = {yn[2], yn[1], yn[0]};
    assign {zn_run, zn_vec, zn_rot} = {zn[2], zn[1], zn[0]};

    genvar m;
    generate
        for (m = 0; m < 3; m = m + 1) begin : g_mode
            localparam integer PAR = (m == 1) ? 0 : PARALLEL;
            if (WMAN == 18) begin : g_m18
                _zkf_cordic_m18 #(
                    .MODE(m), .UNROLL100(UNROLL100), .PARALLEL(PAR), .EXPECT_WX(WX), .EXPECT_WZ(WZ)
                ) u_cordic (
                    .clk(clk), .rst(rst), .start(start), .vectoring(vectoring), .x0(x0), .y0(y0),
                    .z0(z0), .busy(busy[m]), .done(done[m]), .z_done(z_done[m]), .xn(xn[m]),
                    .yn(yn[m]), .zn(zn[m]), .const2pi(), .inv_tau(), .kinv_mag()
                );
            end else begin : g_m53
                _zkf_cordic_m53 #(
                    .MODE(m), .UNROLL100(UNROLL100), .PARALLEL(PAR), .EXPECT_WX(WX), .EXPECT_WZ(WZ)
                ) u_cordic (
                    .clk(clk), .rst(rst), .start(start), .vectoring(vectoring), .x0(x0), .y0(y0),
                    .z0(z0), .busy(busy[m]), .done(done[m]), .z_done(z_done[m]), .xn(xn[m]),
                    .yn(yn[m]), .zn(zn[m]), .const2pi(), .inv_tau(), .kinv_mag()
                );
            end
        end
    endgenerate
endmodule

`default_nettype wire
