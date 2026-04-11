// ============================================================
// alu8_standard.v — 8-bit ALU using standard Verilog operators
//
// This is the reference implementation. Uses +, -, *, &, |,
// ^, ~ operators — synthesis tools map these to technology
// cells/LUTs automatically.
//
// Ops:  0=ADD  1=SUB  2=MUL  3=AND  4=OR  5=XOR  6=CMP
//       7=NOT  8=MIN  9=MAX
// ============================================================

module alu8_standard(
    input  [7:0]  a,
    input  [7:0]  b,
    input  [3:0]  op,
    output [7:0]  result,
    output [15:0] mul_result,
    output        z_flag,
    output        n_flag,
    output        c_flag
);
    // --- Compute all operations ---
    wire [8:0]  add_result = {1'b0, a} + {1'b0, b};        // 9-bit: carry in [8]
    wire [8:0]  sub_result = {1'b0, a} - {1'b0, b};        // borrow in [8]
    assign      mul_result = a * b;                          // 16-bit product

    wire [7:0]  and_out = a & b;
    wire [7:0]  or_out  = a | b;
    wire [7:0]  xor_out = a ^ b;
    wire [7:0]  not_out = ~a;

    // Comparator
    wire cmp_eq = (a == b);
    wire cmp_lt = (a <  b);   // unsigned comparison
    wire cmp_gt = (a >  b);
    wire [7:0] cmp_out = {5'b0, cmp_gt, cmp_lt, cmp_eq};

    // MIN / MAX via comparison + mux
    wire [7:0] min_out = cmp_lt ? a : b;
    wire [7:0] max_out = cmp_gt ? a : b;

    // --- Output mux ---
    reg [7:0] result_r;
    always @(*) begin
        case (op)
            4'd0:    result_r = add_result[7:0];
            4'd1:    result_r = sub_result[7:0];
            4'd2:    result_r = mul_result[7:0];
            4'd3:    result_r = and_out;
            4'd4:    result_r = or_out;
            4'd5:    result_r = xor_out;
            4'd6:    result_r = cmp_out;
            4'd7:    result_r = not_out;
            4'd8:    result_r = min_out;
            4'd9:    result_r = max_out;
            default: result_r = 8'h00;
        endcase
    end
    assign result = result_r;

    // --- Flags ---
    assign z_flag = (result == 8'h00);
    assign n_flag = result[7];
    assign c_flag = (op == 4'd0) ? add_result[8]
                  : (op == 4'd1) ? sub_result[8]
                  : 1'b0;
endmodule
