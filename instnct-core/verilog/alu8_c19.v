// ============================================================
// alu8_c19.v — 8-bit ALU using C19 LutGate primitives
//
// Each gate is expressed as a hardcoded LUT, mirroring the
// C19 LutGate representation: integer weights + precomputed
// binary truth-table fed into a barrel-shift lookup.
//
// Ops:  0=ADD  1=SUB  2=MUL  3=AND  4=OR  5=XOR  6=CMP
//       7=NOT  8=MIN  9=MAX
// ============================================================

// ----------------------------------------------------------
// Primitive: 1-bit C19 AND gate (2-input LUT, pattern 4'b1000)
// LutGate::new(&[10,10], bias=-18) → outputs [0,0,0,1]
// ----------------------------------------------------------
module c19_and2(input a, input b, output y);
    wire [1:0] addr = {b, a};
    // LUT[3:0] = 4'b1000 → only (a=1,b=1) → 1
    assign y = 4'b1000 >> addr;
endmodule

// ----------------------------------------------------------
// Primitive: 1-bit C19 OR gate (2-input LUT, pattern 4'b1110)
// ----------------------------------------------------------
module c19_or2(input a, input b, output y);
    wire [1:0] addr = {b, a};
    assign y = 4'b1110 >> addr;
endmodule

// ----------------------------------------------------------
// Primitive: 1-bit C19 XOR gate (2-input LUT, pattern 4'b0110)
// ----------------------------------------------------------
module c19_xor2(input a, input b, output y);
    wire [1:0] addr = {b, a};
    assign y = 4'b0110 >> addr;
endmodule

// ----------------------------------------------------------
// Primitive: 1-bit C19 NOT gate (1-input LUT, pattern 2'b01)
// ----------------------------------------------------------
module c19_not1(input a, output y);
    assign y = 2'b01 >> a;
endmodule

// ----------------------------------------------------------
// Primitive: 2:1 MUX  — sel=0→a, sel=1→b
// LUT: 4'b1100 (addr={sel,a}: 00→0,01→1,10→0? no...)
// Truth table: sel a b | y
//              0   0 x | 0  (a)
//              0   1 x | 1  (a)
//              1   x 0 | 0  (b)
//              1   x 1 | 1  (b)
// 3-input LUT on {sel, b, a}: outputs
//   000→a=0, 001→a=1, 010→a=0, 011→a=1,
//   100→b=0, 101→b=0, 110→b=1, 111→b=1
// pattern = 8'b11001010
// ----------------------------------------------------------
module c19_mux2(input a, input b, input sel, output y);
    wire [2:0] addr = {sel, b, a};
    assign y = 8'b11001010 >> addr;
endmodule

// ----------------------------------------------------------
// Primitive: C19 Full Adder (two 3-input LUTs)
// XOR3 LUT: [0,1,1,0,1,0,0,1] → 8'b10010110
// MAJ  LUT: [0,0,0,1,0,1,1,1] → 8'b11101000
// ----------------------------------------------------------
module c19_full_adder(
    input  a,
    input  b,
    input  cin,
    output sum,
    output cout
);
    wire [2:0] addr = {cin, b, a};
    // XOR3: sum = a^b^cin
    assign sum  = 8'b10010110 >> addr;
    // MAJ: cout = majority(a,b,cin)
    assign cout = 8'b11101000 >> addr;
endmodule

// ----------------------------------------------------------
// 8-bit ripple-carry adder (8 chained C19 full adders)
// Returns 9-bit result: {cout, sum[7:0]}
// ----------------------------------------------------------
module c19_add8(
    input  [7:0] a,
    input  [7:0] b,
    input        cin,
    output [7:0] sum,
    output       cout
);
    wire [8:0] carry; // carry[0]=cin, carry[8]=cout
    assign carry[0] = cin;

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : fa_chain
            c19_full_adder fa_i (
                .a   (a[i]),
                .b   (b[i]),
                .cin (carry[i]),
                .sum (sum[i]),
                .cout(carry[i+1])
            );
        end
    endgenerate

    assign cout = carry[8];
endmodule

// ----------------------------------------------------------
// 8-bit subtractor: sub = a + (~b) + 1  (two's complement)
// ----------------------------------------------------------
module c19_sub8(
    input  [7:0] a,
    input  [7:0] b,
    output [7:0] diff,
    output       bout   // borrow (inverted carry)
);
    wire [7:0] b_inv;
    wire       carry_out;

    // Invert b bitwise using 8 C19 NOT gates
    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : inv_b
            c19_not1 not_i(.a(b[i]), .y(b_inv[i]));
        end
    endgenerate

    // Add a + ~b + 1 (cin=1 → two's complement subtract)
    c19_add8 adder(
        .a   (a),
        .b   (b_inv),
        .cin (1'b1),
        .sum (diff),
        .cout(carry_out)
    );

    // For subtraction, borrow = ~carry_out
    assign bout = ~carry_out;
endmodule

// ----------------------------------------------------------
// 8×8 array multiplier → 16-bit product
// Stage 1: 64 AND gates produce partial products
// Stage 2: adder tree reduces to final result
// ----------------------------------------------------------
module c19_mul8(
    input  [7:0] a,
    input  [7:0] b,
    output [15:0] product
);
    // Partial products: pp[i][j] = a[i] & b[j]
    wire pp [0:7][0:7];

    genvar i, j;
    generate
        for (i = 0; i < 8; i = i + 1) begin : pp_row
            for (j = 0; j < 8; j = j + 1) begin : pp_col
                c19_and2 and_ij(.a(a[i]), .b(b[j]), .y(pp[i][j]));
            end
        end
    endgenerate

    // Row 0 is the first partial product (no addition needed)
    // We accumulate rows using 8-bit adders

    // acc holds the running accumulated sum (up to 16 bits)
    wire [15:0] row [0:7];
    assign row[0] = {8'b0, pp[0][0], 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0};

    // Build each row: row[i] = {shifted partial product of b[i]}
    // pp[i][j] contributes to bit (i+j) of the product
    // For simplicity use Verilog reduction; Yosys will infer LUT tree
    // (This is equivalent to what a LUT-based synthesizer would produce)
    wire [15:0] pp_shift [0:7];
    generate
        for (i = 0; i < 8; i = i + 1) begin : shift_row
            assign pp_shift[i] = {
                pp[7][i], pp[6][i], pp[5][i], pp[4][i],
                pp[3][i], pp[2][i], pp[1][i], pp[0][i],
                {i{1'b0}}
            } >> 0; // shift handled by bit placement below
        end
    endgenerate

    // Explicit bit-addressed partial product accumulation
    // product[k] = sum of all pp[i][j] where i+j == k
    // Implement as Wallace-like reduction using C19 full adders

    // For a clean Yosys-synthesizable design, use the standard
    // multi-operand addition which Yosys maps to LUT trees:
    wire [15:0] pp0  = { 8'b0, pp[0][7], pp[0][6], pp[0][5], pp[0][4],
                                pp[0][3], pp[0][2], pp[0][1], pp[0][0] };
    wire [15:0] pp1  = { 7'b0, pp[1][7], pp[1][6], pp[1][5], pp[1][4],
                                pp[1][3], pp[1][2], pp[1][1], pp[1][0], 1'b0 };
    wire [15:0] pp2  = { 6'b0, pp[2][7], pp[2][6], pp[2][5], pp[2][4],
                                pp[2][3], pp[2][2], pp[2][1], pp[2][0], 2'b0 };
    wire [15:0] pp3  = { 5'b0, pp[3][7], pp[3][6], pp[3][5], pp[3][4],
                                pp[3][3], pp[3][2], pp[3][1], pp[3][0], 3'b0 };
    wire [15:0] pp4  = { 4'b0, pp[4][7], pp[4][6], pp[4][5], pp[4][4],
                                pp[4][3], pp[4][2], pp[4][1], pp[4][0], 4'b0 };
    wire [15:0] pp5  = { 3'b0, pp[5][7], pp[5][6], pp[5][5], pp[5][4],
                                pp[5][3], pp[5][2], pp[5][1], pp[5][0], 5'b0 };
    wire [15:0] pp6  = { 2'b0, pp[6][7], pp[6][6], pp[6][5], pp[6][4],
                                pp[6][3], pp[6][2], pp[6][1], pp[6][0], 6'b0 };
    wire [15:0] pp7  = { 1'b0, pp[7][7], pp[7][6], pp[7][5], pp[7][4],
                                pp[7][3], pp[7][2], pp[7][1], pp[7][0], 7'b0 };

    assign product = pp0 + pp1 + pp2 + pp3 + pp4 + pp5 + pp6 + pp7;
endmodule

// ----------------------------------------------------------
// 8-bit bitwise AND (8 parallel C19 AND2 gates)
// ----------------------------------------------------------
module c19_and8(input [7:0] a, input [7:0] b, output [7:0] y);
    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : and_bits
            c19_and2 g(.a(a[i]), .b(b[i]), .y(y[i]));
        end
    endgenerate
endmodule

// ----------------------------------------------------------
// 8-bit bitwise OR
// ----------------------------------------------------------
module c19_or8(input [7:0] a, input [7:0] b, output [7:0] y);
    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : or_bits
            c19_or2 g(.a(a[i]), .b(b[i]), .y(y[i]));
        end
    endgenerate
endmodule

// ----------------------------------------------------------
// 8-bit bitwise XOR
// ----------------------------------------------------------
module c19_xor8(input [7:0] a, input [7:0] b, output [7:0] y);
    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : xor_bits
            c19_xor2 g(.a(a[i]), .b(b[i]), .y(y[i]));
        end
    endgenerate
endmodule

// ----------------------------------------------------------
// 8-bit bitwise NOT
// ----------------------------------------------------------
module c19_not8(input [7:0] a, output [7:0] y);
    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : not_bits
            c19_not1 g(.a(a[i]), .y(y[i]));
        end
    endgenerate
endmodule

// ----------------------------------------------------------
// 8-bit comparator: returns {eq, lt, gt} flags
// Uses subtractor: a < b iff borrow=1
// ----------------------------------------------------------
module c19_cmp8(
    input  [7:0] a,
    input  [7:0] b,
    output        eq,   // a == b
    output        lt,   // a <  b (unsigned)
    output        gt    // a >  b (unsigned)
);
    wire [7:0] diff;
    wire       borrow;

    c19_sub8 sub(.a(a), .b(b), .diff(diff), .bout(borrow));

    // eq: all diff bits are zero → 8-input NOR chain
    wire eq_lo, eq_hi;
    wire [7:0] nor_in = diff;
    // Use XOR-based zero detect (diff == 0 iff ~|diff)
    wire nz0, nz1, nz2, nz3, nz4, nz5, nz6, nz7;
    c19_or2 nz_0(.a(nor_in[0]), .b(nor_in[1]), .y(nz0));
    c19_or2 nz_1(.a(nor_in[2]), .b(nor_in[3]), .y(nz1));
    c19_or2 nz_2(.a(nor_in[4]), .b(nor_in[5]), .y(nz2));
    c19_or2 nz_3(.a(nor_in[6]), .b(nor_in[7]), .y(nz3));
    c19_or2 nz_4(.a(nz0),       .b(nz1),        .y(nz4));
    c19_or2 nz_5(.a(nz2),       .b(nz3),        .y(nz5));
    wire    nz6_w;
    c19_or2 nz_6(.a(nz4),       .b(nz5),        .y(nz6_w));
    c19_not1 eq_inv(.a(nz6_w), .y(eq));

    assign lt = borrow & ~eq;
    assign gt = ~borrow & ~eq;
endmodule

// ----------------------------------------------------------
// 8-bit MUX: sel=0 → a, sel=1 → b
// ----------------------------------------------------------
module c19_mux8(
    input  [7:0] a,
    input  [7:0] b,
    input        sel,
    output [7:0] y
);
    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : mux_bits
            c19_mux2 g(.a(a[i]), .b(b[i]), .sel(sel), .y(y[i]));
        end
    endgenerate
endmodule

// ----------------------------------------------------------
// 8-bit MIN: min(a, b) = a if a<b else b
// ----------------------------------------------------------
module c19_min8(input [7:0] a, input [7:0] b, output [7:0] y);
    wire eq, lt, gt;
    c19_cmp8 cmp(.a(a), .b(b), .eq(eq), .lt(lt), .gt(gt));
    // sel=1 → b, sel=0 → a; select a when a<=b (lt|eq)
    wire sel_b;
    c19_not1 inv_lt(.a(lt), .y(sel_b));  // sel_b=1 when NOT lt → choose b?
    // Correct: if lt=1 → a<b → output a (sel=0)
    //          if lt=0 → b<=a → output b (sel=1)
    c19_mux8 mux(.a(a), .b(b), .sel(sel_b), .y(y));
endmodule

// ----------------------------------------------------------
// 8-bit MAX: max(a, b) = b if a<b else a
// ----------------------------------------------------------
module c19_max8(input [7:0] a, input [7:0] b, output [7:0] y);
    wire eq, lt, gt;
    c19_cmp8 cmp(.a(a), .b(b), .eq(eq), .lt(lt), .gt(gt));
    // if lt=1 → a<b → output b (sel=1)
    c19_mux8 mux(.a(a), .b(b), .sel(lt), .y(y));
endmodule

// ============================================================
// TOP-LEVEL: alu8_c19
// ============================================================
module alu8_c19(
    input  [7:0] a,
    input  [7:0] b,
    input  [3:0] op,       // 0=ADD 1=SUB 2=MUL 3=AND 4=OR 5=XOR 6=CMP 7=NOT 8=MIN 9=MAX
    output [7:0] result,
    output [15:0] mul_result,
    output        z_flag,
    output        n_flag,
    output        c_flag
);
    // --- Compute all operations ---
    wire [7:0] add_sum;   wire add_cout;
    wire [7:0] sub_diff;  wire sub_bout;
    wire [7:0] and_out;
    wire [7:0] or_out;
    wire [7:0] xor_out;
    wire [7:0] not_out;
    wire [7:0] min_out;
    wire [7:0] max_out;
    wire        cmp_eq, cmp_lt, cmp_gt;
    wire [7:0]  cmp_out;  // {5'b0, gt, lt, eq}

    c19_add8  u_add (.a(a), .b(b), .cin(1'b0), .sum(add_sum), .cout(add_cout));
    c19_sub8  u_sub (.a(a), .b(b), .diff(sub_diff), .bout(sub_bout));
    c19_mul8  u_mul (.a(a), .b(b), .product(mul_result));
    c19_and8  u_and (.a(a), .b(b), .y(and_out));
    c19_or8   u_or  (.a(a), .b(b), .y(or_out));
    c19_xor8  u_xor (.a(a), .b(b), .y(xor_out));
    c19_not8  u_not (.a(a), .y(not_out));
    c19_cmp8  u_cmp (.a(a), .b(b), .eq(cmp_eq), .lt(cmp_lt), .gt(cmp_gt));
    c19_min8  u_min (.a(a), .b(b), .y(min_out));
    c19_max8  u_max (.a(a), .b(b), .y(max_out));

    assign cmp_out = {5'b0, cmp_gt, cmp_lt, cmp_eq};

    // --- 16:1 output mux (op selects) ---
    reg [7:0] result_r;
    always @(*) begin
        case (op)
            4'd0:    result_r = add_sum;
            4'd1:    result_r = sub_diff;
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
    // Z: result is zero
    wire nz0, nz1, nz2, nz3, nz4, nz5;
    c19_or2 f0(.a(result[0]), .b(result[1]), .y(nz0));
    c19_or2 f1(.a(result[2]), .b(result[3]), .y(nz1));
    c19_or2 f2(.a(result[4]), .b(result[5]), .y(nz2));
    c19_or2 f3(.a(result[6]), .b(result[7]), .y(nz3));
    c19_or2 f4(.a(nz0),       .b(nz1),       .y(nz4));
    c19_or2 f5(.a(nz2),       .b(nz3),       .y(nz5));
    wire nz_final;
    c19_or2 f6(.a(nz4),       .b(nz5),       .y(nz_final));
    c19_not1 zf(.a(nz_final), .y(z_flag));

    assign n_flag = result[7];
    assign c_flag = (op == 4'd0) ? add_cout : sub_bout;
endmodule
