func.func @main() {
  %a = arith.constant 3 : i32
  %b = arith.constant 4 : i32
  %sum = arith.addi %a, %b : i32
  call @print_i32(%sum) : (i32) -> ()
  return
}