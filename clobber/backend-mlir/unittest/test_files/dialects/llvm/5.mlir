module {
  func.func @main() -> i32 {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c1 = arith.constant 1 : i32
    %sum = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %c0) -> (i32) {
      %new = arith.addi %acc, %i : i32
      scf.yield %new : i32
    }
    return %sum : i32
  }
}
