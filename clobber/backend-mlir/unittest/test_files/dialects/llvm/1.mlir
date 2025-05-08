module {
  func.func @main() -> i32 {
    %c = arith.constant 42 : i32
    return %c : i32
  }
}
