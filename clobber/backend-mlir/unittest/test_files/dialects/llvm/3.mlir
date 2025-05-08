module {
  func.func @main(%x: i32, %y: i32) -> i32 {
    %r = arith.muli %x, %y : i32
    return %r : i32
  }
}
