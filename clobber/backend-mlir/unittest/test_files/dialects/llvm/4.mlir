module {
  func.func @main(%x: i32) -> i32 {
    %zero = arith.constant 0 : i32
    %cond = arith.cmpi "eq", %x, %zero : i32
    cf.cond_br %cond, ^if_true, ^if_false

  ^if_true:
    %r1 = arith.constant 100 : i32
    cf.br ^end(%r1 : i32)

  ^if_false:
    %r2 = arith.constant 200 : i32
    cf.br ^end(%r2 : i32)

  ^end(%result: i32):
    return %result : i32
  }
}
