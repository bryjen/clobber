module {
  func.func @main() -> i32 {
    %size = arith.constant 4 : index
    %mem = memref.alloca() : memref<4xi32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %val = arith.constant 123 : i32
    memref.store %val, %mem[%c1] : memref<4xi32>
    %loaded = memref.load %mem[%c1] : memref<4xi32>
    return %loaded : i32
  }
}
