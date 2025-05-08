module {
  func.func @main(%A: memref<2x2xi32>, %B: memref<2x2xi32>, %C: memref<2x2xi32>) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index

    scf.for %i = %c0 to %c2 step %c1 {
      scf.for %j = %c0 to %c2 step %c1 {
        %init = arith.constant 0 : i32
        %sum = scf.for %k = %c0 to %c2 step %c1 iter_args(%acc = %init) -> (i32) {
          %a = memref.load %A[%i, %k] : memref<2x2xi32>
          %b = memref.load %B[%k, %j] : memref<2x2xi32>
          %prod = arith.muli %a, %b : i32
          %acc1 = arith.addi %acc, %prod : i32
          scf.yield %acc1 : i32
        }
        memref.store %sum, %C[%i, %j] : memref<2x2xi32>
      }
    }
    return
  }
}
