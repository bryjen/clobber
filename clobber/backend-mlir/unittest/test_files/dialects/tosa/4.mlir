module {
  // main multiplies two tensors then subtracts a constant
  func.func @main() -> tensor<2xf32> {
    %x = tosa.const dense<[2.0, 4.0]> : tensor<2xf32>
    %y = tosa.const dense<[3.0, 1.5]> : tensor<2xf32>
    %prod = tosa.mul %x, %y : tensor<2xf32>
    %sub = tosa.sub %prod, %y : tensor<2xf32>
    return %sub : tensor<2xf32>
  }
}