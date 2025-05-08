module {
  // main negates a constant tensor
  func.func @main() -> tensor<2xf32> {
    %input = tosa.const dense<[1.0, -2.0]> : tensor<2xf32>
    %neg = tosa.negate %input : tensor<2xf32>
    return %neg : tensor<2xf32>
  }
}