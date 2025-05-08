module {
  // main adds two constant tensors
  func.func @main() -> tensor<2xf32> {
    %a = tosa.const dense<[1.0, 2.0]> : tensor<2xf32>
    %b = tosa.const dense<[3.0, 4.0]> : tensor<2xf32>
    %sum = tosa.add %a, %b : tensor<2xf32>
    return %sum : tensor<2xf32>
  }
}