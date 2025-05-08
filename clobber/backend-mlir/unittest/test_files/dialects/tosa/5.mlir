module {
  // main adds tensors and applies ReLU
  func.func @main() -> tensor<2xf32> {
    %a = tosa.const dense<[-1.0, 5.0]> : tensor<2xf32>
    %b = tosa.const dense<[0.5, 0.5]> : tensor<2xf32>
    %sum = tosa.add %a, %b : tensor<2xf32>
    %relu = tosa.relu %sum : tensor<2xf32>
    return %relu : tensor<2xf32>
  }
}