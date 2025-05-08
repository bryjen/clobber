module {
  // Reduce a 2D tensor along axis 1 to get shape <2xf32>
  func.func @main() -> tensor<2xf32> {
    %input = tosa.const dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %axis = tosa.const dense<1> : tensor<1xi32>
    %reduced = tosa.reduce_sum %input, %axis : (tensor<2x2xf32>, tensor<1xi32>) -> tensor<2xf32>
    return %reduced : tensor<2xf32>
  }
}