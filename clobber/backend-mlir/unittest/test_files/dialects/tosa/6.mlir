module {
  // main does matrix multiply, adds bias, applies ReLU
  func.func @main(
    %lhs: tensor<1x128xf32>,
    %rhs: tensor<128x64xf32>
  ) -> tensor<1x64xf32> {
    %mat = tosa.matmul %lhs, %rhs : (tensor<1x128xf32>, tensor<128x64xf32>) -> tensor<1x64xf32>
    %bias = tosa.const dense<0.0> : tensor<1x64xf32>
    %biased = tosa.add %mat, %bias : tensor<1x64xf32>
    %activated = tosa.relu %biased : tensor<1x64xf32>
    return %activated : tensor<1x64xf32>
  }
}