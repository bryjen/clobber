module {
  // Reshape a 2x2 tensor to 1x4, then add a bias
  func.func @main() -> tensor<1x4xf32> {
    %input = tosa.const dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %shape = tosa.const dense<[1, 4]> : tensor<2xi32>
    %reshaped = tosa.reshape %input, %shape : (tensor<2x2xf32>, tensor<2xi32>) -> tensor<1x4xf32>
    %bias = tosa.const dense<[0.5, 0.5, 0.5, 0.5]> : tensor<1x4xf32>
    %out = tosa.add %reshaped, %bias : tensor<1x4xf32>
    return %out : tensor<1x4xf32>
  }
}