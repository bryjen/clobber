module {
  // main transposes input, multiplies, and applies activation
  func.func @main(
    %a: tensor<2x3xf32>,
    %b: tensor<3x2xf32>
  ) -> tensor<2x2xf32> {
    %perm = tosa.const dense<[1, 0]> : tensor<2xi32>
    %bt = tosa.transpose %b, %perm : (tensor<3x2xf32>, tensor<2xi32>) -> tensor<2x3xf32>
    %mat = tosa.matmul %a, %bt : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x2xf32>
    %bias = tosa.const dense<[[1.0, 1.0], [1.0, 1.0]]> : tensor<2x2xf32>
    %added = tosa.add %mat, %bias : tensor<2x2xf32>
    %relu = tosa.relu %added : tensor<2x2xf32>
    return %relu : tensor<2x2xf32>
  }
}