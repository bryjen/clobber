module {
  // Pad input and multiply with another tensor
  func.func @main() -> tensor<1x4xf32> {
    %input = tosa.const dense<[1.0, 2.0]> : tensor<1x2xf32>
    %pad = tosa.const dense<[[0,0],[1,1]]> : tensor<2x2xi32> // pad 1 element on left/right of dim 1
    %padded = tosa.pad %input, %pad : (tensor<1x2xf32>, tensor<2x2xi32>) -> tensor<1x4xf32>
    %scale = tosa.const dense<[2.0, 2.0, 2.0, 2.0]> : tensor<1x4xf32>
    %out = tosa.mul %padded, %scale : tensor<1x4xf32>
    return %out : tensor<1x4xf32>
  }
}