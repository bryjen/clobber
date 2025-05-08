module {
  // Slice a sub-tensor and apply sigmoid
  func.func @main() -> tensor<1x2xf32> {
    %input = tosa.const dense<[[0.0, 1.0, 2.0]]> : tensor<1x3xf32>
    %out = tosa.slice %input {start = [0, 1], size = [1, 2], stride = [1, 1]} : tensor<1x3xf32> to tensor<1x2xf32>
    %sig = tosa.sigmoid %out : tensor<1x2xf32>
    return %sig : tensor<1x2xf32>
  }
}