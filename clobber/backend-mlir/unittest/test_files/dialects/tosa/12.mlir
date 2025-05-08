module {
  // Use a LUT to transform input values
  func.func @main() -> tensor<4xi32> {
    %input = tosa.const dense<[0, 1, 2, 3]> : tensor<4xi32>
    %lut = tosa.const dense<[10, 20, 30, 40, 50, 60, 70, 80]> : tensor<256xi32> // padded
    %out = tosa.table %input, %lut : (tensor<4xi32>, tensor<256xi32>) -> tensor<4xi32>
    return %out : tensor<4xi32>
  }
}