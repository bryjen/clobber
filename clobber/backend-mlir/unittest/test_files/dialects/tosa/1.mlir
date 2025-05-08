module {
  // main returns a tensor with a single float value
  func.func @main() -> tensor<1xf32> {
    %cst = tosa.const dense<3.14> : tensor<1xf32>
    return %cst : tensor<1xf32>
  }
}