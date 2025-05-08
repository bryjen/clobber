// (def main [x y]
//   (let* (
//     (z (+ x y))
//     (z2 (relu z))
//     (out (matmul z2 weights))
//   )
//   out))

module {
  func.func @main(%x: tensor<1x128xf32>, %y: tensor<1x128xf32>) -> tensor<1x64xf32> {
    %z = tosa.add %x, %y : tensor<1x128xf32>
    %z2 = tosa.relu %z : tensor<1x128xf32>
    %weights = tosa.const dense<...> : tensor<128x64xf32>
    %out = tosa.matmul %z2, %weights : (tensor<1x128xf32>, tensor<128x64xf32>) -> tensor<1x64xf32>
    return %out : tensor<1x64xf32>
  }
}