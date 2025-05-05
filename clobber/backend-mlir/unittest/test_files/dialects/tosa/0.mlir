module {
  func.func private @print_i32(tensor<i32>) -> ()

  func.func @main(%a: tensor<i32>, %b: tensor<i32>) -> () {
    %sum = tosa.add %a, %b : tensor<i32>
    func.call @print_i32(%sum) : (tensor<i32>) -> ()
    return
  }
}