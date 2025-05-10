(fn main []
  (let []
    (let [result
          (accel [x a y b]
                 (tosa-relu (tosa-matmul x y)))]

      (print result))))