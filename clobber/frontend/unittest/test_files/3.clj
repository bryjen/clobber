(def main [x y]
  (let* ((z (+ x y))
         (z2 (relu z))
         (out (matmul z2 weights)))
        out))