(defn square [x]
  (* x x))

(defn sum-of-squares [a b]
  (+ (square a) (square b)))

(let [x 3
      y 4]
  (println "Result:" (sum-of-squares x y)))