(define (problem vc33-plan)
  (:domain vc33-induced)
  (:objects obj1 obj2)
  (:init
    (p4 obj1 obj2)
    (p4 obj2 obj1)
    (p4 obj2 obj2)
    (p5 obj1 obj1)
  )
  (:goal (and
    (p1 obj1 obj1)
    (p1 obj2 obj2)
  ))
)
