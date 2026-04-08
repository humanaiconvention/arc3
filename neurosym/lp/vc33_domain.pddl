(define (domain vc33-induced)
  (:requirements :strips :negative-preconditions)
  (:predicates
    (p1 ?x1 ?x2)
    (p2 ?x1 ?x2)
    (p3 ?x1 ?x2)
    (p4 ?x1 ?x2)
    (p5 ?x1 ?x2)
  )
  (:action a6_x0y0
    :parameters (?o1 ?o2)
    :precondition (and
    )
    :effect (and
      (p2 ?o2 ?o1)
      (p2 ?o1 ?o2)
    )
  )
  (:action a6_x0y1
    :parameters (?o1 ?o2)
    :precondition (and
      (p1 ?o2 ?o1)
      (p2 ?o1 ?o1)
      (p2 ?o2 ?o1)
      (p2 ?o1 ?o2)
    )
    :effect (and
      (not (p1 ?o2 ?o1))
      (not (p2 ?o1 ?o1))
      (not (p2 ?o1 ?o2))
      (p2 ?o2 ?o2)
      (p3 ?o2 ?o1)
    )
  )
  (:action a6_x0y2
    :parameters (?o1 ?o2)
    :precondition (and
      (p1 ?o2 ?o2)
    )
    :effect (and
      (not (p1 ?o2 ?o2))
      (p1 ?o1 ?o1)
      (p3 ?o2 ?o1)
    )
  )
  (:action a6_x1y2
    :parameters (?o1 ?o2)
    :precondition (and
      (p1 ?o2 ?o2)
      (p3 ?o2 ?o1)
      (p3 ?o1 ?o2)
      (p3 ?o2 ?o2)
    )
    :effect (and
      (not (p1 ?o2 ?o2))
      (not (p3 ?o2 ?o1))
      (not (p3 ?o1 ?o2))
      (p2 ?o2 ?o1)
      (p2 ?o1 ?o2)
      (p2 ?o2 ?o2)
    )
  )
  (:action a6_x1y3
    :parameters (?o1 ?o2)
    :precondition (and
      (p1 ?o1 ?o1)
      (p5 ?o1 ?o2)
    )
    :effect (and
      (p3 ?o1 ?o1)
      (p3 ?o2 ?o1)
      (p3 ?o1 ?o2)
      (p3 ?o2 ?o2)
    )
  )
  (:action a6_x2y0
    :parameters (?o1 ?o2)
    :precondition (and
      (p2 ?o2 ?o2)
      (p5 ?o1 ?o1)
      (p5 ?o2 ?o2)
    )
    :effect (and
      (not (p2 ?o2 ?o2))
      (p3 ?o1 ?o1)
      (p3 ?o2 ?o1)
      (p3 ?o1 ?o2)
      (p3 ?o2 ?o2)
    )
  )
  (:action a6_x2y1
    :parameters (?o1 ?o2)
    :precondition (and
      (p1 ?o2 ?o1)
      (p4 ?o1 ?o1)
    )
    :effect (and
    )
  )
  (:action a6_x2y2
    :parameters (?o1 ?o2)
    :precondition (and
      (p1 ?o2 ?o1)
      (p2 ?o1 ?o1)
      (p2 ?o2 ?o1)
      (p2 ?o2 ?o2)
      (p3 ?o1 ?o2)
    )
    :effect (and
      (not (p1 ?o2 ?o1))
      (not (p2 ?o1 ?o1))
      (not (p2 ?o2 ?o1))
      (not (p2 ?o2 ?o2))
      (not (p3 ?o1 ?o2))
      (p1 ?o1 ?o2)
      (p3 ?o1 ?o1)
      (p3 ?o2 ?o1)
    )
  )
  (:action a6_x2y3
    :parameters (?o1 ?o2)
    :precondition (and
      (p2 ?o2 ?o1)
      (p3 ?o1 ?o1)
    )
    :effect (and
      (not (p2 ?o2 ?o1))
      (not (p3 ?o1 ?o1))
      (p1 ?o1 ?o2)
    )
  )
  (:action a6_x3y2
    :parameters (?o1 ?o2)
    :precondition (and
      (p1 ?o2 ?o2)
      (p3 ?o1 ?o1)
    )
    :effect (and
      (not (p1 ?o2 ?o2))
      (not (p3 ?o1 ?o1))
      (p1 ?o1 ?o1)
      (p1 ?o1 ?o2)
    )
  )
  (:action a6_x3y3
    :parameters (?o1 ?o2)
    :precondition (and
      (p3 ?o2 ?o1)
      (p3 ?o2 ?o2)
    )
    :effect (and
      (not (p3 ?o2 ?o1))
      (not (p3 ?o2 ?o2))
      (p1 ?o1 ?o1)
      (p1 ?o2 ?o2)
      (p2 ?o2 ?o1)
      (p2 ?o2 ?o2)
    )
  )
)
