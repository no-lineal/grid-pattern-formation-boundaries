import numpy as np
import math
from shapely.geometry import Polygon

def square( box_width, box_height ):

    square = Polygon(
        [ 
            (- box_width/2, - box_height/2), 
            (- box_width/2, box_height/2), 
            (box_width/2, box_height/2),
            (box_width/2, - box_height/2) 
        ] 
    )
    
    return square

def trapezoid( box_width, box_height ):

    trapezoid = Polygon(
        [
            (-box_width/2, -box_height/2), 
            (-box_width/2 + box_width/4, box_height/2), 
            (box_width/2 - box_width/4, box_height/2),
            (box_width/2, -box_height/2)
        ]
    )
    
    return trapezoid

def circle( box_width, box_height, resolution=100 ):

    points = []
    for i in range(resolution):

        angle = (2 * math.pi * i) / resolution

        x = 0 + box_width/2 * math.cos(angle)
        y = 0 + box_height/2 * math.sin(angle)

        points.append((x, y))

    circle = Polygon(points)

    return circle

def cube( side_length ):

    center = np.array([0, 0, 0])

    half_side = side_length / 2.0

    vertices = np.array(
        [
            [ -half_side, -half_side, -half_side ], 
            [ -half_side, -half_side, half_side ],
            [ -half_side,  half_side, -half_side ],
            [ -half_side,  half_side, half_side ],
            [ half_side, -half_side,  -half_side ],
            [ half_side, -half_side,  half_side ],
            [ half_side,  half_side,  -half_side ],
            [ half_side,  half_side,  half_side ]
        ]
    )

    boundaries = center + vertices

    return boundaries