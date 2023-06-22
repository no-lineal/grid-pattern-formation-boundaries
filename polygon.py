
import numpy as np
import math
from shapely.geometry import Polygon, Point

def triangle( width, height ):

    triangle = Polygon(
        [
            ( -width/2, -height/2 ), 
            (0, width/2), 
            ( width/2, -height/2 )
        ]
    )
    
    return triangle

def annulus(outer_radius, inner_radius, num_points=100):

    center = Point(0, 0)

    outer_circle = center.buffer(outer_radius)
    inner_circle = center.buffer(inner_radius)

    annulus = outer_circle.difference(inner_circle)

    return annulus

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

def get_polygon( polygon ):

    if polygon == 'triangle':

        return triangle(2.2, 2.2)
    
    elif polygon == 'annulus':

        return annulus(2.2, 1.1)
    
    elif polygon == 'cube':

        return cube( 2.2 )

    else:

        raise ValueError('Polygon not recognized.')