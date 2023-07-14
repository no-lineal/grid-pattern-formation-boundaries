import numpy as np
import math
from shapely.geometry import Polygon, Point

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

def triangle( width, height ):

    triangle = Polygon(
        [
            ( -width/2, -height/2 ), 
            (0, width/2), 
            ( width/2, -height/2 )
        ]
    )
    
    return triangle

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

def hall_square( box_width, box_height ):

    sh = np.round( box_width / 3, 2 )
    sw = (box_width - sh) / 2

    a = sh * sw 
    x = a / box_width

    hall_square = Polygon(
        [
            (-box_width/2 - x, -box_height/2), 
            (-box_width/2 - x, box_height/2), 
            (-sh/2, box_height/2), 
            (-sh/2, box_height/2 - sw), 
            (sh/2, box_height/2 - sw),
            (sh/2, box_height/2), 
            (box_width/2 + x, box_height/2),
            (box_width/2 + x, -box_height/2), 
            (sh/2, -box_height/2),
            (sh/2, -box_height/2 + sw),
            (-sh/2, -box_height/2 + sw),
            (-sh/2, -box_height/2)
        ]
    )
    
    return hall_square

def circle( radius ):

    center = Point(0, 0)

    circle = center.buffer( radius * 2 )

    return circle

def get_polygon( polygon ):

    if polygon == 'square':

        return square( 2.2, 2.2 )
    
    elif polygon == 'triangle':

        return triangle( 2.2, 2.2 )
    
    elif polygon == 'trapezoid':

        return trapezoid( 2.2, 2.2 )
    
    elif polygon == 'hall_square':

        return hall_square( 4.4, 4.4 )
    
    elif polygon == 'circle':

        return circle( 1.1 )
    
    else:

        raise ValueError( 'Invalid polygon.' )