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

def get_polygon( polygon ):

    if polygon == 'square':

        return square( 2.2, 2.2 )
    
    elif polygon == 'triangle':

        return triangle( 2.2, 2.2 )
    
    elif polygon == 'trapezoid':

        return trapezoid( 2.2, 2.2 )