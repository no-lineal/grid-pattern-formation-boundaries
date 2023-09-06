import numpy as np
import math
from shapely.geometry import Polygon, Point, LineString

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

def triangle( area, height ):

    base = ( 2 * area ) / height

    vertices =[
        (0, height/2), 
        (-base/2, -height/2),
        (base/2, -height/2)
    ]

    triangle = Polygon( vertices )

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

    circle = center.buffer( radius )

    return circle

def pentagon( area ):

    side_lenght = np.sqrt( ( 4 * area ) / ( 5 * np.tan( np.pi / 5 ) ) )

    vertices = []

    for i in range( 5 ):

        x = side_lenght * np.cos( 2 * np.pi * i / 5 )
        y = side_lenght * np.sin( 2 * np.pi * i / 5 )

        vertices.append( (x, y) )

    pentagon = Polygon( vertices )

    return pentagon

def get_polygon( polygon ):

    if polygon == 'square':

        return square( 2.2, 2.2 )
    
    elif polygon == 'rectangle':

        return square( 4.4, 1.1 )
    
    elif polygon == 'triangle':

        return triangle( (2.2 ** 2), 2.2 )
    
    elif polygon == 'trapezoid':

        return trapezoid( 2.2, 2.2 )
    
    elif polygon == 'hall_square':

        return hall_square( 2.2, 2.2 )
    
    elif polygon == 'circle':

        return circle( np.sqrt( 4.84 / np.pi ) )
    
    elif polygon == 'pentagon':

        return pentagon( 2.2 ** 2 )
    
    else:

        raise ValueError( 'Invalid polygon.' )