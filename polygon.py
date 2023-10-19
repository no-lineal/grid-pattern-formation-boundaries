import numpy as np
import math
from shapely.geometry import Polygon, Point, LineString

def square( area, factor = 1 ):

    area = area * factor

    box_width = np.sqrt( area )
    box_height = box_width

    square = Polygon(
        [ 
            (- box_width/2, - box_height/2), 
            (- box_width/2, box_height/2), 
            (box_width/2, box_height/2),
            (box_width/2, - box_height/2) 
        ] 
    )
    
    return square

def triangle( area, factor = 1 ):

    area = area * factor
    height = np.sqrt( area )

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

def hall_square( area, factor = 1 ):

    area = area * factor

    box_width = np.sqrt( area )
    box_height = box_width

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

def circle( area, factor ):

    area = area * factor

    radius = np.sqrt( area / np.pi )

    center = Point(0, 0)

    circle = center.buffer( radius )

    return circle

def pentagon( area, factor = 1 ):

    area = area * factor
    
    side_length = np.sqrt( (4 * area ) / ( np.sqrt( 5 * ( 5 + 2 * (np.sqrt(5)) ) ) ) )

    vertices = []
    for i in range(5):

        angle = 2 * math.pi / 5 * i  # Angle between each vertex
        x = side_length * math.cos(angle)
        y = side_length * math.sin(angle)
        vertices.append((x, y))

    pentagon = Polygon( vertices )

    return pentagon

def get_polygon( polygon ):

    if polygon == 'square':

        return square( (2.2)**2, 1 )
    
    elif polygon == 'hall_square':

        return hall_square( (2.2)**2, 1 )
    
    elif polygon == 'pentagon':

        return pentagon( 2.2**2, 1 )
    
    elif polygon == 'triangle':

        return triangle( (2.2 ** 2), 1 )
    
    elif polygon == 'circle':

        return circle( 2**2, 1 )
    
    else:

        raise ValueError( 'Invalid polygon.' )