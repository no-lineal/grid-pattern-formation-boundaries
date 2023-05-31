import numpy as np
import torch
import os
from shapely.geometry import Point, LineString

class TrajectoryGenerator( object ):

    def __init__(self, options, place_cells, polygon):

        self.device = options.device
        self.sequence_length = options.sequence_length
        self.periodic = options.periodic
        self.batch_size = options.batch_size

        self.place_cells = place_cells
        self.polygon = polygon

        self.border_region = 0.03 # meters

    def avoid_wall(self, position, head_direction):

        """
        
            compute the distance to the nearest wall

        """

        if not isinstance(self.polygon, np.ndarray):

            points = [ Point(x, y) for x, y in position ]
            
            dists = []
            for p in points:

                d_seg = []
                for seg_start, seg_end in zip( self.polygon.exterior.coords[:-1], self.polygon.exterior.coords[1:] ):
                    
                    segment = LineString([seg_start, seg_end])
                    distance = p.distance(segment)

                    d_seg.append( distance )
                
                dists.append( d_seg )

            # compute distance to nearest wall
            dists = np.array( dists)
            d_wall = np.min( dists, axis=1 )

            num_vertices = len( self.polygon.exterior.coords )

            angles = np.arange( num_vertices - 1 ) * np.pi / 2
            theta = angles[ np.argmin( dists, axis=1 ) ]

            head_direction = np.mod( head_direction, 2 * np.pi )
            a_wall = head_direction - theta
            a_wall = np.mod( a_wall + np.pi, 2 * np.pi ) - np.pi

            is_near_wall = ( d_wall < self.border_region ) * ( np.abs( a_wall ) < np.pi / 2 )
            turn_angle = np.zeros_like( head_direction )
            turn_angle[ is_near_wall ] = np.sign( a_wall[ is_near_wall ] ) * ( np.pi / 2 - np.abs( a_wall[ is_near_wall ] ) )

        elif isinstance(self.polygon, np.ndarray):

            x = position[:, 0]
            y = position[:, 1]
            z = position[:, 2]

        return is_near_wall, turn_angle

    def generate_trajectory( self, batch_size ):

        """
        
            generate a random walk trajectory
        
        """

        samples = self.sequence_length # steps in trajectory

        dt = 0.02 # time step increment (seconds)
        sigma = 5.76 * 2 # standard deviation of rotation velocity (rads / second)
        b = 0.13 * 2 * np.pi # forward velocity rayleigh distribution scale (m/sec)
        mu = 0.0 # turn angle bias

        if not isinstance(self.polygon, np.ndarray):

            min_x, min_y, max_x, max_y = self.polygon.bounds

            width = max_x - min_x
            height = max_y - min_y

            # initialize variables
            position = np.zeros( [ batch_size, samples + 2, 2 ] ) # batch, steps, (x,y)
            head_direction = np.zeros( [ batch_size, samples + 2 ] ) # batch, steps

            start_points = []
            while len(start_points) < batch_size:
                
                x = np.random.uniform(-width/2, width/2)
                y = np.random.uniform(-height/2, height/2)

                point = Point( x, y )

                if self.polygon.contains( point ):

                    start_points.append( point )

            position[:, 0, 0] = np.array( [ point.x for point in start_points ] )
            position[:, 0, 1] = np.array( [ point.y for point in start_points ] )

            head_direction[:, 0] = np.random.uniform(0, 2 * np.pi, batch_size) # radians

            velocity = np.zeros( [ batch_size, samples + 2 ] ) # batch, steps

            # generate a sequence of random boosts and turns
            random_turn = np.random.normal( mu, sigma, [ batch_size, samples + 1 ] )
            random_vel = np.random.rayleigh( b, [ batch_size, samples + 1 ] )

            v = np.abs( np.random.normal( 0, b * np.pi / 2, batch_size ) )

            for t in range( samples + 1 ):

                # update velocity
                v = random_vel[:, t]
                turn_angle = np.zeros( batch_size )

                # not false == true
                if not self.periodic:

                    # if in border region, turn and slow down
                    is_near_wall, turn_angle = self.avoid_wall( position[:, t], head_direction[:, t] )
                    v[ is_near_wall ] *= 0.25

                # update turn angle
                turn_angle += dt * random_turn[:, t]

                # take a step
                velocity[:, t] = v * dt
                update = velocity[:, t, None] * np.stack([np.cos(head_direction[:, t]), np.sin(head_direction[:, t])], axis=-1)
                position[:, t + 1] = position[:, t] + update

                # rotate head direction
                head_direction[:, t + 1] = head_direction[:, t] + turn_angle

            head_direction = np.mod(head_direction + np.pi, 2 * np.pi) - np.pi  # Periodic variable

            trajectory = {}

            # input variables
            trajectory['init_hd'] = head_direction[:, 0, None]
            trajectory['init_x'] = position[:, 1, 0, None]
            trajectory['init_y'] = position[:, 1, 1, None]
            trajectory['ego_v'] = velocity[:, 1:-1 ]
            ang_v = np.diff( head_direction, axis=-1 )
            trajectory['phi_x'], trajectory['phi_y'] = np.cos(ang_v)[:, :-1], np.sin(ang_v)[:, :-1]

            # target variables
            trajectory['target_hd'] = head_direction[:, 1:-1]
            trajectory['target_x'] = position[:, 2:, 0]
            trajectory['target_y'] = position[:, 2:, 1]

        return trajectory
    
    def get_test_batch( self, batch_size=None ):

        """

        test generator, return a batch of sample trajectories
        
        """

        if not batch_size:

            batch_size = self.batch_size

        trajectory = self.generate_trajectory( batch_size )

        # velocity
        v = np.stack(
            [
                trajectory['ego_v'] * np.cos(trajectory['target_hd']), 
                trajectory['ego_v'] * np.sin(trajectory['target_hd'])
            ], 
            axis=-1
        )

        v = torch.tensor(v, dtype=torch.float32).transpose(0, 1)

        # position
        pos = np.stack( [ trajectory['target_x'], trajectory['target_y'] ], axis=-1 )
        pos = torch.tensor( pos, dtype=torch.float32 ).transpose(0, 1)
        pos = pos.to( self.device )

        # activation
        place_outputs = self.place_cells.get_activation(pos)

        # initial position
        init_pos = np.stack( [ trajectory['init_x'], trajectory['init_y'] ] , axis=-1 )
        init_pos = torch.tensor(init_pos, dtype=torch.float32)
        init_pos = init_pos.to( self.device )

        # initial activation
        init_actv = self.place_cells.get_activation(init_pos).squeeze()

        v = v.to( self.device )
        inputs = (v, init_actv)

        return (inputs, pos, place_outputs)
    
    def get_generator( self, batch_size=None ):

        """
        
            return a generator that yields batches of trajectories
        
        """

        if not batch_size:

            batch_size = self.batch_size

        n = 0
        while True:

            trajectory = self.generate_trajectory( batch_size )

            # velocity
            v = np.stack(
                [
                    trajectory['ego_v'] * np.cos(trajectory['target_hd']), 
                    trajectory['ego_v'] * np.sin(trajectory['target_hd'])
                ], 
                axis=-1
            )

            v = torch.tensor(v, dtype=torch.float32).transpose(0, 1)

            # position
            pos = np.stack( [ trajectory['target_x'], trajectory['target_y'] ], axis=-1 )
            pos = torch.tensor( pos, dtype=torch.float32 ).transpose(0, 1)
            pos = pos.to( self.device )

            # activation
            place_outputs = self.place_cells.get_activation(pos)

            # initial position
            init_pos = np.stack( [ trajectory['init_x'], trajectory['init_y'] ] , axis=-1 )
            init_pos = torch.tensor(init_pos, dtype=torch.float32)
            init_pos = init_pos.to( self.device )

            # initial activation
            init_actv = self.place_cells.get_activation(init_pos).squeeze()

            v = v.to( self.device )
            inputs = (v, init_actv)

            n += 1

            yield (inputs, place_outputs, pos)