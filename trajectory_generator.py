# -*- coding: utf-8 -*-
import torch
import os
import numpy as np

from shapely.geometry import Point, LineString

# viz
from matplotlib import pyplot as plt


class TrajectoryGenerator(object):

    def __init__(self, options, place_cells, polygon):

        self.options = options
        self.place_cells = place_cells
        self.polygon = polygon

    def iterate_polygon_sides(self, vertices):

        num_vertices = len( vertices )

        print( f'num_vertices: {num_vertices}')

        for i in range( num_vertices ):

            current_vertex = vertices[i]
            next_vertex = vertices[ (i + 1) % num_vertices ]

            yield current_vertex, next_vertex

    def avoid_wall(self, position, hd):

        """

        Compute distance and angle to nearest wall

        """ 

        points = [ ( x, y ) for x, y in zip( position[:,0], position[:,1] ) ] 
        sides = list( self.polygon.exterior.coords )

        if len( sides ) != 100:
            sides = sides[ ::-1 ] 
            idx = 1
            sides = sides[ idx :  ] + sides[ 1 : idx + 1 ]

        dists = []
        for p in points:

            dist = []
            for s in range( len( sides )  - 1 ):

                seg_start = sides[ s ]
                seg_end = sides[ s+1 ]

                if seg_start[0] == seg_end[0]:

                    d = abs( p[0] - seg_start[0]) 

                else:

                    m = ( seg_end[1] - seg_start[1] ) / ( seg_end[0] - seg_start[0] )
                    b = seg_start[1] - m * seg_start[0]

                    d = np.abs( m * p[0] - p[1] + b ) / np.sqrt( m**2 + 1 )

                dist.append( d )

            dists.append( dist )

        dists = [ np.array(x) for x in dists ]
        dists = np.stack( dists, axis=1 )
        dists = list( dists )

        d_wall = np.min(dists, axis=0)

        slides = len( dists )
        angles = np.arange( slides ) * ( 2 * np.pi / slides )
        theta = angles[ np.argmin(dists, axis=0) ]

        hd = np.mod(hd, 2 * np.pi)
        a_wall = hd - theta
        a_wall = np.mod(a_wall + np.pi, 2 * np.pi) - np.pi

        is_near_wall = (d_wall < self.border_region) * (np.abs(a_wall) < np.pi / 2)
        turn_angle = np.zeros_like(hd)
        turn_angle[is_near_wall] = np.sign(a_wall[is_near_wall]) * (np.pi / 2 - np.abs(a_wall[is_near_wall]))

        return is_near_wall, turn_angle

    def generate_trajectory(self, batch_size):

        """

        Generate a random walk in a rectangular box

        """

        samples = self.options.sequence_length # steps in trajectory

        dt = 0.02  # time step increment (seconds)
        sigma = 5.76 * 2  # stdev rotation velocity (rads/sec)
        b = 0.13 * 2 * np.pi  # forward velocity rayleigh dist scale (m/sec)
        mu = 0  # turn angle bias 

        self.border_region = 0.03  # meters

        # initialize variables
        position = np.zeros( [ batch_size, samples + 2, 2 ] ) # batch, steps, (x,y)
        head_direction = np.zeros( [ batch_size, samples + 2 ] ) # batch, steps


        # validate starting points

        min_x, min_y, max_x, max_y = self.polygon.bounds
        width = max_x - min_x
        height = max_y - min_y

        start_points = []
        while len(start_points) < batch_size:
                
            x = np.random.uniform(-width/2, width/2)
            y = np.random.uniform(-height/2, height/2)

            point = Point( x, y )

            if self.polygon.contains( point ):

                start_points.append( point )
        
        position[:, 0, 0] = np.array( [ point.x for point in start_points ] )
        position[:, 0, 1] = np.array( [ point.y for point in start_points ] )
        
        head_direction[:, 0] = np.random.uniform(0, 2 * np.pi, batch_size)
        
        velocity = np.zeros([batch_size, samples + 2])

        # Generate sequence of random boosts and turns
        random_turn = np.random.normal(mu, sigma, [batch_size, samples + 1])
        random_vel = np.random.rayleigh(b, [batch_size, samples + 1])
        v = np.abs(np.random.normal(0, b * np.pi / 2, batch_size)) # velocity

        for t in range(samples + 1):

            # Update velocity
            v = random_vel[:, t]
            turn_angle = np.zeros(batch_size)

            # not False = True
            if not self.options.periodic:

                # If in border region, turn and slow down
                is_near_wall, turn_angle = self.avoid_wall(position[:, t], head_direction[:, t])
                v[is_near_wall] *= 0.25

            # Update turn angle
            turn_angle += dt * random_turn[:, t]

            # Take a step
            velocity[:, t] = v * dt
            update = velocity[:, t, None] * np.stack([np.cos(head_direction[:, t]), np.sin(head_direction[:, t])], axis=-1)
            position[:, t + 1] = position[:, t] + update

            # Rotate head direction
            head_direction[:, t + 1] = head_direction[:, t] + turn_angle

        # Periodic boundaries
        #if self.options.periodic:

        #    position[:, :, 0] = np.mod(position[:, :, 0] + box_width / 2, box_width) - box_width / 2
        #    position[:, :, 1] = np.mod(position[:, :, 1] + box_height / 2, box_height) - box_height / 2

        head_direction = np.mod(head_direction + np.pi, 2 * np.pi) - np.pi  # Periodic variable

        traj = {}
        # Input variables
        traj['init_hd'] = head_direction[:, 0, None]
        traj['init_x'] = position[:, 1, 0, None]
        traj['init_y'] = position[:, 1, 1, None]

        traj['ego_v'] = velocity[:, 1:-1]
        ang_v = np.diff(head_direction, axis=-1)
        traj['phi_x'], traj['phi_y'] = np.cos(ang_v)[:, :-1], np.sin(ang_v)[:, :-1]

        # Target variables
        traj['target_hd'] = head_direction[:, 1:-1]
        traj['target_x'] = position[:, 2:, 0]
        traj['target_y'] = position[:, 2:, 1]

        return traj

    def get_generator(self, batch_size=None):

        """

        Returns a generator that yields batches of trajectories
        
        """

        if not batch_size:

            batch_size = self.options.batch_size

        n = 0
        while True:

            traj = self.generate_trajectory(batch_size)

            v = np.stack([traj['ego_v'] * np.cos(traj['target_hd']),
                          traj['ego_v'] * np.sin(traj['target_hd'])], axis=-1)
            v = torch.tensor(v, dtype=torch.float32).transpose(0, 1)

            pos = np.stack([traj['target_x'], traj['target_y']], axis=-1)
            pos = torch.tensor(pos, dtype=torch.float32).transpose(0, 1)
            # Put on GPU if GPU is available
            pos = pos.to(self.options.device)
            place_outputs = self.place_cells.get_activation(pos)

            init_pos = np.stack([traj['init_x'], traj['init_y']], axis=-1)
            init_pos = torch.tensor(init_pos, dtype=torch.float32)
            init_pos = init_pos.to(self.options.device)
            init_actv = self.place_cells.get_activation(init_pos).squeeze()

            v = v.to(self.options.device)
            inputs = (v, init_actv)

            # viz
            #us = self.place_cells.us

            #plt.figure(figsize=(5,5))
            #plt.scatter(us.cpu()[:,0], us.cpu()[:,1], c='lightgrey', label='Place cell centers')

            #for i in range(batch_size):
            #    plt.plot(pos.cpu()[:,i,0],pos.cpu()[:,i,1], label='Simulated trajectory', c='C1')
                
            #    if i==0:
            #        plt.legend()

            #plt.savefig( self.options.save_dir + 'trajectories/' + 'trajectories_' + str(n) + '.png' )
            #

            n += 1

            yield (inputs, place_outputs, pos)

    def get_test_batch(self, batch_size=None, box_width=None, box_height=None):

        """
        For testing performance, returns a batch of smample trajectories
        """

        if not batch_size:

            batch_size = self.options.batch_size

        traj = self.generate_trajectory(batch_size)

        # velocity
        v = np.stack([traj['ego_v'] * np.cos(traj['target_hd']),
                      traj['ego_v'] * np.sin(traj['target_hd'])], axis=-1)

        v = torch.tensor(v, dtype=torch.float32).transpose(0, 1)

        # position
        pos = np.stack([traj['target_x'], traj['target_y']], axis=-1)
        pos = torch.tensor(pos, dtype=torch.float32).transpose(0, 1)
        pos = pos.to(self.options.device)

        # activation
        place_outputs = self.place_cells.get_activation(pos)

        # initial position
        init_pos = np.stack( [ traj['init_x'], traj['init_y'] ] , axis=-1 )
        init_pos = torch.tensor(init_pos, dtype=torch.float32)
        init_pos = init_pos.to(self.options.device)

        # initial activation
        init_actv = self.place_cells.get_activation(init_pos).squeeze()

        v = v.to(self.options.device)
        inputs = (v, init_actv)

        return (inputs, pos, place_outputs)