import torch

import boundaries

from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator

import os

import argparse

# viz
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as mpl_polygon

parser = argparse.ArgumentParser()

# directories

load_path = os.getcwd() + '/pre_computed/'
save_path = os.getcwd() + '/output/'

# load and save directories

parser.add_argument(
    '--load_path',
    default=load_path,
    help='directory to load example models'
    )

parser.add_argument(
    '--save_path',
    default=save_path,
    help='directory to save trained models'
    )

# device

parser.add_argument(
    '--device',
    default='cuda' if torch.cuda.is_available() else 'cpu',
    help='device to use for training'
    )  

# precumputed

parser.add_argument(
    '--precomputed',
    default=False,
    help='use precomputed place cell centers'
)  

# place cell parameters

parser.add_argument(
    '--Np',
    default=512, 
    help='number of place cells'
)

parser.add_argument(
    '--sigma',
    default=0.12,
    help='width of place cell center tuning curve (m)'
)

parser.add_argument(
    '--surround_scale',
    default=2.0,
    help='if DoG, ratio of sigma2^2 to sigma1^2'
)

parser.add_argument(
    '--periodic',
    default=False,
    help='trajectories with periodic boundary conditions'
)

parser.add_argument(
    '--DoG',
    default=True,
    help='use difference of gaussians for place cell tuning curves'
)

# trajectory parameters

parser.add_argument(
    '--sequence_length',
    default=20,
    help='number of steps in trajectory'
)

parser.add_argument(
    '--batch_size',
    default=200,
    help='number of trajectories per batch'
)

options = parser.parse_args()

if __name__ == '__main__':
    
    print('\n')
    print(f'load_dir: { options.load_path }')
    print(f'save_dir: { options.save_path }')
    print('\n')

    print(f'device: { options.device }')
    print('\n')

    # generate bounds
    #polygon = boundaries.square( 2.2, 2.2 )
    polygon = boundaries.trapezoid( 2.4, 2.4 )
    #polygon = boundaries.circle( 2.2, 2.2 )
    #polygon = boundaries.cube( 2.2 )

    # generate place cells object
    place_cells = PlaceCells( options, polygon)

    # generate place cell centers
    us = place_cells.us

    print(f'us shape: { us.shape }')
    print('\n')

    if us.shape[1] == 2:

        plt.figure(figsize=(5,5))
        plt.scatter( us.cpu()[:,0], us.cpu()[:,1], c='lightgrey', label='Place cell centers' )
        plt.savefig( options.save_path + 'place_cells.png' )

    else:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter( us.cpu()[:,0], us.cpu()[:,1], us.cpu()[:,2], c='lightgrey', label='Place cell centers' )

        plt.savefig( options.save_path + 'place_cells.png' )

    # trajectory simmulation

    trajectory_generator = TrajectoryGenerator( options, place_cells, polygon )
    inputs, pos, pc_outputs = trajectory_generator.get_test_batch()
    pos = pos.cpu()

    plt.figure(figsize=(5,5))
    plt.scatter(us[:,0], us[:,1], c='lightgrey', label='Place cell centers')
    for i in range( options.batch_size ):
        plt.plot(pos[:,i,0],pos[:,i,1], label='Simulated trajectory', c='C1')
        if i==0:
            plt.legend()

    plt.savefig( options.save_path + 'trajectory.png' )