import torch

import boundaries

from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator

from model import RNN
from trainer import Trainer

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

parser.add_argument(
    '--model_name',
    default='100_1000_trapezoid/',
    help='name of model'
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

# grid cell 

parser.add_argument(
    '--Ng',
    default=4096,
    help='number of grid cells'
)

# model

parser.add_argument(
    '--RNN_type', 
    default='RNN', 
    help='RNN or LSTM'
)

parser.add_argument(
    '--weight_decay', 
    default=1e-4, 
    help='strength of weight decay on recurrent weights'
)

parser.add_argument(
    '--activation', 
    default='relu', 
    help='recurrent nonlinearity'
)

parser.add_argument(
    '--learning_rate',
    default=1e-4,
    help='gradient descent learning rate'
)

parser.add_argument(
    '--n_epochs',
    default=100,
    help='number of training epochs'
)

parser.add_argument(
    '--n_steps',
    default=1000,
    help='batches per epoch'
)

options = parser.parse_args()

try:
    os.mkdir( save_path + options.model_name )
except:
    pass

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
    #polygon = boundaries.donut( 2.2, 2.2 )
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
        plt.savefig( options.save_path + options.model_name + 'place_cells.png' )

    else:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter( us.cpu()[:,0], us.cpu()[:,1], us.cpu()[:,2], c='lightgrey', label='Place cell centers' )

        plt.savefig( options.save_path + options.model_name + 'place_cells.png' )

    # trajectory simmulation

    trajectory_generator = TrajectoryGenerator( options, place_cells, polygon )

    test_trajectory = True

    if test_trajectory:

        inputs, pos, pc_outputs = trajectory_generator.get_test_batch()
        pos = pos.cpu()

        plt.figure(figsize=(5,5))
        plt.scatter(us.cpu()[:,0], us.cpu()[:,1], c='lightgrey', label='Place cell centers')
        for i in range( options.batch_size ):
            plt.plot(pos.cpu()[:,i,0], pos.cpu()[:,i,1], label='Simulated trajectory', c='C1')
            if i==0:
                plt.legend()

        plt.savefig( options.save_path + options.model_name + 'trajectory.png' )

    # model
    if options.RNN_type == 'RNN':

        model = RNN( options, place_cells )
        model = model.to( options.device )

        print('\n')
        print(f'model parameters: ')
        print(model)
        print('\n')

    else:

        pass

    # training
    trainer = Trainer( options, model, trajectory_generator, polygon )
    trainer.train( n_epochs=options.n_epochs, n_steps=options.n_steps )