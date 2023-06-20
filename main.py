import torch

from polygon import get_polygon

from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator

from model import RNN
from trainer import Trainer

import json

import sys
import os
import argparse

import time

# viz
from matplotlib import pyplot as plt

def generate_options( parameters ):

    # directories

    load_path = os.getcwd() + '/pre_computed/'
    save_path = os.getcwd() + '/output/'

    parser = argparse.ArgumentParser()

    global_parameters = parameters.keys()

    for p in global_parameters:
        try:
            for k, v in parameters[p].items():

                parser.add_argument(
                    '--' + k,
                    default=v,
                    help=f'{k} parameter'
                )
        except:
            parser.add_argument(
                '--' + p,
                default=parameters[p],
                help=f'{p} parameter'
            )

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

    return parser.parse_args()

def plot_place_cells( place_cells, polygon, options ):

    # place cell centers
    us = place_cells.us

    if us.shape[1] == 2:

        plt.figure(figsize=(5,5))

        try:

            exterior_coords = polygon.exterior.xy
            interior_coords = [interior_ring.coords.xy for interior_ring in polygon.interiors]

            plt.plot(*exterior_coords, color='blue', alpha=0.5, label='Exterior')
            for interior_coords_ring in interior_coords:
                plt.plot(*interior_coords_ring, color='red', alpha=0.5, label='Interior')

        except:

            exterior_coords = polygon.exterior.xy
            plt.fill(*exterior_coords, color='blue', alpha=0.5, label='Exterior')

        plt.scatter( us.cpu()[:,0], us.cpu()[:,1], c='lightgrey', label='Place cell centers' )
        plt.savefig( options.save_path + options.model_name + 'place_cells.png' )

    else:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter( us.cpu()[:,0], us.cpu()[:,1], us.cpu()[:,2], c='lightgrey', label='Place cell centers' )

        plt.savefig( options.save_path + options.model_name + 'place_cells.png' )

def plot_trajectory( place_cells, trajectory_generator, polygon, options ):

    # place cell centers
    us = place_cells.us

    if us.shape[1] == 2:
        
        inputs, pos, pc_outputs = trajectory_generator.get_test_batch()
        pos = pos.cpu()

        plt.figure(figsize=(5,5))
        plt.scatter(us.cpu()[:,0], us.cpu()[:,1], c='lightgrey', label='Place cell centers')

        try:

            exterior_coords = polygon.exterior.xy
            interior_coords = [interior_ring.coords.xy for interior_ring in polygon.interiors]

            plt.plot(*exterior_coords, color='blue', alpha=0.5, label='Exterior')
            for interior_coords_ring in interior_coords:
                plt.plot(*interior_coords_ring, color='red', alpha=0.5, label='Interior')

        except:

            exterior_coords = polygon.exterior.xy
            plt.fill(*exterior_coords, color='blue', alpha=0.5, label='Exterior')

        for i in range( options.batch_size ):
            plt.plot(pos.cpu()[:,i,0], pos.cpu()[:,i,1], label='Simulated trajectory', c='C1')
            if i==0:
                plt.legend()

        plt.savefig( options.save_path + options.model_name + 'trajectory.png' )

if __name__ == '__main__':

    log = {}

    test_trajectory = True

    # retrieve the path to the JSON file
    json_file = './experiments/annulus.json'

    # load JSON file
    with open( json_file ) as f:
        parameters = json.load( f )

    # update the log
    log.update( {'parameters': parameters} )

    # generate options
    options = generate_options( parameters )

    # create save directory
    try:
        os.mkdir( options.save_path + options.model_name )
    except:
        pass

    # get polygon
    polygon = get_polygon( options.shape )

    # place cells
    place_cells = PlaceCells( options, polygon)

    # plot place
    plot_place_cells( place_cells, polygon, options )

    # trajectory simmulation
    trajectory_generator = TrajectoryGenerator( options, place_cells, polygon )

    # plot test trajectory
    if test_trajectory:

        plot_trajectory( place_cells, trajectory_generator, polygon, options )

    # model
    model = RNN( options, place_cells )
    model = model.to( options.device )

    # log update
    log.update( {'model': model} )

    # train
    trainer = Trainer( options, model, trajectory_generator, polygon )

    tic = time.perf_counter()
    trainer.train( n_epochs=options.n_epochs, n_steps=options.n_steps )
    toc = time.perf_counter()

    # log update
    log.update( { 'training_time': (toc - tic) / 60 } )