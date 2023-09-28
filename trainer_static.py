# -*- coding: utf-8 -*-
import torch
import numpy as np

from visualize import save_ratemaps

from tqdm import tqdm

import os

torch.autograd.set_detect_anomaly(True)

class Trainer(object):

    def __init__(self, options, model, trajectory_generator, polygon, restore=True):

        self.options = options
        self.model = model
        self.trajectory_generator = trajectory_generator
        self.polygon = polygon

        lr = self.options.learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.loss = []
        self.err = []

        # Set up checkpoints

        self.ckpt_dir = os.path.join(options.save_path, options.model_name)

        ###
        #ckpt_path = os.path.join(self.ckpt_dir, 'most_recent_model.pth')

        #if restore and os.path.isdir(self.ckpt_dir) and os.path.isfile(ckpt_path):

        #    self.model.load_state_dict(torch.load(ckpt_path))
        #    print("Restored trained model from {}".format(ckpt_path))

        #else:

        #    if not os.path.isdir(self.ckpt_dir):
        ###

        os.makedirs(self.ckpt_dir, exist_ok=True)

        print("Initializing new model from scratch.")
        print("Saving to: {}".format(self.ckpt_dir))

    def train_step(self, inputs, pc_outputs, pos):

        """

        Train on one batch of trajectories.

        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.

        """

        self.model.zero_grad()

        loss, err = self.model.compute_loss(inputs, pc_outputs, pos)

        loss.backward()
        self.optimizer.step()

        return loss.item(), err.item()

    def train(self, n_epochs: int = 1000, n_steps=10, save=True):
        
        """
        Train model on simulated trajectories.

        Args:
            n_steps: Number of training steps
            save: If true, save a checkpoint after each epoch.
        """

        inputs_lst = sorted( [ x for x in os.listdir( self.ckpt_dir + 'data/' ) if 'inputs' in x ] )
        pc_outputs_lst = sorted( [ x for x in os.listdir( self.ckpt_dir + 'data/' ) if 'pc_outputs' in x ] )
        pos_lst = sorted( [ x for x in os.listdir( self.ckpt_dir + 'data/' ) if 'pos' in x ] )

        for epoch_idx in tqdm(range(n_epochs)):
            
            for chunk_idx in tqdm(range( len( inputs_lst ) )):

                inputs_chunk = torch.load( self.ckpt_dir + 'data/' + inputs_lst[ chunk_idx ] )
                pc_outputs_chunk = torch.load( self.ckpt_dir + 'data/' + pc_outputs_lst[ chunk_idx ] )
                pos_chunk = torch.load( self.ckpt_dir + 'data/' + pos_lst[ chunk_idx ] )

                for step_idx in tqdm(range( len( inputs_chunk ) )):

                    inputs = inputs_chunk[ step_idx ]
                    pc_outputs = pc_outputs_chunk[ step_idx ]
                    pos = pos_chunk[ step_idx ]

                    loss, err = self.train_step(inputs, pc_outputs, pos)

                    self.loss.append(loss)
                    self.err.append(err)
                    
            if save:

                # Save checkpoint
                ckpt_path = os.path.join(self.ckpt_dir, 'epoch_{}.pth'.format(epoch_idx))
                torch.save(self.model.state_dict(), ckpt_path)
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, 'most_recent_model.pth'))