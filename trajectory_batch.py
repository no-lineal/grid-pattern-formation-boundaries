import torch

import os

from tqdm import tqdm

class TrajectoryStatic( object ):

    def __init__(self, options, trajectory_generator, polygon):

        self.options = options
        self.trajectory_generator = trajectory_generator
        self.polygon = polygon

        # Set up checkpoints

        self.ckpt_dir = os.path.join(options.save_path, options.model_name )

        ###
        #ckpt_path = os.path.join(self.ckpt_dir, 'most_recent_model.pth')

        #if restore and os.path.isdir(self.ckpt_dir) and os.path.isfile(ckpt_path):

        #    self.model.load_state_dict(torch.load(ckpt_path))
        #    print("Restored trained model from {}".format(ckpt_path))

        #else:

        #    if not os.path.isdir(self.ckpt_dir):
        ###

        os.makedirs(self.ckpt_dir, exist_ok=True)

        print( 'initialize trajectory generator' )
        print("Saving to: {}".format(self.ckpt_dir))

    def trajectory_iterator( self, n_chunks: int = 1000, n_steps=10, save=True ):

        # Construct generator
        gen = self.trajectory_generator.get_generator()

        for chunk_idx in range( n_chunks ):

            inputs_lst = []
            pc_outputs_lst = []
            pos_lst = []

            for step_idx in tqdm( range(n_steps) ):

                inputs, pc_outputs, pos = next(gen)

                inputs_lst.append( inputs )
                pc_outputs_lst.append( pc_outputs )
                pos_lst.append( pos )

            # save outputs
            torch.save( inputs_lst, self.ckpt_dir + 'data/' + f'inputs_lst_{ chunk_idx }.pth' )
            torch.save( pc_outputs_lst, self.ckpt_dir + 'data/' + f'pc_outputs_lst_{ chunk_idx }.pth' )
            torch.save( pos_lst, self.ckpt_dir + 'data/' + f'pos_lst_{ chunk_idx }.pth' )