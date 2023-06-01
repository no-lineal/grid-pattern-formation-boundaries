import numpy as np
import torch
import scipy
from shapely.geometry import Point

import warnings
warnings.filterwarnings("ignore")

class PlaceCells( object ):

    def __init__( self, options, polygon, us=None ):

        self.load_path = options.load_path
        self.save_path = options.save_path

        self.Np = options.Np
        self.sigma = options.sigma # width of place cell center tuning curve (m)
        self.surround_scale = options.surround_scale
        self.periodic  = options.periodic
        self.DoG = options.DoG

        self.device = options.device

        # environment boundaries
        self.polygon = polygon

        self.softmax = torch.nn.Softmax( dim=-1 )

        print('Initializing place cells...')
        
        if options.precomputed:

            self.us = torch.tensor(
                np.load( self.load_path + 'place_cell_centers.npy' )
            ).to( self.device )

        elif not isinstance(self.polygon, np.ndarray):

            # random seed
            np.random.seed( 0 )

            min_x, min_y, max_x, max_y = self.polygon.bounds

            width = max_x - min_x
            height = max_y - min_y

            points = []
            while len(points) < self.Np:
                
                x = np.random.uniform(-width/2, width/2)
                y = np.random.uniform(-height/2, height/2)

                point = Point( x, y )

                if self.polygon.contains( point ):
                    
                    points.append( point )

            self.us = torch.tensor(
                np.array( 
                    [ [ point.x, point.y ] for point in points ]
                )
            )

            self.us = self.us.to( self.device )

        elif isinstance(self.polygon, np.ndarray):

            # random seed
            np.random.seed( 0 )

            min_values = np.min(polygon, axis=0)
            max_values = np.max(polygon, axis=0)

            points = []
            while len(points) < self.Np:

                x = np.random.uniform(min_values[0], max_values[0])
                y = np.random.uniform(min_values[1], max_values[1])
                z = np.random.uniform(min_values[2], max_values[2])

                point = np.array([x, y, z])

                if np.all( point >= np.min( polygon , axis=0 ) ) and np.all( point <= np.max( polygon , axis=0 ) ):

                    points.append( point )

            self.us = torch.tensor( points ).to( self.device )

    def get_activation(self, pos):
        

        '''

        Get place cell activations for a given position.

        Args:
            pos: 2d position of shape [batch_size, sequence_length, 2].

        Returns:
            outputs: Place cell activations with shape [batch_size, sequence_length, Np].

        '''

        d = torch.abs(pos[:, :, None, :] - self.us[None, None, ...]).float()

        if self.periodic:
            
            dx = d[:,:,:,0]
            dy = d[:,:,:,1]
            dx = torch.minimum(dx, self.box_width - dx) 
            dy = torch.minimum(dy, self.box_height - dy)
            d = torch.stack([dx,dy], axis=-1)

        norm2 = (d**2).sum(-1)

        # Normalize place cell outputs with prefactor alpha=1/2/np.pi/self.sigma**2,
        # or, simply normalize with softmax, which yields same normalization on 
        # average and seems to speed up training.
        outputs = self.softmax(-norm2/(2*self.sigma**2))

        if self.DoG:

            # Again, normalize with prefactor 
            # beta=1/2/np.pi/self.sigma**2/self.surround_scale, or use softmax.
            outputs -= self.softmax(-norm2/(2*self.surround_scale*self.sigma**2))

            # Shift and scale outputs so that they lie in [0,1].
            min_output,_ = outputs.min(-1,keepdims=True)
            outputs += torch.abs(min_output)
            outputs /= outputs.sum(-1, keepdims=True)

        return outputs
    
    def get_nearest_cell_pos(self, activation, k=3):

        """

        Decode position using centers of k maximally active place cells.
        
        Args: 
            activation: Place cell activations of shape [batch_size, sequence_length, Np].
            k: Number of maximally active place cells with which to decode position.

        Returns:
            pred_pos: Predicted 2d position with shape [batch_size, sequence_length, 2].

        """
        _, idxs = torch.topk(activation, k=k)
        pred_pos = self.us[idxs].mean(-2)
        
        return pred_pos
    
    def grid_pc(self, pc_outputs, res=32):

        """
        
        Interpolate place cell outputs onto a grid

        """

        if self.environment == 'square':

            coordsx = np.linspace(-self.box_width/2, self.box_width/2, res)
            coordsy = np.linspace(-self.box_height/2, self.box_height/2, res)

            grid_x, grid_y = np.meshgrid(coordsx, coordsy)
            grid = np.stack([grid_x.ravel(), grid_y.ravel()]).T

        elif self.environment == 'trapezoid':

            trapezoid = Polygon(
                [
                    (-self.box_width/2, -self.box_height/2),
                    (-self.box_width/2 + self.box_width/4, self.box_height/2),
                    (self.box_width/2 - self.box_width/4, self.box_height/2),
                    (self.box_width/2, -self.box_height/2)
                ]
            )

            xmin, ymin, xmax, ymax = trapezoid.bounds

            coordsx = np.arange( xmin, xmax, res )
            coordsy = np.arange( ymin, ymax, res )

            grid_x, grid_y = np.meshgrid( coordsx, coordsy )
            grid = np.stack([grid_x.ravel(), grid_y.ravel()]).T

        # Convert to numpy
        pc_outputs = pc_outputs.reshape(-1, self.Np)
        
        T = pc_outputs.shape[0] #T vs transpose? What is T? (dim's?)
        pc = np.zeros([T, res, res])

        for i in range(len(pc_outputs)):

            gridval = scipy.interpolate.griddata(self.us.cpu(), pc_outputs[i], grid)
            pc[i] = gridval.reshape([res, res])
        
        return pc
    
    def compute_covariance(self, res=30):

        """
        
        Compute spatial covariance matrix of place cell outputs
        
        """

        pos = np.array(
            np.meshgrid(
                np.linspace( -self.box_width/2, self.box_width/2, res),
                np.linspace( -self.box_height/2, self.box_height/2, res)
            )
        ).T

        pos = torch.tensor(pos)

        # Put on GPU if available
        pos = pos.to(self.device)

        # maybe specify dimensions here again?
        pc_outputs = self.get_activation( pos ).reshape( -1, self.Np ).cpu()

        C = pc_outputs@pc_outputs.T # matrix multiplication
        Csquare = C.reshape(res, res, res, res)

        Cmean = np.zeros([res,res])

        for i in range(res):
            for j in range(res):

                Cmean += np.roll(np.roll(Csquare[i,j], -i, axis=0), -j, axis=1)
                
        Cmean = np.roll(np.roll(Cmean, res//2, axis=0), res//2, axis=1)

        return Cmean
