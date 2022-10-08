import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from advchain.augmentor.adv_transformation_base import AdvTransformBase  # noqa

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def get_base_grid(batch_size, image_height, image_width, image_depth=None, device=torch.device('cuda')):
    '''

    :param batch_size:
    :param image_height:
    :param image_width:
    :param use_gpu:
    :param requires_grad:
    :return:
    grid-wh: 4d grid N*2*H*W
    '''
    # get base grid
    if image_depth is None:
        y_ind, x_ind = torch.meshgrid(
            [torch.linspace(-1, 1, image_height,device=device), torch.linspace(-1, 1, image_width,device=device)],indexing='ij')  # image space [0-H]
        x_ind = x_ind.unsqueeze(0).unsqueeze(0)  # 1*1*H*W
        y_ind = y_ind.unsqueeze(0).unsqueeze(0)  # 1*1*H*W
        x_ind = x_ind.repeat(batch_size, 1, 1, 1)  # N*1*H*W
        y_ind = y_ind.repeat(batch_size, 1, 1, 1)
        x_ind.float()
        y_ind.float()
        grid = torch.cat((x_ind, y_ind), dim=1)
    else:
        z_ind,y_ind, x_ind = torch.meshgrid(
            [torch.linspace(-1, 1, image_height,device=device), torch.linspace(-1, 1, image_width,device=device),torch.linspace(-1, 1, image_depth,device=device)],indexing='ij')  # image space [0-H]
        
        x_ind = x_ind.unsqueeze(0).unsqueeze(0)  
        y_ind = y_ind.unsqueeze(0).unsqueeze(0)  
        z_ind = z_ind.unsqueeze(0).unsqueeze(0)  

        x_ind = x_ind.repeat(batch_size, 1, 1, 1,1)  # N*1*H*W*D
        y_ind = y_ind.repeat(batch_size, 1, 1, 1,1)
        z_ind = z_ind.repeat(batch_size, 1, 1, 1,1)

        x_ind.float()
        y_ind.float()
        z_ind.float()
   

        grid = torch.cat((x_ind, y_ind,z_ind), dim=1)
        # print(grid.size())
    return grid

def calculate_image_diff(images):
    """Difference map of the image.
    :param images: 4D tensor, batch of images, [batch,ch,h,w]
    return :
    dx: difference in x-direction: batch*ch*H*W
    dy: difference in y-direction: batch*ch*H*W

    """
    assert len(images.size()) == 4 , 'only support 2D version'
    dx = torch.zeros_like(images)
    dy = torch.zeros_like(images)
    # forward difference in first column
    dx[:, :, :, 0] = images[:, :, :, 1] - images[:, :, :, 0]
    dx[:, :, :, -1] = images[:, :, :, -1] - images[:, :, :, -2]
    dx[:, :, :, 1:-1] = 0.5 * (images[:, :, :, 2:] - images[:, :, :, :-2])

    dy[:, :, 0, :] = images[:, :, 1, :] - images[:, :, 0, :]
    dy[:, :, -1, :] = images[:, :, -1, :] - images[:, :, -2, :]
    dy[:, :, 1:-1, :] = 0.5 * (images[:, :, 2:, :] - images[:, :, :-2, :])
    return dx, dy


def calculate_jacobian_determinant(data, type='displacement'):
    '''
    calculate the jacobian determinant over a batch of transformations in pytorch
    :param data: N*2*H*W Input array, changes in x direction: dx: data[:,0]
    :param type: str: 'displacement'
    :return: nd tensor: N*1*H*W determinant of jacobian for transformation
    '''
    type_library = ['displacement']
    assert len(data.size()) == 4 and data.size(
        1) == 2, 'only support 2D version, and transformation format is NCHW'
    assert type in type_library, 'only support {} but found: '.format(
        type_library, type)
    # for each point on the grid, get a 4d tuple [dxx,dyy,dxy,dyx] and calc the determinant using det=(1+dxx)*(1+dyy)-dxy*dyx
    dx = data[:, [0], :, :]
    dy = data[:, [1], :, :]
    dxx, dxy = calculate_image_diff(dx)
    dyx, dyy = calculate_image_diff(dy)

    determinant = (1+dxx)*(1+dyy)-dxy*dyx
    return determinant


def integrate_by_add(basegrid, dxy):
    '''
    transform images with the given deformation fields
    :param basegrid
    :param dxy: dense deformation in vertical direction:N*1*H*W
    :return:
    new_grid: the input to the torch grid_sample function.
    torch tensor matrix: N*H*W*2:[dx,dy]
    '''

    basegrid+= dxy
    # basegrid = torch.clamp(basegrid, -1, 1)
    return basegrid


def vectorFieldExponentiation2D(duv, nb_steps=8, type='ss', device=torch.device("cuda")):
    '''
        Computes fast vector field exponentiation as proposed in:
        https://hal.inria.fr/file/index/docid/349600/filename/DiffeoDemons-NeuroImage08-Vercauteren.pdf
        :param duv: velocity field in ,y direction : N*2*H*W,
        :param N: number of steps for integration
        :return:
        integrated deformation field at time point 1: N2HW, [dx,dy]
   '''

    # phi(i/2^n)=x+u(x)
    grid_wh = get_base_grid(batch_size=duv.size(0), image_height=duv.size(2), image_width=duv.size(3),
                            device=device)
    duv_interval = duv/(2.0 ** nb_steps)
    phi = integrate_by_add(grid_wh, duv_interval)

    if type == 'ss':
        for i in range(nb_steps):
            # e.g. phi(2^i/2^n) =phi(2^(i-1)/2^n) \circ phi((2^(i-1)/2^n))
            phi = applyComposition2D(phi, phi)
    else:
        # euler integration, here nb_steps becomes exact time steps
        interval_phi = phi
        for i in range(nb_steps):
            # . phi((i+1)/2^n) =phi(1/n) \circ phi(i/n))
            phi = applyComposition2D(interval_phi, phi)
    # get the offset flow
    phi = phi-grid_wh
    return phi

def vectorFieldExponentiation3D(duv, nb_steps=8, type='ss', device=torch.device("cuda")):
    '''
        Computes fast vector field exponentiation as proposed in:
        https://hal.inria.fr/file/index/docid/349600/filename/DiffeoDemons-NeuroImage08-Vercauteren.pdf
        :param duv: velocity field in ,y direction : N*2*H*W,
        :param N: number of steps for integration
        :return:
        integrated deformation field at time point 1: N2HW, [dx,dy] OR N2HWD [dx,dy,dz]
   '''

    # phi(i/2^n)=x+u(x)
    grid_whd = get_base_grid(batch_size=duv.size(0), image_height=duv.size(2), image_width=duv.size(3),image_depth=duv.size(4),
                            device=device)
    duv_interval = duv/(2.0 ** nb_steps)
    while torch.norm(duv_interval)>0.5:
        nb_steps+=1
        duv_interval = duv/(2.0 ** nb_steps)
    phi = integrate_by_add(grid_whd, duv_interval)

    if type == 'ss':
        for i in range(nb_steps):
            # e.g. phi(2^i/2^n) =phi(2^(i-1)/2^n) \circ phi((2^(i-1)/2^n))
            phi = applyComposition3D(phi, phi)
    else:
        # euler integration
        interval_phi = phi
        for i in range((2.0 ** nb_steps)):
            # . phi((i+1)/2^n) =phi(1/n) \circ phi(i/n))
            phi = applyComposition3D(interval_phi, phi)
    # get the offset flow
    phi = phi-grid_whd
    return phi

def applyComposition2D(flow1, flow2):
    """
    Compose two deformation fields using linear interpolation.
    :param flow1 [f]::N*2*H*W, [dx,dy] A->B, the left is the 'static' deformation
    :param flow2 [g]:N*2*H*W  [dx,dy] B->C, the right is the 'delta' deformation
    :return:
    flow_field/deformation field h= g(f(x)):A->C, [dx,dy], N*2*H*W
    """     
    interpolated_f1 = F.grid_sample(flow1, flow2.permute(
        0, 2, 3, 1), padding_mode='border', align_corners=True)  # NCHW
    
    return interpolated_f1

def applyComposition3D(flow1, flow2):
    """
    Compose two deformation fields using linear interpolation.
    :param flow1 [f]::N*3*H*W*D, [dx,dy, dz] A->B, the left is the 'static' deformation
    :param flow2 [g]:N*3*H*W*D  [dx,dy, dz] B->C, the right is the 'delta' deformation
    :return:
    flow_field/deformation field h= g(f(x)):A->C, [dx,dy, dz], N*3*H*W*D
    """
    interpolated_f1 = F.grid_sample(flow1, flow2.permute(
        0, 2, 3, 4, 1), padding_mode='border', align_corners=True)  # NCHW
    return interpolated_f1

class AdvMorph(AdvTransformBase):
    """
     Adv Morph
    """

    def __init__(self,
                 spatial_dims=2,
                 config_dict={'epsilon': 1.5,
                              'data_size': [10, 1, 8, 8],
                              'vector_size': [4, 4],
                              'interpolator_mode': 'bilinear'
                              },
                 power_iteration=False,
                 device = torch.device("cuda"),padding_value=None,
                 use_gpu: bool = True, debug: bool = False):
        """_summary_

        Args:
            spatial_dims (int, optional): _description_. Defaults to 2.
            config_dict (dict, optional): _description_. Defaults to {'epsilon': 1.5, 'data_size': [10, 1, 8, 8], 'vector_size': [4, 4], 'interpolator_mode': 'bilinear' }.
            power_iteration (bool, optional): _description_. Defaults to False.
            device (_type_, optional): _description_. Defaults to torch.device("cuda").
            padding_value (_type_, optional): float. padding values when performing image warping to the out-of-image region. Defaults to None (0). You can change it to other float values to make it consistent with background values.
            use_gpu (bool, optional): whether to use gpu. Defaults to True.
            debug (bool, optional): debug mode. Defaults to False.
        """
        super(AdvMorph, self).__init__(spatial_dims=spatial_dims,
            config_dict=config_dict, use_gpu=use_gpu, debug=debug,device=device)
        self.align_corners = True
        # in the original demons paper, the sigma for gaussian smoothing is recommended to set to 1.
        self.sigma = 1
        self.gaussian_ks = 5
        self.smooth_iter = 1
        self.num_steps = 8  # internal steps for scaling and squaring intergration
        self.interpolator_mode = 'bilinear'
        self.integration_type = 'ss'
        self.param = None
        self.power_iteration = power_iteration
        self.padding_value = padding_value

    def init_config(self, config_dict):
        '''
        initialize a set of transformation configuration parameters
        '''
        self.epsilon = config_dict['epsilon']
        self.xi = 0.5
        self.data_size = config_dict['data_size']
        self.vector_size = config_dict['vector_size']
        self.interpolator_mode = config_dict['interpolator_mode']

    def init_parameters(self):
        '''
        initialize transformation parameters
        return random transformaion parameters
        '''
        self.init_config(self.config_dict)
        if self.spatial_dims == 2:
            self.base_grid = get_base_grid(
                batch_size=self.data_size[0], image_height=self.data_size[2], image_width=self.data_size[3], device=self.device)
            vector = self.init_velocity(
                batch_size=self.data_size[0],  height=self.vector_size[0], width=self.vector_size[1], use_zero=False)
            
        elif self.spatial_dims == 3:
            self.base_grid = get_base_grid(
                batch_size=self.data_size[0], image_height=self.data_size[2], image_width=self.data_size[3], image_depth = self.data_size[4], 
                     device=self.device)
            vector = self.init_velocity(
                batch_size=self.data_size[0],  height=self.vector_size[0], width=self.vector_size[1],depth=self.vector_size[2], use_zero=False)
        else:
            raise NotImplementedError('only 2D and 3D are supported')
        self.param = vector
        if self.debug:
            print('init velocity:', vector.size())
        return vector

    def forward(self, data, interpolation_mode=None):
        '''
        forward the data to get transformed data
        :param data: input images x, N4HW
        :return:
        tensor: transformed images
        '''
        if self.debug:
            print('apply morphological transformation')
        if self.param is None:
            self.param = self.init_parameters()
        if interpolation_mode is None:
            interpolation_mode = self.interpolator_mode
        if self.power_iteration and self.is_training:
            dxy, displacement = self.get_deformation_displacement_field(
                duv=self.xi*self.param)
        else:
            dxy, displacement = self.get_deformation_displacement_field(
                duv=self.epsilon*self.param)
        dxy = torch.clamp(
            dxy, -1, 1)
        transformed_image = self.transform(data, dxy, mode=interpolation_mode)

        self.diff = transformed_image-data
        self.displacement = displacement
      
        return transformed_image

    def backward(self, data, interpolation_mode=None):
        '''
        backward image
        '''
        if interpolation_mode is None:
            interpolation_mode = self.interpolator_mode
        if self.power_iteration and self.is_training:
            dxy, displacement = self.get_deformation_displacement_field(
                duv=-self.xi*self.param)
        else:
            dxy, displacement = self.get_deformation_displacement_field(
                duv=-self.epsilon*self.param)
        
        # dxy = torch.clamp(
        #     dxy, -1, 1)
        transformed_image = self.transform(
            data, dxy, mode=self.interpolator_mode)
        if self.debug:
            logging.info('warp back.')
        return transformed_image

    def predict_forward(self, data):
        return self.forward(data)

    def predict_backward(self, data):
        return self.backward(data)

    def get_deformation_displacement_field(self, duv=None):
        if duv is None:
            duv = self.param
        dxy = self.DemonsCompose(
            duv=duv, init_deformation_dxy=self.base_grid, smooth=True)
        if self.spatial_dims==2: disp = dxy.permute(0, 2, 3, 1)-self.base_grid.permute(0, 2, 3, 1)
        elif self.spatial_dims==3: disp = dxy.permute(0, 2, 3, 4, 1)-self.base_grid.permute(0, 2, 3, 4, 1)
        else: raise NotImplementedError('only 2D and 3D are supported')
        return dxy, disp

    def init_velocity(self, batch_size, height, width,depth=None, use_zero=False):
        '''

        :param batch_size:
        :param height:
        :param width:
        :param use_zero: initialize with zero values
        :return:
        nd tensor: N*2*H*W, a velocity field/offset field with values between -1 and 1.
        '''
        # offsets = offsets.cuda()
        if self.spatial_dims == 2:
            if use_zero:
                velocity = torch.zeros(batch_size, 2, height, width,device=self.device)
            else:
                velocity = torch.rand(batch_size, 2, height, width,device=self.device)
                velocity = velocity*2-1
        elif self.spatial_dims == 3:
            if use_zero:
                velocity = torch.zeros(batch_size, 3, height, width,depth,device=self.device)
            else:
                velocity = torch.rand(batch_size, 3, height, width,depth,device=self.device)
                velocity = velocity*2-1
        else:
            raise NotImplementedError('only 2D and 3D are supported')
        velocity =self.unit_normalize(velocity)
        return velocity

    def gaussian_smooth(self, inputvector, iter=1, kernel_size=41, sigma=8):
        '''
        apply gaussian smooth functions to deformation field to avoid unrealistic and too aggressive deformations
        :param input: NCHW
        :param iter: max number of iterations, avoid infinestimal.
        :return: smoothed deformation
        '''
        n_channel = inputvector.size(1)
        gaussian_conv = self.get_gaussian_kernel(
            kernel_size=kernel_size, sigma=sigma, channels=n_channel)
        for i in range(iter):
            inputvector = gaussian_conv(inputvector)
        return inputvector

    def get_gaussian_kernel(self, kernel_size=5, sigma=8, channels=3):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        # Use n_sd sigmas
        if self.spatial_dims == 2:
            if kernel_size < 2*int(4 * sigma + 0.5)+1: ## change to 4, to make it align with the scipy implementation https://github.com/scipy/scipy/blob/v1.8.1/scipy/ndimage/_filters.py#L264-L347
                # odd size so padding results in correct output size
                kernel_size = 2*int(4 * sigma + 0.5)+1
        elif self.spatial_dims == 3:
                if kernel_size <= 2*int(4 * sigma + 0.5)+1:
                    kernel_size = 2*int(4 * sigma + 0.5)+1

        x_coord = torch.arange(kernel_size)
       
        if self.spatial_dims == 2:
            x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
            y_grid = x_grid.t()
            grid = torch.stack([x_grid, y_grid], dim=-1).float()
        elif self.spatial_dims == 3:
            x_grid_2d = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
            x_grid = x_coord.repeat(kernel_size*kernel_size).view(kernel_size, kernel_size, kernel_size)
            y_grid_2d = x_grid_2d.t()           
            y_grid  = y_grid_2d.repeat(kernel_size,1).view(kernel_size, kernel_size, kernel_size)
            z_grid = y_grid_2d.repeat(1,kernel_size).view(kernel_size, kernel_size, kernel_size)
            grid = torch.stack([x_grid, y_grid, z_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = torch.exp(
            -torch.sum((grid - mean) ** 2., dim=-1) /
            (2 * variance)
        )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        pad_size = kernel_size // 2


        if self.spatial_dims == 2:
            # Reshape to 2d depthwise convolutional weight
            gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
            gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
            gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                        kernel_size=kernel_size, groups=channels, dilation=1, stride=1, bias=False,
                                        padding=pad_size)
        elif self.spatial_dims == 3:
            gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
            gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)
            gaussian_filter = nn.Conv3d(in_channels=channels, out_channels=channels,
                                        kernel_size=kernel_size, groups=channels,
                                        bias=False, padding=pad_size)


        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        if self.use_gpu:
            gaussian_filter = gaussian_filter.to(self.device)
        return gaussian_filter

    def DemonsCompose(self, duv, init_deformation_dxy, smooth=False):
        '''
        :param duv: velocity field
        :param init_deformation_dxy:
        :return:
        new composed_deformation_grid N*2*H*W
        '''
        interpolate_mode = 'bilinear' if self.spatial_dims == 2 else 'trilinear'
        duv = self.gaussian_smooth(
                duv, iter=self.smooth_iter, kernel_size=self.gaussian_ks, sigma=self.sigma)
        duv = F.interpolate(duv, size=self.base_grid.size()[2:], mode=interpolate_mode, align_corners=False) 
        if self.spatial_dims == 2:
            integrated_offsets = vectorFieldExponentiation2D(duv=duv, nb_steps=self.num_steps,
                                                            type=self.integration_type,device=self.device)
            if integrated_offsets.size(2) != self.base_grid.size(2) or integrated_offsets.size(3) != self.base_grid.size(3):
                integrated_offsets = F.interpolate(integrated_offsets, size=(self.base_grid.size(
                    2), self.base_grid.size(3)), mode=interpolate_mode, align_corners=False)

            # update deformation with composition
            composed_deformation_grid = applyComposition2D(
                init_deformation_dxy, integrated_offsets + self.base_grid)
        elif self.spatial_dims == 3:
            integrated_offsets = vectorFieldExponentiation3D(duv=duv, nb_steps=self.num_steps,
                                                            type=self.integration_type,device=self.device)
            if integrated_offsets.size(2) != self.base_grid.size(2) or integrated_offsets.size(3) != self.base_grid.size(3) or integrated_offsets.size(4) != self.base_grid.size(4):
                integrated_offsets = F.interpolate(integrated_offsets, size=self.base_grid.size()[2:], mode=interpolate_mode, align_corners=False)

            # update deformation with composition
            composed_deformation_grid = applyComposition3D(
                init_deformation_dxy, integrated_offsets + self.base_grid)

        # smooth with gaussian for smoothness regularization
        if smooth:
            smoothed_offset = self.gaussian_smooth(composed_deformation_grid - self.base_grid, sigma=self.sigma,
                                                   kernel_size=self.gaussian_ks, iter=1)
            composed_deformation_grid = smoothed_offset + self.base_grid
        composed_deformation_grid = torch.clamp(composed_deformation_grid,-1,1)
        return composed_deformation_grid

    def train(self):
        self.is_training = True
        if self.param is None:
            self.init_parameters()
        if self.power_iteration:
            self.param = self.unit_normalize(self.param)
        self.param = torch.nn.Parameter(self.param, requires_grad=True)

    def optimize_parameters(self, step_size=None):
        if step_size is None:
            self.step_size = step_size
        if self.debug:
            logging.info('optimize morph')
        try:
            if self.power_iteration:
                duv = self.unit_normalize(self.param.grad)
                param = duv.detach()
            else:
                duv = self.unit_normalize(self.param.grad)
                param = self.param+step_size*duv.detach()
            self.param = param.detach()
        except:
            logging.warning('fail to optimize.This may due to the strength of deformation is too strong, that the structure cannot be well preserved. Try use smaller epsilon')
        return self.param
    
    def rescale_parameters(self,param=None):
        if param is None:
            param =self.param
        self.param =self.unit_normalize(param)
        return self.param

    def transform(self, data, deformation_dxy, mode='bilinear', padding_mode='border'):
        '''
        transform images with the given deformation fields
        :param data: input data, N*C*H*W
        :param deformation_dxy: deformation N*2*H*W
        :return:
        transformed data: torch tensor matrix: N*ch*H*W
        deformed_grid: torch tensor matrix: N*H*W*2
        offsets: N*H*W*2
        '''
        if self.spatial_dims == 2:
            grid_tensor = deformation_dxy.permute(0, 2, 3, 1)  # N*H*W*2
        else:
            grid_tensor = deformation_dxy.permute(0, 2, 3, 4, 1)
        # transform images
        if self.padding_value is not None:
            if isinstance(self.padding_value,float):
                data = data-self.padding_value
                transformed_image = F.grid_sample(
                        data, grid_tensor, mode=mode, align_corners=self.align_corners,padding_mode = 'zeros')
                transformed_image +=self.padding_value
        else:
              transformed_image = F.grid_sample(
                        data, grid_tensor, mode=mode, align_corners=self.align_corners,padding_mode = 'zeros')
        # gen flow field
        return transformed_image

    def get_name(self):
        return 'morph'

    def is_geometric(self):
        return 1




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from os.path import join as join
    from advchain.common.utils import check_dir
    dir_path = './log'
    check_dir(dir_path, create=True)
    images = torch.zeros((10, 1, 128, 128)).float()
    images = images.cuda()
    images[:, :, ::8, :] = 0.5
    images[:, :, :, ::8] = 0.5

    print('input:', images)
    augmentor = AdvMorph(
                    spatial_dims=2,
                    config_dict={'epsilon': 1.5,
                                      'xi': 0.1,
                                      'data_size': [10, 1, 128, 128],
                                      'vector_size': [128//16, 128//16],
                                      'interpolator_mode': 'bilinear'
                                      },

                         debug=True, use_gpu=True)
    augmentor.init_parameters()
    transformed = augmentor.forward(images.cuda())
    recovered = augmentor.backward(transformed)
    error = recovered-images
    print('sum error', torch.sum(error))

    plt.subplot(131)
    plt.imshow(images.cpu().numpy()[0, 0])

    plt.subplot(132)
    plt.imshow(transformed.cpu().numpy()[0, 0])

    plt.subplot(133)
    plt.imshow(recovered.cpu().numpy()[0, 0])

    plt.savefig(join(dir_path, 'test_morph.png'))
    
    ## 3D 

    images_3D = torch.zeros((10, 1, 128, 128,128)).float()
    images_3D = images_3D.cuda()
    images_3D[:, :, ::8, ::8,:] = 2*128.0
    images_3D[:, :, :, ::8,::8] = 2*128.0
    images_3D[:, :, ::8,:,::8] = 2*128.0
    images_3D[:,:,0:80,0:80,0:80] = 128.0
    print('input:', images_3D)
    augmentor = AdvMorph(
                    spatial_dims=3,
                    config_dict={'epsilon': 2.5,
                                      'xi': 0.1,
                                      'data_size': [10, 1, 128, 128,128],
                                      'vector_size': [8,8,8],
                                      'interpolator_mode': 'bilinear'
                                      },

                         debug=True, use_gpu=True)
    augmentor.init_parameters()
    transformed = augmentor.forward(images_3D.cuda())
    recovered = augmentor.backward(transformed)
    error = recovered-images_3D
    print('sum error', torch.sum(error))

    plt.subplot(331)
    plt.imshow(images_3D.cpu().numpy()[0, 0,9])

    plt.subplot(332)
    plt.imshow(transformed.cpu().numpy()[0, 0, 9 ])

    plt.subplot(333)
    plt.imshow(recovered.cpu().numpy()[0, 0,9])

    plt.subplot(334)
    plt.imshow(images_3D.cpu().numpy()[0,0,:,9])

    plt.subplot(335)
    plt.imshow(transformed.cpu().numpy()[0,0, :,9 ])

    plt.subplot(336)
    plt.imshow(recovered.cpu().numpy()[0,0,:,9,])

    plt.subplot(337)
    plt.imshow(images_3D.cpu().numpy()[0, 0,:,:,9])

    plt.subplot(338)
    plt.imshow(transformed.cpu().numpy()[0, 0, :,:,9 ])

    plt.subplot(339)
    plt.imshow(recovered.cpu().numpy()[0, 0,:,:,9])

    plt.savefig(join(dir_path, 'test_morph_3D.png'))