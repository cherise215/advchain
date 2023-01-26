
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import torch
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from advchain.augmentor.adv_transformation_base import AdvTransformBase  # noqa


class AdvAffine(AdvTransformBase):
    """
     Adv Affine
    """

    def __init__(self,
                 spatial_dims=2,
                 config_dict={
                     'rot': 30.0 / 180.0,
                     'scale_x': 0.2,
                     'scale_y': 0.2,
                     'shift_x': 0.1,
                     'shift_y': 0.1,
                    #  'shear_x': 0.,
                    #  'shear_y': 0.,
                     'data_size': [1, 1, 8, 8],
                     'forward_interp': 'bilinear',
                     'backward_interp': 'bilinear'
                 },
                 image_padding_mode = "zeros",
                 power_iteration=False,
                 use_gpu=True, debug=False, device=torch.device("cuda")):
        '''
        initialization,

        for 3D, the default config should be like:
        config_dict={
            ## rotation in radians about different axis
            'rot_x': 0.0,
            'rot_y':0.0,
            'rot_z':0.0,

            ## scaling in radians along different axis
            'scale_x': 0.0,
            'scale_y': 0.0,
            'scale_z': 0.0,
            ## shift in x,y, z direction
            'shift_x': 0.,
            'shift_y': 0.,
            'shift_z': 0.,
            ## shearing functions (not implemented yet)
            # 'shear_z_x': 0.,
            # 'shear_z_y': 0.,
            # 'shear_x_z': 0.,
            # 'shear_y_z': 0.,

            'data_size': [*images_3D.size()],
            'forward_interp': 'bilinear',
            'backward_interp': 'bilinear'
        }, debug=True)

        image_padding_mode: padding mode for the image, default is "zeros", other options are "border", "reflection","lowest". You can also specify it as a float value, e.g., -1, if the image is normalized to be within [-1,1]
        '''
        super(AdvAffine, self).__init__(spatial_dims=spatial_dims,
            config_dict=config_dict, use_gpu=use_gpu, debug=debug,device=device)
        self.power_iteration = power_iteration
        self.image_padding_mode=image_padding_mode
        self.forward_interp = 'bilinear'
        self.backward_interp = 'bilinear'    

    def init_config(self, config_dict):
        '''
        initialize a set of transformation configuration parameters
        '''
        if self.spatial_dims <=3:
            self.translation_x = config_dict['shift_x']
            self.translation_y = config_dict['shift_y']
            self.scale_x = config_dict['scale_x']
            self.scale_y = config_dict['scale_y']
            if self.spatial_dims == 2:
                self.rot_ratio = config_dict['rot']
                # if 'shear_x' in config_dict.keys():  self.shear_x = config_dict['shear_x']
                # else: self.shear_x = 0
                # if 'shear_y' in config_dict.keys():  self.shear_y = config_dict['shear_y']
                # else: self.shear_y = 0
        if self.spatial_dims == 3:
            self.rot_x = config_dict['rot_x']
            self.rot_y = config_dict['rot_y']
            self.rot_z = config_dict['rot_z']

            self.scale_z = config_dict['scale_z']
            self.translation_z= config_dict['shift_z'] 
            # self.shear_z_x= config_dict['shear_z_x']            
            # self.shear_z_y = config_dict['shear_z_y']
            # self.shear_x_z = config_dict['shear_x_z']
            # self.shear_y_z = config_dict['shear_y_z']

        self.xi = 1e-6
        self.data_size = config_dict['data_size']
        if 'forward_interp' in config_dict:
            self.forward_interp = config_dict['forward_interp']
        if 'backward_interp' in config_dict:
            self.backward_interp = config_dict['backward_interp']


    def init_parameters(self):
        '''
        initialize transformation parameters
        return random transformaion parameters
        '''
        self.init_config(self.config_dict)
        batch_size = self.data_size[0]
        self.batch_size = batch_size
        affine_tensor = self.draw_random_affine_tensor_list(
            batch_size=batch_size)
        self.param = affine_tensor
        return affine_tensor

    def forward(self, data, interp=None,padding_mode=None):
        '''
        forward the data to get transformed data
        :param data: input images x, N4HW
        :return:
        tensor: transformed images
        '''
        if padding_mode is None:
            padding_mode = self.image_padding_mode
        if self.debug:
            print('apply affine transformation')
        if self.param is None:
            self.init_parameters()
        if interp is None:
            interp = self.forward_interp
        if self.power_iteration and self.is_training:
            self.affine_matrix = self.gen_batch_affine_matrix(
                self.xi * self.param)
        else:
            self.affine_matrix = self.gen_batch_affine_matrix(self.param)
     
        transformed_input = self.transform(
                data, self.affine_matrix, interp=interp, padding_mode=padding_mode)
        self.diff = data - transformed_input

        return transformed_input

    def predict_forward(self, data,interp=None,padding_mode=None):
        return self.forward(data,interp=interp,padding_mode=padding_mode)

    def predict_backward(self, data,interp=None,padding_mode=None):
        return self.backward(data,interp=interp, padding_mode=padding_mode)

    def backward(self, data, interp = None, padding_mode=None):
        assert not self.param is None, 'play forward before backward'
   
        inverse_matrix = self.get_inverse_matrix(self.affine_matrix)
        if interp is None:
            interp = self.backward_interp
        if padding_mode is None:
            padding_mode = self.image_padding_mode
        warped_back_output = self.transform(
            data, inverse_matrix, interp=interp, padding_mode=padding_mode)
        return warped_back_output

    def draw_random_affine_tensor_list(self, batch_size, identity_init=False):
        if self.spatial_dims == 2:
                num_params = 5
        elif self.spatial_dims == 3:
                num_params = 9
        if identity_init:
            affine_tensor = torch.zeros(
                batch_size, num_params, device=self.device, dtype=torch.float32)
        else:
            ## initialization [-1,1]
            affine_tensor = (2 * torch.rand(batch_size, num_params,
                                            dtype=torch.float32, device=self.device) - 1)
            affine_tensor = torch.nn.Hardtanh()(affine_tensor)

        return affine_tensor

    def optimize_parameters(self, step_size=None):
        if step_size is None:
            self.step_size = step_size
        if self.debug:
            # we assume that affine parameters are independent to each other.
            logger.info('optimize affine')
        try:
            if self.power_iteration:
                grad = self.param.grad.sign()
                self.param = grad.detach()
            else:
                grad = self.param.grad.sign().detach()
                param = self.param + step_size * grad
                self.param = param.detach()
        except:
            Warning('fail to optimize')
        return self.param

    def rescale_parameters(self):
        ## the scale of each operation is explicitly constrained in the transformation model, see get_batch_affine_matrix
        return self.param

    def train(self):
        self.is_training = True
        if self.power_iteration:
            self.param = self.param.sign()
        self.param = torch.nn.Parameter(self.param, requires_grad=True)

    def gen_batch_affine_matrix(self, affine_tensors):
        '''
        given affine parameters, gen batch-wise affine_matrix [bs*2*3]
        :param affine_tensors:N*7 for 2D or N*10 for 3D , [rot_radius, scalex, scale_y, tx,ty,shear_x,shear_y],[rot_radius, scalex, scale_y, scale_z, tx,ty,tz, shear_x,shear_y, shear_z]
        :return:
         affine_matrix [bs*2*3]
        '''
        # restrict transformations between [-1,1]
        affine_tensors = torch.nn.Hardtanh()(affine_tensors)
        if self.spatial_dims == 2:
            rot_radius, scalex, scaley, tx, ty = affine_tensors[:, 0], affine_tensors[:, 1], affine_tensors[:, 2], affine_tensors[:, 3], affine_tensors[:, 4]
            transformation_matrix = torch.stack([
            torch.stack([(1+scalex*self.scale_x) * (torch.cos(rot_radius*self.rot_ratio*math.pi)), (1+scaley *
                                                                                                    self.scale_y) * (-torch.sin(rot_radius*self.rot_ratio*math.pi)), tx*self.translation_x], dim=-1),
            torch.stack([(1+scalex*self.scale_x) * (torch.sin(rot_radius*self.rot_ratio*math.pi)), (1+scaley *
                                                                                                    self.scale_y) * (torch.cos(rot_radius*self.rot_ratio*math.pi)), ty*self.translation_y], dim=-1)
        ], dim=1)
        elif self.spatial_dims == 3:
             ## rotation 3D matrix
            (rot_x, rot_y,rot_z, scalex, scaley, scalez,tx, ty,tz) = (affine_tensors[:, 0], affine_tensors[:,1], affine_tensors[:, 2], 
                                                                                                        affine_tensors[:, 3], 
                                                                                                        affine_tensors[:, 4],
                                                                                                         affine_tensors[:, 5], 
                                                                                                         affine_tensors[:, 6],
                                                                                                         affine_tensors[:, 7],
                                                                                                         affine_tensors[:, 8])

                                                                                                         
            batch_size = self.batch_size
            device = self.device
            O = torch.zeros(batch_size, dtype=torch.float32, device = device)
            I = torch.ones(batch_size, dtype=torch.float32, device = device)        
            translation_matrix =torch.stack(
                               [torch.stack([I, O, O, tx*self.translation_x], dim=-1),
                                torch.stack([O, I, O, ty*self.translation_y], dim=-1),
                                torch.stack([O, O, I, tz*self.translation_z], dim=-1),
                                torch.stack([O, O, O, I], dim=-1)], dim=1)
            scale_matrix =torch.stack(
                               [torch.stack([(1 + scalex * self.scale_x), O, O,O], dim=-1),
                                torch.stack([O, (1 + scaley * self.scale_y), O, O], dim=-1),
                                torch.stack([O, O, (1 + scalez * self.scale_z), O], dim=-1),
                                torch.stack([O, O, O, I], dim=-1)], dim=1)
            ## rotation 3D matrix: https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions: Euler angles (z-y′-x″ intrinsic) → rotation matrix
            phi = rot_x*self.rot_x*math.pi
            theta = rot_y*self.rot_y*math.pi
            psi = rot_z*self.rot_z*math.pi

            rotation_matrix =torch.stack([
                                torch.stack([torch.cos(theta)*torch.cos(psi), -torch.cos(phi)*torch.sin(psi)+torch.sin(phi)*torch.sin(theta)*torch.cos(psi),torch.sin(phi)*torch.sin(psi)+torch.cos(phi)*torch.sin(theta)*torch.cos(psi) , O], dim=-1),
                                torch.stack([torch.cos(theta)*torch.sin(psi), torch.cos(phi)*torch.cos(psi)+torch.sin(phi)*torch.sin(theta)*torch.sin(psi),-torch.sin(phi)*torch.cos(psi)+torch.cos(phi)*torch.sin(theta)*torch.sin(psi), O], dim=-1),
                                torch.stack([-torch.sin(theta), torch.sin(phi)*torch.cos(theta) , torch.cos(phi)*torch.cos(theta) , O], dim=-1),
                                torch.stack([O, O, O, I], dim=-1)], dim=1)
            # ## reference: https://www.fil.ion.ucl.ac.uk/spm/doc/books/hbf2/pdfs/Ch2.pdf
            # shear_matrix =torch.stack(
            #                    [torch.stack([I, (shear_x * self.shear_a), (shear_y * self.shear_b) ,O], dim=-1),
            #                     torch.stack([O, I,  (shear_z * self.shear_c), O], dim=-1),
            #                     torch.stack([O, O, I, O], dim=-1),
            #                     torch.stack([O, O, O, I], dim=-1)], dim=1)
            transformation_matrix = torch.matmul(translation_matrix, torch.matmul(rotation_matrix, scale_matrix))
            transformation_matrix = transformation_matrix[:,:3, :4]
        # print ('transformation matrix size',transformation_matrix.size())
        if self.use_gpu:
            transformation_matrix.to(device=self.device)
        return transformation_matrix

    def make_batch_eye_matrix(self, batch_size, device):
        O = torch.zeros(batch_size, dtype=torch.float32,device=device)
        I = torch.ones(batch_size, dtype=torch.float32,device=device)
        if self.spatial_dims ==2:
            eyeMtrx = torch.stack([torch.stack([I, O, O], dim=-1),
                                torch.stack([O, I, O], dim=-1),
                                torch.stack([O, O, I], dim=-1)], dim=1)
        elif self.spatial_dims ==3:
            eyeMtrx = torch.stack([torch.stack([I, O, O, O], dim=-1),
                                torch.stack([O, I, O, O], dim=-1),
                                torch.stack([O, O, I, O], dim=-1),
                                torch.stack([O, O, O, I], dim=-1)], dim=1)
        return eyeMtrx

    def transform(self, data, affine_matrix, interp=None, padding_mode=None):
        '''
        transform images with the given affine matrix
        '''
        if padding_mode is not None:
            padding_mode = self.image_padding_mode
        if interp is None:
            interp = self.forward_interp
        grid_tensor = F.affine_grid(
            affine_matrix, data.size(), align_corners=True)
        if padding_mode=="lowest":
            flatten_data = data.view(data.size(0),-1)
            self.padding_values = torch.min(flatten_data,dim=1,keepdim=True).values.detach().clone()
            shift_data = data -  self.padding_values
            transformed_input = F.grid_sample(shift_data, grid_tensor, mode=interp, align_corners=True, padding_mode='zeros')
            transformed_input = transformed_input+ self.padding_values
        elif isinstance(padding_mode,float) or isinstance(padding_mode,int):
            self.padding_values = padding_mode
            shift_data = data -  self.padding_values
            transformed_input = F.grid_sample(
                shift_data, grid_tensor, mode=interp, align_corners=True, padding_mode='zeros')
            transformed_input = transformed_input+ self.padding_values
        else:
            transformed_input = F.grid_sample(
                data, grid_tensor, mode=interp, align_corners=True, padding_mode=padding_mode)
        return transformed_input

    def get_inverse_matrix(self, affine_matrix):
        homo_matrix = self.make_batch_eye_matrix(
            batch_size=affine_matrix.size(0), device=affine_matrix.device)
        homo_matrix[:, :self.spatial_dims] = affine_matrix
        inverse_matrix = homo_matrix.inverse()
        inverse_matrix = inverse_matrix[:, :self.spatial_dims, :]
        if self.debug:
            logger.info('inverse matrix', inverse_matrix.size())
        return inverse_matrix

    def get_name(self):
        return 'affine'

    def is_geometric(self):
        return 1




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from os.path import join as join
    from advchain.common.utils import check_dir

    dir_path = './log'
    check_dir(dir_path, create=True)
    images = torch.zeros((1, 1, 120, 120)).cuda()
    images[:, :, 20:50, 20:50] = 1.0
    print('input:', images)
    augmentor = AdvAffine(
        spatial_dims=2,
        config_dict={
            'rot': 0.2,
            'scale_x': 0.1,
            'scale_y': 0.1,
            'shift_x': 0.2,
            'shift_y': 0.2,
            'shear_x': 0.2,
            'shear_y': 0.2,
            'data_size': [*images.size()],
            'forward_interp': 'bilinear',
            'backward_interp': 'bilinear'
        }, debug=True)
    augmentor.init_parameters()
    transformed = augmentor.forward(images)
    recovered = augmentor.backward(transformed)
    mask = torch.ones_like(images)
    with torch.no_grad():
        mask = augmentor.backward(augmentor.forward(mask))

    mask[mask == 1] = True
    mask[mask != 1] = False
    print('mask', mask)
    error = recovered - images
    print('sum error', torch.sum(error))
    plt.subplot(131)
    plt.imshow(images.cpu().numpy()[0, 0])

    plt.subplot(132)
    plt.imshow(transformed.cpu().numpy()[0, 0])

    plt.subplot(133)
    plt.imshow(recovered.cpu().numpy()[0, 0])

    # plt.subplot(144)
    # plt.imshow(mask.cpu().numpy()[0,0])
    plt.savefig(join(dir_path, 'test_affine.png'))

    # adv test three D

    images_3D = torch.zeros(2, 1, 20,20 , 20).cuda()
    images_3D[:, :, 0:10,  0:10, 0:10] =1.0
    images_3D =images_3D.clone()

    images_3D = images_3D.float().clone()
    images_3D.requires_grad = False
    print('input:', images_3D.size())
    augmentor = AdvAffine(
        spatial_dims=3,
        config_dict={
            'rot_x': 0.,
            'rot_y': 0.,
            'rot_z': 0.2,

            'scale_x': 0.0,
            'scale_y': 0.0,
            'scale_z': 0.0,

            'shift_x': 0.,
            'shift_y': 0.,
            'shift_z': 0.,

            # 'shear_a': 0.,
            # 'shear_b': 0.,
            # 'shear_c': 0.,

            'data_size': [*images_3D.size()],
            'forward_interp': 'bilinear',
            'backward_interp': 'bilinear'
        }, debug=True)
    augmentor.init_parameters()
    transformed = augmentor.forward(images_3D)
    recovered = augmentor.backward(transformed)
    mask = torch.ones_like(images_3D)
    with torch.no_grad():
        mask = augmentor.backward(augmentor.forward(mask))

    mask[mask == 1] = True
    mask[mask != 1] = False
    print('mask', mask)
    error = recovered - images_3D
    print('sum error', torch.sum(error))
    plt.subplot(331)
    plt.imshow(images_3D.cpu().numpy()[0, 0,0])

    plt.subplot(332)
    plt.imshow(transformed.cpu().numpy()[0, 0, 0 ])

    plt.subplot(333)
    plt.imshow(recovered.cpu().numpy()[0, 0,0])

    plt.subplot(334)
    plt.imshow(images_3D.cpu().numpy()[0,0,:,0])

    plt.subplot(335)
    plt.imshow(transformed.cpu().numpy()[0,0, :,0 ])

    plt.subplot(336)
    plt.imshow(recovered.cpu().numpy()[0,0,:,0,])

    plt.subplot(337)
    plt.imshow(images_3D.cpu().numpy()[0, 0,:,:,0])

    plt.subplot(338)
    plt.imshow(transformed.cpu().numpy()[0, 0, :,:,0 ])

    plt.subplot(339)
    plt.imshow(recovered.cpu().numpy()[0, 0,:,:,0])
    # plt.subplot(144)
    # plt.imshow(mask.cpu().numpy()[0,0])
    plt.savefig(join(dir_path, 'test_affine_3D.png'))
