
from advchain.augmentor.adv_transformation_base import AdvTransformBase
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import torch
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AdvAffine(AdvTransformBase):
    """
     Adv Affine
    """

    def __init__(self,
                 config_dict={
                     'rot': 15.0/180.0,
                     'scale_x': 0.2,
                     'scale_y': 0.2,
                     'shift_x': 0.1,
                     'shift_y': 0.1,
                     'data_size': [1, 1, 8, 8],
                     'forward_interp': 'bilinear',
                     'backward_interp': 'bilinear'
                 },
                 power_iteration=False,
                 use_gpu=True, debug=False):
        '''
        initialization
        '''
        super(AdvAffine, self).__init__(
            config_dict=config_dict, use_gpu=use_gpu, debug=debug)
        self.power_iteration = power_iteration

    def init_config(self, config_dict):
        '''
        initialize a set of transformation configuration parameters
        '''
        self.rot_ratio = config_dict['rot']
        self.translation_x = config_dict['shift_x']
        self.translation_y = config_dict['shift_y']
        self.scale_x = config_dict['scale_x']
        self.scale_y = config_dict['scale_y']
        self.xi = 1e-6
        self.data_size = config_dict['data_size']
        self.forward_interp = config_dict['forward_interp']
        self.backward_interp = config_dict['backward_interp']

    def init_parameters(self):
        '''
        initialize transformation parameters
        return random transformaion parameters
        '''
        self.init_config(self.config_dict)
        batch_size = self.data_size[0]
        affine_tensor = self.draw_random_affine_tensor_list(
            batch_size=batch_size)
        self.param = affine_tensor
        return affine_tensor

    def forward(self, data, padding_mode='zeros', interp=None):
        '''
        forward the data to get transformed data
        :param data: input images x, N4HW
        :return:
        tensor: transformed images
        '''

        assert self.param is not None, 'init param before transform data'
        if interp is None:
            interp = self.forward_interp
        if self.power_iteration and self.is_training:
            self.affine_matrix = self.gen_batch_affine_matrix(
                self.xi*self.param)
        else:
            self.affine_matrix = self.gen_batch_affine_matrix(self.param)

        transformed_input = self.transform(
            data, self.affine_matrix, interp=interp, padding_mode=padding_mode)
        self.diff = data-transformed_input

        if self.debug:
            logger.debug('afffine transformed', transformed_input.size())
        return transformed_input

    def predict_forward(self, data):
        return self.forward(data)

    def predict_backward(self, data):
        return self.backward(data)

    def backward(self, data, padding_mode='zeros'):
        assert not self.param is None, 'play forward before backward'
        inverse_matrix = self.get_inverse_matrix(self.affine_matrix)
        warped_back_output = self.transform(
            data, inverse_matrix, interp=self.backward_interp, padding_mode=padding_mode)
        return warped_back_output

    def draw_random_affine_tensor_list(self, batch_size, identity_init=False):
        if identity_init:
            affine_tensor = torch.zeros(
                batch_size, 5, device=self.device, dtype=torch.float32)
        else:
            ## initialization [-1,1]
            affine_tensor = (2*torch.rand(batch_size, 5,
                                          dtype=torch.float32, device=self.device)-1)
            affine_tensor = torch.nn.Hardtanh()(affine_tensor)

        return affine_tensor

    def optimize_parameters(self, step_size=None):
        if step_size is None:
            self.step_size = step_size
        if self.debug:
            # we assume that affine parameters are independent to each other.
            logger.info('optimize affine')

        if self.power_iteration:
            grad = self.param.grad.sign()
            self.param = grad.detach()
        else:
            grad = self.param.grad.sign().detach()
            param = self.param+step_size*grad
            self.param = param.detach()
        return self.param

    def set_parameters(self, param):
        self.param = param.detach()

    def rescale_parameters(self):
        return self.param

    def train(self):
        self.is_training = True
        if self.power_iteration:
            self.param = self.param.sign()
        self.param = torch.nn.Parameter(self.param, requires_grad=True)

    def gen_batch_affine_matrix(self, affine_tensors):
        '''
        given affine parameters, gen batch-wise affine_matrix [bs*2*3]
        :param affine_tensors:N*5 , [rot_radius, scale, tx,ty]
        :return:
         affine_matrix [bs*2*3]
        '''
        # restrict transformations between [-1,1]
        affine_tensors = torch.nn.Hardtanh()(affine_tensors)
        rot_radius, scalex, scaley, tx, ty = affine_tensors[:, 0], affine_tensors[:,
                                                                                  1], affine_tensors[:, 2], affine_tensors[:, 3], affine_tensors[:, 4]
        transformation_matrix = torch.stack([
            torch.stack([(1+scalex*self.scale_x) * (torch.cos(rot_radius*self.rot_ratio*math.pi)), (1+scaley *
                                                                                                    self.scale_y) * (-torch.sin(rot_radius*self.rot_ratio*math.pi)), tx*self.translation_x], dim=-1),
            torch.stack([(1+scalex*self.scale_x) * (torch.sin(rot_radius*self.rot_ratio*math.pi)), (1+scaley *
                                                                                                    self.scale_y) * (torch.cos(rot_radius*self.rot_ratio*math.pi)), ty*self.translation_y], dim=-1)
        ], dim=1)
        if self.use_gpu:
            transformation_matrix.cuda()
        return transformation_matrix

    def make_batch_eye_matrix(self, batch_size, device):
        O = torch.zeros(batch_size, dtype=torch.float32).to(device)
        I = torch.ones(batch_size, dtype=torch.float32).to(device)
        eyeMtrx = torch.stack([torch.stack([I, O, O], dim=-1),
                               torch.stack([O, I, O], dim=-1),
                               torch.stack([O, O, I], dim=-1)], dim=1)
        return eyeMtrx

    def transform(self, data, affine_matrix, interp='bilinear', padding_mode='zeros'):
        '''
        transform images with the given affine matrix
        '''
        grid_tensor = F.affine_grid(
            affine_matrix, data.size(), align_corners=True)
        transformed_image = F.grid_sample(
            data, grid_tensor, mode=interp, align_corners=True, padding_mode=padding_mode)
        return transformed_image

    def get_inverse_matrix(self, affine_matrix):
        homo_matrix = self.make_batch_eye_matrix(
            batch_size=affine_matrix.size(0), device=affine_matrix.device)
        homo_matrix[:, :2] = affine_matrix
        inverse_matrix = homo_matrix.inverse()
        inverse_matrix = inverse_matrix[:, :2, :]
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
    images = torch.zeros((1, 1, 8, 8)).cuda()
    images[:, :, 2:5, 2:5] = 1.0
    print('input:', images)
    augmentor = AdvAffine(debug=True)
    augmentor.init_parameters()
    transformed = augmentor.forward(images)
    recovered = augmentor.backward(transformed)
    mask = torch.ones_like(images)
    with torch.no_grad():
        mask = augmentor.backward(augmentor.forward(mask))

    mask[mask == 1] = True
    mask[mask != 1] = False
    print('mask', mask)
    error = recovered-images
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

    # adv test
