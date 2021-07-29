from advchain.augmentor.adv_transformation_base import AdvTransformBase
import numpy as np
import torch.nn.functional as F
import torch

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def bspline_kernel_2d(sigma=[1, 1], order=3, asTensor=False, dtype=torch.float32, device='gpu'):
    '''
    generate bspline 2D kernel matrix for interpolation
    From wiki: https://en.wikipedia.org/wiki/B-spline, Fast b-spline interpolation on a uniform sample domain can be
    done by iterative mean-filtering
    :param sigma: tuple integers, control smoothness
    :param order: the order of interpolation, default=3
    :param asTensor: if true, return torch tensor rather than numpy array
    :param dtype: data type
    :param use_gpu: bool
    :return:
    '''
    kernel_ones = torch.ones(1, 1, *sigma)
    kernel = kernel_ones
    padding = np.array(sigma)

    for i in range(1, order + 1):
        kernel = F.conv2d(kernel, kernel_ones, padding=(
            i * padding).tolist()) / ((sigma[0] * sigma[1]))

    if asTensor:
        return kernel[0, 0, ...].to(dtype=dtype, device=device)
    else:
        return kernel[0, 0, ...].numpy()


class AdvBias(AdvTransformBase):
    "Adv Bias"

    def __init__(self,
                 config_dict={
                     'epsilon': 0.3,  # magnitude, (0,1)
                     # spacings between two control points along the x- and y- direction.
                     'control_point_spacing': [64, 64],
                     # downscale images to speed up the computation.
                     'downscale': 2,
                     # [ns,ch,h,w], change it to your tensor size
                     'data_size': [2, 1, 128, 128],
                     'interpolation_order': 3,  # b-spline interpolation order
                     'init_mode': 'random',  # uniform sampling or sample from a gaussian distribution
                     'space': 'log'},  # generate it in the log space rather than image space, bias =exp(bspline(cpoints)) other wise bias =1+ bspline(cpoints)
                 # perform power iteration to find the saddle points like virtual adversarial training
                 power_iteration=False,
                 use_gpu=True, debug=False):
        """[adv bias field augmentation]

        Args:
            config_dict (dict, optional): [description]. Defaults to { 'epsilon':0.3, 'control_point_spacing':[32,32], 'downscale':2, 'data_size':[2,1,128,128], 'interpolation_order':3, 'init_mode':'random', 'space':'log'}.
            power_iteration (bool, optional): [description]. Defaults to False.
            use_gpu (bool, optional): [description]. Defaults to True.
            debug (bool, optional): [description]. Defaults to False.
        """
        super(AdvBias, self).__init__(
            config_dict=config_dict, use_gpu=use_gpu, debug=debug)
        self.param = None
        self.power_iteration = power_iteration

    def init_config(self, config_dict):
        '''
        initialize a set of transformation configuration parameters
        '''
        self.epsilon = config_dict['epsilon']
        self.xi = 1e-6
        self.data_size = config_dict['data_size']
        self.downscale = config_dict['downscale']
        assert self.downscale <= min(
            self.data_size[2:]), 'downscale factor is too  large'
        self.control_point_spacing = [
            i//self.downscale for i in config_dict['control_point_spacing']]
        if (sum(self.control_point_spacing) > sum([40]*len(self.control_point_spacing))):
            logging.warning(
                'control point spacing may be too large, please increase the downscale factor.')
        self.interpolation_order = config_dict['interpolation_order']

        self.space = config_dict['space']
        self.init_mode = config_dict['init_mode']

    def init_parameters(self):
        '''
        initialize transformation parameters
        return random transformaion parameters
        '''
        self.init_config(self.config_dict)
        self._device = 'cuda' if self.use_gpu else 'cpu'

        self._dim = len(self.control_point_spacing)
        self.spacing = self.control_point_spacing
        self._dtype = torch.float32
        self.batch_size = self.data_size[0]
        self._image_size = np.array(self.data_size[2:])
        self.magnitude = self.epsilon
        self.order = self.interpolation_order
        self.downscale = self.downscale  # reduce image size to save memory

        self.use_log = True  # if self.space == 'log' else False

        # contruct and initialize control points grid with random values
        self.param, self.interp_kernel, self.bias_field = self.init_bias_field()
        return self.param

    def train(self):
        self.is_training = True
        if self.power_iteration:
            self.param = self.unit_normalize(self.param.data)
        self.param = torch.nn.Parameter(self.param.data, requires_grad=True)

    def rescale_parameters(self):
        # restrict control points values in the 1-ball space
        self.param = self.unit_normalize(self.param, p_type='l2')

    def optimize_parameters(self, step_size=0.3):
        if self.power_iteration:
            grad = self.unit_normalize(self.param.grad, p_type='l2')
            self.param = grad.clone().detach()
        else:
            grad = self.unit_normalize(self.param.grad, p_type='l2')
            # Gradient ascent
            self.param = self.param + step_size*grad.detach()
            self.param = self.param.clone().detach()
        return self.param

    def set_parameters(self, param):
        self.param = param.detach()

    def forward(self, data):
        '''
        forward the data to get transformed data
        :param data: input images x, N4HW
        :return:
        tensor: transformed images
        '''
        assert self.param is not None, 'init param before transform data'
        if self.power_iteration and self.is_training:
            bias_field = self.compute_smoothed_bias(self.xi*self.param)
        else:
            bias_field = self.compute_smoothed_bias(self.param)

        self.bias_field = bias_field
        self.diff = bias_field

        # in case the input image is a multi-channel input.
        if bias_field.size(1) < data.size(1):
            bias_field = bias_field.expand(data.size())

        transformed_input = bias_field*data
        if self.debug:
            print('bias transformed', transformed_input.size())
        return transformed_input

    def backward(self, data):
        if self.debug:
            print('max magnitude', torch.max(
                torch.abs(self.bias_field-1)))
        return data

    def predict_forward(self, data):
        return data

    def predict_backward(self, data):
        return data

    def init_bias_field(self, init_mode=None):
        '''
        init cp points, interpolation kernel, and  corresponding bias field.
        :param batch_size:
        :param spacing: tuple of ints
        :param order:
        :return:bias field
        reference:
        bspline interpoplation is adapted from airlab: class _KernelTransformation(_Transformation):
https://github.com/airlab-unibas/airlab/blob/1a715766e17c812803624d95196092291fa2241d/airlab/transformation/pairwise.py
        '''
        if init_mode is None:
            mode = self.init_mode

        # set up cpoints grid
        self._stride = np.array(self.spacing)
        cp_grid = np.ceil(np.divide(
            self._image_size/(1.0*self.downscale), self._stride)).astype(dtype=int)
        # new image size after convolution
        inner_image_size = np.multiply(
            self._stride, cp_grid) - (self._stride - 1)
        # add one control point outside each side, e.g.2 by 2 grid, requires 4 by 4 control points
        cp_grid = cp_grid + 2
        # image size with additional control points
        new_image_size = np.multiply(
            self._stride, cp_grid) - (self._stride - 1)
        # center image between control points
        image_size_diff = inner_image_size - \
            self._image_size/(1.0*self.downscale)
        image_size_diff_floor = np.floor(
            (np.abs(image_size_diff)/2))*np.sign(image_size_diff)
        self._crop_start = image_size_diff_floor + \
            np.remainder(image_size_diff, 2)*np.sign(image_size_diff)
        self._crop_end = image_size_diff_floor
        self.cp_grid = [self.batch_size, 1] + cp_grid.tolist()

        # initialize control points parameters for optimization
        if mode == 'gaussian':
            self.param = torch.ones(*self.cp_grid).normal_(mean=0, std=0.5)
        elif mode == 'random':
            self.param = (torch.rand(*self.cp_grid)*2-1) * \
                self.epsilon  # -eps,eps

        elif mode == 'identity':
            # static initialization, bias free
            self.param = torch.zeros(*self.cp_grid)
        else:
            raise NotImplementedError

        self.param = self.unit_normalize(self.param, p_type='l2')
        self.param = self.param.to(dtype=self._dtype, device=self._device)

        # convert to integer
        self._stride = self._stride.astype(dtype=int).tolist()
        self._crop_start = self._crop_start.astype(dtype=int)
        self._crop_end = self._crop_end.astype(dtype=int)

        size = [self.batch_size, 1] + new_image_size.astype(dtype=int).tolist()
        # initialize interpolation kernel
        self.interp_kernel = self.get_bspline_kernel(
            order=self.order, spacing=self.spacing)
        self.interp_kernel = self.interp_kernel.to(self.param.device)
        self.bias_field = self.compute_smoothed_bias(
            self.param, padding=self._padding, stride=self._stride)
        if self.debug:
            print('initialize {} control points'.format(
                str(self.param.size())))

        return self.param, self.interp_kernel, self.bias_field

    def compute_smoothed_bias(self, cpoint=None, interpolation_kernel=None, padding=None, stride=None):
        '''
        generate bias field given the cppints N*1*k*l
        :return: bias field bs*1*H*W
        '''
        if interpolation_kernel is None:
            interpolation_kernel = self.interp_kernel
        if padding is None:
            padding = self._padding
        if stride is None:
            stride = self._stride
        if cpoint is None:
            cpoint = self.param
        bias_field = F.conv_transpose2d(cpoint, interpolation_kernel,
                                        padding=padding, stride=stride, groups=1)
        # crop bias
        bias_field_tmp = bias_field[:, :,
                                    stride[0] + self._crop_start[0]:-stride[0] - self._crop_end[0],
                                    stride[1] + self._crop_start[1]:-stride[1] - self._crop_end[1]]

        # recover bias field to original image resolution for efficiency.
        if self.debug:
            logger.info('after intep, size:', bias_field_tmp.size())
        scale_factor_h = self._image_size[0] / bias_field_tmp.size(2)
        scale_factor_w = self._image_size[1] / bias_field_tmp.size(3)

        if scale_factor_h > 1 or scale_factor_w > 1:
            upsampler = torch.nn.Upsample(scale_factor=(scale_factor_h, scale_factor_w), mode='bilinear',
                                          align_corners=False)
            diff_bias = upsampler(bias_field_tmp)
        else:
            diff_bias = bias_field_tmp

        bias_field = torch.exp(diff_bias)
        bias_field = self.clip_bias(bias_field, self.epsilon)
        return bias_field

    def clip_bias(self, bias_field, magnitude=None):
        """[summary]
        clip the bias field so that it values fall in [1-magnitude, 1+magnitude]
        Args:
            bias_field ([torch 4d tensor]): [description]
            magnitude ([scalar], optional): [description]. Defaults to use predefined value.

        Returns:
            [type]: [description]
        """
        if magnitude is None:
            magnitude = self.magnitude
        assert magnitude > 0

        # bias_field =1+magnitude*self.unit_normalize(bias_field-1, p_type ='Infinity')
        bias = bias_field-1
        bias_field = 1+torch.clamp(bias, -magnitude, magnitude)
        if self.debug:
            print('max |bias-id|', torch.max(torch.abs(bias_field-1)))
        return bias_field

    def get_bspline_kernel(self, spacing, order=3):
        '''

        :param order init: bspline order, default to 3
        :param spacing tuple of int: spacing between control points along h and w.
        :return:  kernel matrix
        '''
        self._kernel = bspline_kernel_2d(
            spacing, order=order, asTensor=True, dtype=self._dtype, device=self._device)
        self._padding = (np.array(self._kernel.size()) - 1) / 2
        self._padding = self._padding.astype(dtype=int).tolist()
        self._kernel.unsqueeze_(0).unsqueeze_(0)
        self._kernel = self._kernel.to(dtype=self._dtype, device=self._device)
        return self._kernel

    def get_name(self):
        return 'bias'

    def is_geometric(self):
        return 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from os.path import join as join

    from advchain.common.utils import check_dir
    dir_path = './log'
    check_dir(dir_path, create=True)
    images = torch.ones(2, 1, 128, 128).cuda()
    images[:, :, ::2, ::2] = 2.0
    images[:, :, ::3, ::3] = 3.0
    images[:, :, ::1, ::1] = 1.0
    images = images.float()
    images.requires_grad = False
    print('input:', images)
    augmentor = AdvBias(
        config_dict={'epsilon': 0.3,
                     'control_point_spacing': [32, 32],
                     'downscale': 2,
                     'data_size': [2, 1, 128, 128],
                     'interpolation_order': 3,
                     'init_mode': 'gaussian',
                     'space': 'log'},
        power_iteration=False,
        debug=True, use_gpu=True)

    # perform random bias field
    augmentor.init_parameters()
    transformed = augmentor.forward(images)
    error = transformed-images
    print('sum error', torch.sum(error))

    plt.subplot(131)
    plt.imshow(images.detach().cpu().numpy()[0, 0])

    plt.subplot(132)
    plt.imshow(transformed.detach().cpu().numpy()[0, 0])

    plt.subplot(133)
    plt.imshow((transformed/images).detach().cpu().numpy()[0, 0])
    plt.savefig(join(dir_path, 'test_bias.png'))
