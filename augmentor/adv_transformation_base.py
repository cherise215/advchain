

import torch


class AdvTransformBase(object):
    """
     Adv Transformer base
    """

    def __init__(self,
                 config_dict={
                 },
                 use_gpu=True,
                 debug=False):
        '''


        '''

        self.config_dict = config_dict
        self.param = None
        self.is_training = False
        self.use_gpu = use_gpu
        self.debug = debug
        self.diff = None
        # by default this is False
        self.is_training = False
        if self.use_gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.init_config(self.config_dict)
        self.param = None
        self.step_size = 1  # step size for optimizing data augmentation

    def init_config(self):
        '''
        initialize a set of transformation configuration parameters
        '''
        if self.debug:
            print('init base class')
        raise NotImplementedError

    def init_parameters(self):
        '''
        initialize transformation parameters
        return random transformaion parameters
        '''
        raise NotImplementedError

    def set_parameters(self, param):
        self.param = param.detach().clone()

    def get_parameters(self):
        return self.param

    def set_step_size(self, step_size=1):
        self.step_size = step_size

    def get_step_size(self):
        return self.step_size

    def train(self):
        self.is_training = True
        self.param = self.param.detach().clone()
        self.param.requires_grad = True

    def eval(self):
        self.param.requires_grad = False
        self.is_training = False

    def rescale_parameters(self):
        raise NotImplementedError

    def optimize_parameters(self, step_size=None):
        raise NotImplementedError

    def forward(self, data):
        '''
        forward the data to get augmented data
        :param data: input images x, N4HW
        :return:
        tensor: augmented images
        '''
        raise NotImplementedError

    def backward(self, data):
        """[warps images back to its  original image coordinates if this is a geometric transformation ]

        Args:
            data ([torch tensor]): [input]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def predict_forward(self, data):
        """[summary]
        transforms predictions using the corresponding transformation matrix if this is a geometric transformation ]
        Args:
            data ([torch tensor]): [input]

        Returns:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def predict_backward(self, data):
        """[warps predictions back to its  original image coordinates if this is a geometric transformation ]

        Args:
            data ([torch tensor]): [input]

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def unit_normalize(self, d, p_type='l2'):
        """[summary]
        Performs normalization on batch vectors
        Args:
            d ([torch tensor]): [input vectors]: [NC(D)HW]
            p_type (str, optional): [specify normalization types: 'l1','l2','infinity']. Defaults to 'l2'.

        Returns:
            [type]: [description]
        """
        old_size = d.size()
        d_flatten = d.view(d.size(0), -1)
        if p_type == 'l1':
            norm = d_flatten.norm(p=1, dim=1, keepdim=True)
            d = d_flatten.div(norm.expand_as(d_flatten))
        elif p_type == 'infinity':

            d_abs_max = torch.max(d_flatten, 1, keepdim=True)[
                0].expand_as(d_flatten)
            # print(d_abs_max.size())
            d = d_flatten/(1e-20 + d_abs_max)  # d' =d/d_max

        elif p_type == 'l2':
            l = len(d.shape) - 1
            d_norm = torch.norm(
                d.view(d.shape[0], -1), dim=1).view(-1, *([1]*l))
            d = d / (d_norm + 1e-20)
        return d.view(old_size)

    def rescale_intensity(self, data, new_min=0, new_max=1, eps=1e-20):
        '''
        rescale pytorch batch data
        :param data: N*1*H*W
        :return: data with intensity ranging from 0 to 1
        '''
        bs, c, h, w = data.size(0), data.size(1), data.size(2), data.size(3)
        flatten_data = data.view(bs*c, -1)
        old_max = torch.max(flatten_data, dim=1, keepdim=True).values
        old_min = torch.min(flatten_data, dim=1, keepdim=True).values
        new_data = (flatten_data - old_min+eps) / \
            (old_max - old_min + eps)*(new_max-new_min)+new_min
        new_data = new_data.view(bs, c, h, w)
        return new_data

    def get_name(self):
        '''
        return the name of this transformation
        '''
        raise NotImplementedError

    def is_geometric(self):
        """[summary]
        Returns 1 if this is a geometric transformation, default, 0.
        """
        return 0

    def rescale_parameters(self, param=None):
        if param is None:
            param = self.param
        self.param = param.renorm(p=2, dim=0, maxnorm=self.epsilon)
        return self.param
