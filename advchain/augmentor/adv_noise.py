import logging
import torch
from advchain.augmentor.adv_transformation_base import AdvTransformBase  # noqa

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class AdvNoise(AdvTransformBase):
    """
     Adv Noise
    """

    def __init__(self,
                spatial_dims=2,
                config_dict={'epsilon': 0.1,
                              'xi': 1e-6,
                              'data_size': [10, 1, 8, 8]
                              },
                 power_iteration=False,
                 ignore_values = None,
                 use_gpu=True, debug=False, device=torch.device("cuda")):
        '''
        initialization
        ignore_values: ignore the regions with particular values in the data when perturbing. This applies to the senario that the data has been padded with a mask, and we do not want to perturb the mask.
        '''
        super(AdvNoise, self).__init__(spatial_dims=spatial_dims,
            config_dict=config_dict, use_gpu=use_gpu, debug=debug, device=device)
        self.power_iteration = power_iteration
        self.ignore_values = ignore_values ## out-of-image region

    def init_config(self, config_dict):
        '''
        initialize a set of transformation configuration parameters
        '''
        self.epsilon = config_dict['epsilon']
        self.xi = config_dict['xi']
        self.data_size = config_dict['data_size']

    def init_parameters(self):
        '''
        initialize transformation parameters
        return random transformaion parameters
        '''
        noise =self.unit_normalize(torch.randn(
                *self.data_size, device=self.device, dtype=torch.float32))
        self.param = noise
        return noise

    def optimize_parameters(self, step_size=None):
        if step_size is None:
            step_size = self.step_size
        if self.debug:
            logging.info('optimize noise')
        grad = self.unit_normalize(self.param.grad)
        if self.debug:
            print('grad', grad.size())
        if self.power_iteration:
            self.param = grad.detach()
        else:
            self.param = self.param+step_size*grad.detach()
            self.param = self.param.detach()
        return self.param


    def forward(self, data, **kwargs):
        '''
        forward the data to get transformed data
        :param data: input images x, N4HW
        :return:
        tensor: transformed images
        '''
        if self.debug:
            print('add noise')
        if self.param is None:
            self.init_parameters()
       
        if self.power_iteration and self.is_training:
            self.diff = self.xi*self.param
            transformed_input = data+self.xi*self.param
        else:
            self.diff= self.epsilon * self.param
            transformed_input = data+self.epsilon * self.param
        if self.ignore_values is not None:
            mask = abs(data-self.ignore_values)<1e-8
            mask = mask.detach().clone()
            transformed_input[mask]= self.ignore_values
        self.diff = transformed_input - data
        return transformed_input
    
    def rescale_parameters(self):
        #restrict noise in the 1-ball space
        self.param = self.unit_normalize(self.param, p_type='l2')

    def backward(self, data, **kwargs):
        if self.debug:
            print('noise back, no action:, maxium noise',
                  torch.max(self.diff))
        return data

    def predict_forward(self, data, **kwargs):
        return data

    def predict_backward(self, data, **kwargs):
        return data

    def train(self):
        self.is_training = True
        if self.param is None:
            self.init_parameters()
        if self.power_iteration:
            self.param = self.unit_normalize(self.param)
        self.param = torch.nn.Parameter(self.param, requires_grad=True)

    def get_name(self):
        return 'noise'



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from os.path import join as join

    from advchain.common.utils import check_dir

    dir_path = './log'
    check_dir(dir_path, create=True)
    images = torch.zeros((10, 1, 8, 8)).cuda()
    print('input:', images)
    augmentor = AdvNoise(debug=True)
    augmentor.init_parameters()
    transformed = augmentor.forward(images)
    recovered = augmentor.backward(transformed)
    error = recovered-images
    print('sum error', torch.sum(error))
    plt.subplot(131)
    plt.imshow(images.cpu().numpy()[0, 0])

    plt.subplot(132)
    plt.imshow(transformed.cpu().numpy()[0, 0])

    plt.subplot(133)
    plt.imshow(recovered.cpu().numpy()[0, 0])

    plt.savefig(join(dir_path, 'test_noise.png'))
    

    # 3D 
    images = torch.zeros((10, 1, 8, 8,8)).cuda()

    augmentor = AdvNoise(
                        spatial_dims=3,
                        config_dict={'epsilon': 0.1,
                                                'xi': 1e-6,
                                                'data_size': [10, 1, 8, 8,8]
                                                },
                 power_iteration=False,
                 use_gpu=True, debug=False)

    augmentor.init_parameters()
    augmented_image = augmentor.forward(images)