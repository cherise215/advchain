import logging
import numpy as np
import torch
import torch.nn.functional as F

from advchain.common.loss import calc_segmentation_consistency  # noqa
from advchain.common.utils import _disable_tracking_bn_stats,_fix_dropout


class ComposeAdversarialTransformSolver(object):
    """
    apply a chain of transformation
    """

    def __init__(self, chain_of_transforms=[], divergence_types=['mse', 'contour'],
                 divergence_weights=[1.0, 0.5], use_gpu=True,
                 debug=False,
                 if_norm_image=False,
                 is_gt=False,
                 ):
        '''
        adversarial data augmentation solver
        #TODO: implement class-aware consistency loss for segmentation tasks.
        '''
        self.chain_of_transforms = chain_of_transforms
        self.use_gpu = use_gpu
        self.debug = debug
        self.divergence_weights = divergence_weights
        self.divergence_types = divergence_types
        self.require_bi_loss = self.if_contains_geo_transform()
        self.if_norm_image = if_norm_image
        self.is_gt = is_gt
        self.class_weights = None

    
    def train(self):
        assert self.self.chain_of_transforms is not None and len(self.chain_of_transforms)>=1, 'chain_of_transforms is None or empty'
        for transform in self.chain_of_transforms:
            transform.train()
    
    def eval(self):
        assert self.chain_of_transforms is not None and len(self.chain_of_transforms)>=1, 'chain_of_transforms is None or empty'
        for transform in self.chain_of_transforms:
            transform.eval()
    
    def adversarial_training(self, data, model,
                             optimize_flags=None,
                             init_output=None,
                             lazy_load=False,
                             power_iteration=False,
                             n_iter=1,
                             step_sizes=None,
                             ):
        """
        given a batch of images: NCHW, and a current segmentation model
        find optimized transformations 
        return the adversarial consistency loss for network training
        Args:
            data (torch 4d tensor): input images
            model (torch.nn.Module): segmentation model
            optimize_flags (list of boolean, optional): if set to true, will optimize corresponding the params in the corresponding transformation function.  Defaults to None. If n_iter>0, will set to be true for the chain,  otherwise, false.
            init_output(torch 4d tensor],optional):network predictions on input images using the current model. Defaults to None. 
            lazy_load (bool, optional): if true, if will use previous random parameters (if have been initialized). Defaults to False.
            power_iteration (list of boolean, optional): if set to true, will perform power iteration to update the transformation parameters, see virtual adversarial training for details. Defaults to 'smart'. if set to 'smart', will apply power iteration for noise generation (as what has been done in VAT)
                                                             while projected gradient descent (PGD) to the others. If set to False, will perform basic PGD to all transformations.
            n_iter (int, optional): innner iterations to optimize data augmentation. Defaults to 1.

            step_sizes(float or a  list of float, optional): initial  step size for update. Defaults to 1. During optimization, the step size for iteration t will be decreased for stabilizing the inner optimization, using the schedule: 1/sqrt(t+1)*step_size.  
        Raises:
            NotImplementedError: [check whether the string for specifying optimization_mode is valid]

        Returns:
            dist [loss 1d tensor]: [adv consistency loss for network regularisation]
        """
        '''
     
        '''
        # 1. initialization
        # set up optimization mode:  whether to update transformation parameters or nots
        if optimize_flags is not None:
            assert len(self.chain_of_transforms) == len(
                optimize_flags), f'must specify each transform is learnable or not, expect {len(self.chain_of_transforms)} flags, but got {optimize_flags}'
        else:
            if n_iter == 0:
                optimize_flags = [False] * len(self.chain_of_transforms)
            elif n_iter > 0:
                optimize_flags = [True] * len(self.chain_of_transforms)
            else:
                raise NotImplementedError
        # set up optimization alg.
        if isinstance(power_iteration, bool):
            power_iterations = [power_iteration]*len(self.chain_of_transforms)
        elif isinstance(power_iteration, list):
            assert len(self.chain_of_transforms) == len(
                power_iteration), 'must specify each transform optimization mode'
            power_iterations = power_iteration

        elif isinstance(power_iteration, str):
            if "smart" == power_iteration:
                power_iterations = []
                for i, transform in enumerate(self.chain_of_transforms):
                    # use power_iteration for adv noise generation, following VAT
                    power = True if transform.get_name() == 'noise' else False
                    power_iterations.append(power)
        for i, power_iteration in enumerate(power_iterations):
            self.chain_of_transforms[i].power_iteration = power_iteration

        # set up step size for the first iteration.
        if step_sizes is None:
            logging.info('use default step size: 1 for every transformation')
            step_sizes = [1]*len(self.chain_of_transforms)
        else:
            if isinstance(step_sizes, float) or isinstance(step_sizes, int):
                logging.info(
                    'set the same step size:{} for every transformation'.format(step_sizes))
                step_sizes = [step_sizes]*len(self.chain_of_transforms)

            elif isinstance(step_sizes, list):
                assert len(step_sizes) == len(
                    self.chain_of_transforms), 'specify step size for each transformation'
            else:
                raise ValueError(
                    'please use scalar or a  list of scalar to set step size')
        # 2. get reference predictions f(x)
        if init_output is None:
            init_output = self.get_init_output(data=data, model=model)

        # 3. optimize transformation to maxmize the difference between f(x) and f(t(x))
        self.init_random_transformation(lazy_load)
        if n_iter == 1 or n_iter > 1:
            optimized_transforms = self.optimizing_transform(
                data=data, model=model, init_output=init_output, n_iter=n_iter, optimize_flags=optimize_flags, step_sizes=step_sizes)

            self.chain_of_transforms = optimized_transforms
        else:
            pass
        # 4. augment data with optimized transformation t, and calc the adversarial consistency loss with the composite transformation
        dist, adv_data, adv_output, warped_back_adv_output = self.calc_adv_consistency_loss(
            data.detach().clone(), model, init_output=init_output, chain_of_transforms=self.chain_of_transforms)

        self.init_output = init_output
        self.warped_back_adv_output = warped_back_adv_output
        self.origin_data = data
        self.adv_data = adv_data
        self.adv_predict = adv_output
        if self.debug:
            print('[outer loop] loss', dist.item())
        return dist

    def train():
        if self.chain_of_transforms is None:
            raise ValueError('please initialize the transformation chain first')
        else:
            for tr in self.chain_of_transforms:
                tr.train()
    def eval():
        if self.chain_of_transforms is None:
            raise ValueError('please initialize the transformation chain first')
        else:
            for tr in self.chain_of_transforms:
                tr.eval()

    def forward(self, data, chain_of_transforms=None):
        '''
        forward the data to get transformed data
        :param data: input images x, NCHW
        :return:
        tensor: transformed images, NCHW
        '''
        data.requires_grad = False
        bs= data.size(0)
        flatten_data = data.reshape(bs, -1)
        original_min = torch.min(flatten_data, dim=1, keepdim=True).values
        original_max = torch.max(flatten_data, dim=1, keepdim=True).values
        t_data = data.detach().clone()
        self.diffs = []
        if chain_of_transforms is None:
            chain_of_transforms = self.chain_of_transforms
        is_training = False
        for transform in chain_of_transforms:
            t_data = transform.forward(t_data)
            self.diffs.append(transform.diff)
            is_training = (is_training or transform.is_training)
        if self.if_norm_image:
            t_data = self.rescale_intensity(t_data, original_min, original_max)
        return t_data

    def predict_forward(self, data, chain_of_transforms=None):
        '''
        transform the prediction with the learned/random data augmentation, only applies to geomtric transformations.
        :param data: input images x, NCHW
        :return:
        tensor: transformed images, NCHW
        '''
        self.diffs = []
        if chain_of_transforms is None:
            chain_of_transforms = self.chain_of_transforms
        for transform in chain_of_transforms:
            data = transform.predict_forward(data)
            self.diffs.append(transform.diff)
        return data

    def backward(self, data, chain_of_transforms=None):
        '''
        warp it back to image space
        only activate when the augmentation is a geometric transformation
        '''
        if chain_of_transforms is None:
            chain_of_transforms = self.chain_of_transforms
        for transform in reversed(chain_of_transforms):
            data = transform.backward(data)
        return data

    def predict_backward(self, data, chain_of_transforms=None):
        '''
        warp it back to image space
        only activate when the augmentation is a geometric transformation
        '''
        if chain_of_transforms is None:
            chain_of_transforms = self.chain_of_transforms
        for transform in reversed(chain_of_transforms):
            data = transform.predict_backward(data)
        return data

    def loss_fn(self, pred, reference, mask=None):
        """[compute the inconsistency between two predictions (in the same coordinates)

        Args:
            pred ([torch tensor]): 4-dim output
            reference ([torch tensor]): 4-dim reference
            mask ([torch tensor], optional): 4-dim mask with 0-1, indicating which element (mask=0) should be ignored when computing loss. Defaults to None.
        Returns:
            loss [torch tensor]: a scalar.
        """
        scales = [0]
        loss = calc_segmentation_consistency(output=pred, reference=reference, divergence_types=self.divergence_types, divergence_weights=self.divergence_weights, scales=scales, mask=mask, class_weights=self.class_weights,
                                             is_gt=self.is_gt)
        return loss

    def calc_adv_consistency_loss(self, data, model, init_output, chain_of_transforms=None):
        """[summary]  
        calc adversarial consistency loss with adversarial data augmentation 

        Args:
            data ([torch 4d tensor]): a batch of clean images
            model ([torch.nn.Module]):segmentation model
            init_output ([torch 4d tensor]): predictions on clean images (before softmax)
            chain_of_transforms ([list of adversarial image transformation], optional): [description].
             Defaults to None. use self.chain_of_transform

        Returns:
            loss [torch.tensor]: The consistency loss  
        """
        if chain_of_transforms is None:
            chain_of_transforms = self.chain_of_transforms
        for tr in chain_of_transforms:
            tr.eval()
        adv_data = self.forward(data, chain_of_transforms)
        torch.cuda.empty_cache()
        # set_grad(model, requires_grad=True)
        old_state = model.training
        model.train()
        with  _fix_dropout(model):
            adv_output = self.get_net_output(model,adv_data.detach().clone())
        
        if self.if_contains_geo_transform(chain_of_transforms):
            masks = torch.ones_like(
                init_output, dtype=init_output.dtype, device=init_output.device,requires_grad=False)
            forward_mask =  self.predict_forward(masks, chain_of_transforms)
            forward_backward_mask = self.predict_backward(forward_mask, chain_of_transforms)
            warped_back_adv_output = self.predict_backward(
                adv_output, chain_of_transforms)
            forward_backward_mask[forward_backward_mask != 0] = 1
            dist = self.loss_fn(pred=warped_back_adv_output, reference=init_output.detach(
            ), mask=forward_backward_mask)

  
        else:
            # no geomtric transformation
            warped_back_adv_output = adv_output
            dist = self.loss_fn(
                pred=adv_output, reference=init_output.detach())
        model.train(old_state)
        return dist, adv_data, adv_output, warped_back_adv_output

    def optimizing_transform(self, model, data, init_output, optimize_flags, n_iter=1, step_sizes=None):
        # optimize each transform with one forward pass.
        for i in range(n_iter):
            torch.cuda.empty_cache()
            model.zero_grad()
            self.make_learnable_transformation(
                optimize_flags=optimize_flags, chain_of_transforms=self.chain_of_transforms)
            augmented_data = self.forward(data.detach().clone())
            with _disable_tracking_bn_stats(model):
                perturbed_output = self.get_net_output(model,augmented_data)
            # calc divergence loss
            if self.if_contains_geo_transform(self.chain_of_transforms):
                warped_back_prediction = self.predict_backward(
                    perturbed_output)
                masks = torch.ones_like(
                    init_output, dtype=init_output.dtype, device=init_output.device,requires_grad=False)
                forward_backward_mask = self.predict_backward(
                    self.predict_forward(masks))
                forward_backward_mask[forward_backward_mask != 0] = 1
                dist = self.loss_fn(
                    pred=warped_back_prediction, reference=init_output,mask=forward_backward_mask)

            else:
                dist = self.loss_fn(pred=perturbed_output,
                                    reference=init_output.detach())
            if self.debug:
                print('[inner loop], step {}: dist {}'.format(
                    str(i), dist.item()))
            dist.backward()
            i_tr = 0
            for flag, transform in zip(optimize_flags, self.chain_of_transforms):
                if flag:
                    if self.debug:
                        print('update {} parameters'.format(
                            transform.get_name()))
                    try:
                        step_size = step_sizes[i_tr]
                    except:
                        logging.warning('use default step size: 1')
                        step_size = transform.get_step_size()
                    transform.optimize_parameters(
                        step_size=step_size/np.sqrt((i+1)))
                i_tr += 1

            model.zero_grad()

        transforms = []
        for flag, transform in zip(optimize_flags, self.chain_of_transforms):
            if flag:
                transform.rescale_parameters()
            transform.eval()
            transforms.append(transform)
        return transforms

    def rescale_intensity(self, data, new_min=0, new_max=1, eps=1e-20):
        '''
        rescale pytorch batch data
        :param data: N*1*H*W
        :return: data with intensity ranging from 0 to 1
        '''
        old_size = data.size()
        bs= data.size(0)
        flatten_data = data.view(bs, -1)
        old_max = torch.max(flatten_data, dim=1, keepdim=True).values
        old_min = torch.min(flatten_data, dim=1, keepdim=True).values
        new_data = (flatten_data - old_min+eps) / \
            (old_max - old_min + eps)*(new_max-new_min)+new_min
        new_data = new_data.view(old_size)
        return new_data

    def get_net_output(self,model, data):
        '''
        set up network output function
        '''
        return model.forward(data)

    def get_init_output(self, model, data):
        with torch.no_grad():
            with _disable_tracking_bn_stats(model):
                reference_output = self.get_net_output(model,data)
        return reference_output

    def get_adv_data(self, data, model, init_output=None, n_iter=0):
        """
        given an input data and current segmentation model, return augmented input, and corresponding reference
        if init_output is now, use original prediction as pseudo labels.
        Args:
            data (torch tensor): input data
            model (torch.nn.Module): segmentation model
            init_output ([type], optional): [4-dim onehot labels or probs predictions]. Defaults to None.
            n_iter (int, optional): [description]. Defaults to 0.

        Returns:
            [augmented data]: torch tensor, 4-dim output,
            [correspondong labels for reference]: torch tensor, 4-dim output
        """
        if init_output is None:
            init_output = self.get_init_output(model, data)
        self.init_random_transformation()
        origin_data = data.detach().clone()
        if n_iter > 0:
            optimized_transforms = self.optimizing_transform(
                data=data, model=model, init_output=init_output, n_iter=n_iter, optimize_flags=[True]*len(self.chain_of_transforms))
        else:
            optimized_transforms = self.chain_of_transforms
        augmented_data = self.forward(origin_data, optimized_transforms)
        augmented_label = self.predict_forward(
            init_output, optimized_transforms)
        return augmented_data, augmented_label

    def if_contains_geo_transform(self, chain_of_transforms=None):
        """
        check if the predefined transformation contains geometric transform
        Returns:
            [boolean]: return True if geometric transformation is involved, otherwise false.
        """
        if chain_of_transforms is None:
            chain_of_transforms = self.chain_of_transforms
        sum_flag = 0

        for transform in chain_of_transforms:
            sum_flag += transform.is_geometric()
        return sum_flag > 0

    def init_random_transformation(self, lazy_load=False):
        # initialize transformation parameters
        '''
        randomly initialize random parameters
        return 
        list of random parameters, and list of random transform

        '''
        for transform in self.chain_of_transforms:
            if lazy_load:
                if transform.param is None:
                    transform.init_parameters()
            else:
                transform.init_parameters()

    def reset_transformation(self):
        self.init_random_transformation(lazy_load=False)

    def set_transformation(self, parameter_list):
        """
        set the values of transformations accordingly

        Args:
            parameter_list ([type]): [description]
        """
        # reset transformation parameters
        for i, param in enumerate(parameter_list):
            self.chain_of_transforms[i].set_parameters(param)

    def make_learnable_transformation(self, optimize_flags, chain_of_transforms=None):
        """[summary]
        make transformation parameters learnable
        Args:
            power_iterations ([boolean]): [description]
            chain_of_transforms ([list of adv transformation functions], optional): 
            [description]. Defaults to None. if not specified, use self.transformation instead
        """
        # reset transformation parameters
        if chain_of_transforms is None:
            chain_of_transforms = self.chain_of_transforms
        for flag, transform in zip(optimize_flags, chain_of_transforms):
            if flag:
                transform.train()
# if __name__ == "__main__":
#     import os
#     import torch
#     import torch.nn as nn
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     from skimage import data
#     import SimpleITK as sitk
#     import numpy as np
#     from os.path import join as join

#     from advchain.common.utils import check_dir, load_image_label
#     from advchain.augmentor import *

#     log_dir = "./log"
#     check_dir(log_dir, create=True)

#     sns.set(font_scale=1)

#     image_path = './example/data/cardiac/img.nrrd'
#     label_path = './example/data/cardiac/seg.nrrd'
#     slice_id = 5
#     crop_size = [192, 192]
#     cropped_image, cropped_label = load_image_label(
#         image_path, label_path, slice_id=slice_id, crop_size=crop_size)
#     images = torch.from_numpy(
#         cropped_image[np.newaxis, np.newaxis, :, :]).float()
#     images.cuda()
#     images.requires_grad = False
#     print('input', images)
#     # 2. set up data augmentation and its optimizer
#     augmentor_bias = AdvBias(
#         config_dict={'epsilon': 0.3,
#                      'control_point_spacing': [*images.size()//4, *images.size()//4],
#                      'downscale': 2,
#                      'data_size': [*images.size()],
#                      'interpolation_order': 3,
#                      'init_mode': 'random',
#                      'space': 'log'}, debug=True)

#     chain_of_transforms = [augmentor_bias]

#     # optimizer
#     power_iteration = False
#     n_iter = 1
#     composed_augmentor = ComposeAdversarialTransformSolver(
#         chain_of_transforms=chain_of_transforms,
#         divergence_types=['kl', 'contour'],
#         divergence_weights=[1.0, 0.5],
#         use_gpu=True,
#         debug=True,
#     )

#     # 3. set up  the segmentor
#     model = torch.nn.Conv2d(1, 4, 3, 1, 1)
#     # model = get_unet_model(num_classes=4,model_path='./result/UNet_16$SAX$_Segmentation.pth',model_arch='UNet_16')
#     model.cuda()

#     # 4. start learning
#     composed_augmentor.init_random_transformation()

#     # 4.1 get randomly augmented results for reference
#     rand_transformed_image = composed_augmentor.forward(images)
#     rand_predict = model.forward(rand_transformed_image)
#     # rand_predict = F.softmax(rand_predict,dim=1)
#     model.zero_grad()
#     rand_recovered_predict = composed_augmentor.predict_backward(rand_predict)
#     rand_recovered_image = composed_augmentor.backward(rand_transformed_image)
#     diff = rand_recovered_image-images

#     print('sum image diff', torch.sum(diff))

#     # 4.2 adv data augmentation
#     loss = composed_augmentor.adversarial_training(
#         data=images, model=model,
#         n_iter=n_iter, lazy_load=True, optimization_mode='chain',
#         optimize_flags=[True]*len(chain_of_transforms),
#         power_iteration=[power_iteration]*len(chain_of_transforms))
#     print('consistency loss', loss.item())

#     warped_back_adv_image = composed_augmentor.backward(
#         composed_augmentor.adv_data)
#     adv_predict = composed_augmentor.adv_predict
#     adv_recovered_predict = composed_augmentor.warped_back_adv_output
#     init_output = composed_augmentor.init_output

#     fig, axes = plt.subplots(2, 8)

#     axes[0, 0].imshow(images.detach().cpu().numpy()[0, 0],
#                       cmap='gray', interpolation=None)
#     axes[0, 0].set_title('Input')

#     axes[0, 1].imshow(composed_augmentor.adv_data.detach().cpu().numpy()[
#                       0, 0], cmap='gray', interpolation=None)
#     axes[0, 1].set_title('Transformed')

#     axes[0, 2].imshow(warped_back_adv_image.detach().cpu().numpy()[
#                       0, 0], cmap='gray', interpolation=None)
#     axes[0, 2].set_title('Recovered')

#     axes[0, 3].imshow(torch.argmax(adv_predict, dim=1).unsqueeze(
#         1).detach().cpu().numpy()[0, 0], cmap='gray', interpolation=None)
#     axes[0, 3].set_title('Adv Predict')

#     axes[0, 4].imshow(torch.argmax(adv_recovered_predict, dim=1).unsqueeze(
#         1).detach().cpu().numpy()[0, 0], cmap='gray', interpolation=None)
#     axes[0, 4].set_title('Recovered')

#     axes[0, 5].imshow(torch.argmax(init_output, dim=1).unsqueeze(
#         1).detach().cpu().numpy()[0, 0], cmap='gray', interpolation=None)
#     axes[0, 5].set_title('Original')

#     axes[0, 6].imshow((composed_augmentor.adv_data-images).detach().cpu().numpy()
#                       [0, 0], cmap='gray', interpolation=None)
#     axes[0, 6].set_title('diff')

#     axes[1, 0].imshow(images.detach().cpu().numpy()[0, 0],
#                       cmap='gray', interpolation=None)
#     axes[1, 0].set_title('Input')

#     axes[1, 1].imshow(rand_transformed_image.detach().cpu().numpy()[
#                       0, 0], cmap='gray', interpolation=None)
#     axes[1, 1].set_title('Rand ')

#     axes[1, 2].imshow(rand_recovered_image.detach().cpu().numpy()[
#                       0, 0], cmap='gray', interpolation=None)
#     axes[1, 2].set_title('Rand')

#     axes[1, 3].imshow(torch.argmax(rand_predict, dim=1).detach().cpu().numpy()[
#                       0], cmap='gray', interpolation=None)
#     axes[1, 3].set_title('Predict')
#     axes[1, 4].imshow(torch.argmax(rand_recovered_predict, dim=1).detach(
#     ).cpu().numpy()[0], cmap='gray', interpolation=None)
#     axes[1, 4].set_title('Recovered')
#     axes[1, 5].imshow(torch.argmax(init_output, dim=1).detach().cpu().numpy()[
#                       0], cmap='gray', interpolation=None)
#     axes[1, 5].set_title('Original')

#     axes[1, 6].imshow((rand_transformed_image-images).detach().cpu().numpy()
#                       [0, 0], cmap='gray', interpolation=None)

#     for ax in axes.ravel():
#         ax.set_axis_off()
#         ax.grid(False)
#     plt.tight_layout(w_pad=0, h_pad=0)

#     plt.savefig(join(dir_path, 'test_compose.png'))
#     plt.clf()
