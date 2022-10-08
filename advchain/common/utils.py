import logging

import os
import torch
import contextlib
import numpy as np
import SimpleITK as sitk
import random
from pathlib import Path

from advchain.common.layers import Fixable2DDropout, Fixable3DDropout

def check_dir(dir_path, create=False):
    '''
    check the existence of a dir, when create is True, will create the dir if it does not exist.
    dir_path: str.
    create: bool
    return:
    exists (1) or not (-1)
    '''
    if os.path.exists(dir_path):
        return 1
    else:
        if create:
            os.makedirs(dir_path)
        return -1


def load_image_label(image_path, label_path=None, slice_id=0, crop_size=(192, 192)):
    """[summary]
    loads image and labels (optional) from disk (Nifti, nrrd)
    return cropped 3dim numpy array  [DHW] 
    Args:
        image_path ([type]): [description]
        label_path ([type]): [description]
        slice_id (int, optional): [description]. Defaults to 0.
        crop_size (tuple, optional): [description]. Defaults to (192, 192).
    Returns:
        [type]: [description]
    """
    support_formats = ['.nrrd', '.nii', '.nii.gz']
    assert Path(image_path).suffix in support_formats, 'only support loading images/labels with extensions:{}.'.format(
        str(support_formats))
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    if slice_id >= 0:
        image =image[slice_id]
        h_ind = 0
        w_ind =1
    else: 
        h_ind = 1
        w_ind = 2
    # print('image size:', image.shape)

    h_diff = (image.shape[h_ind]-crop_size[0])//2
    w_diff = (image.shape[w_ind]-crop_size[1])//2
    if slice_id >= 0:
        cropped_image = image[h_diff:crop_size[0] +
                            h_diff, w_diff:crop_size[1]+w_diff]
    else:cropped_image = image[:,h_diff:crop_size[0] +
                            h_diff, w_diff:crop_size[1]+w_diff]
    # rescale image intensities to 0-1
    cropped_image = (cropped_image-cropped_image.min()) / \
        (cropped_image.max()-cropped_image.min()+1e-10)

    if label_path is not None:
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        if slice_id>=0: label = label[slice_id]
        # logging.info('label size:', label.shape)
        assert image.shape == label.shape, "The sizes of the input image and label do not match, image:{}label:{}".format(
            str(image.shape), str(label.shape))
        if slice_id >= 0:
            cropped_label = label[h_diff:crop_size[0] +
                                h_diff, w_diff:crop_size[1]+w_diff]
        else:
            cropped_label = label[:,h_diff:crop_size[0] +
                                h_diff, w_diff:crop_size[1]+w_diff]

        return cropped_image, cropped_label
    else:
        return cropped_image

def rescale_intensity(data, new_min=0, new_max=1, eps=1e-20):
    '''
    rescale pytorch batch data
    :param data: N*1*H*W
    :return: data with intensity ranging from 0 to 1
    '''
    bs, c, h, w = data.size(0), data.size(1), data.size(2), data.size(3)
    data = data.view(bs * c, -1)
    old_max = torch.max(data, dim=1, keepdim=True).values
    old_min = torch.min(data, dim=1, keepdim=True).values
    new_data = (data - old_min) / (old_max - old_min + eps) * \
        (new_max - new_min) + new_min
    new_data = new_data.view(bs, c, h, w)
    return new_data


def check_dir(dir_path, create=False):
    '''
    check the existence of a dir, when create is True, will create the dir if it does not exist.
    dir_path: str.
    create: bool
    return:
    exists (1) or not (-1)
    '''
    if os.path.exists(dir_path):
        return 1
    else:
        if create:
            os.makedirs(dir_path)
        return -1


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(model, new_state=None, hist_states=None):
        """[summary]

        Args:
            model ([torch.nn.Module]): [description]
            new_state ([bool], optional): [description]. Defaults to None.
            hist_states ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        # please upgrade torch version to the latest due to the bug reported in https://github.com/pytorch/pytorch/issues/37823
        old_states = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm3d):
                # print('here batch norm')
                old_states[name] = module.track_running_stats
                # old_state = module.track_running_stats
                if hist_states is not None:
                    module.track_running_stats = hist_states[name]
                else:
                    if new_state is not None:
                        module.track_running_stats = new_state
              
            if isinstance(module, Fixable2DDropout) or isinstance(module, Fixable3DDropout):
                old_state = module.lazy_load ## freeze dropout to make the computation graph static
                module.lazy_load = not old_state
        return old_states

    old_states = switch_attr(model, new_state=False)
    yield
    switch_attr(model, hist_states=old_states)

@contextlib.contextmanager
def _fix_dropout(model):
    def switch_attr(model):
        """[summary]

        Args:
            model ([torch.nn.Module]): [description]
            new_state ([bool], optional): [description]. Defaults to None.
            hist_states ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        old_states = {}
        for name, module in model.named_modules():
            if isinstance(module, Fixable2DDropout) or isinstance(module, Fixable3DDropout):
                old_state = module.lazy_load ## freeze dropout to make the computation graph static
                module.lazy_load = not old_state
                old_states [name] = old_state
        return old_states

    old_states = switch_attr(model)
    # print('fix dropout', old_states)
    yield
    switch_attr(model)

def set_grad(module, requires_grad=False):
    for p in module.parameters():  # reset requires_grad
        p.requires_grad = requires_grad


def random_chain(alist, max_length=None, size_list=None):
    """[select a sub list from  a list and chained them in a random order]

    Args:
        list ([type]): [description]
    """
    length = len(alist)
    if max_length is None:
        max_length = length
    else:
        max_length  = min(max_length,length)
    assert length >= 1, "input list must contains at least one element"
    if length == 1:
        results = [alist[0]]
        if len(args) > 0:
            for arg in args:
                assert len(arg) == 1, 'must share equal size'
                results.append(arg)

            return results
        else:
            return results[0]

    sub_len = np.random.randint(low=1, high=max_length+ 1)

    r = random.random()            # randomly generating a real in [0,1)
    # lambda : r is an unary function which returns r
    random.shuffle(alist, lambda: r)
    # using the same function as used in prev line so that shuffling order is the same
    if size_list is not None and len(size_list) >= 0:
        random.shuffle(size_list, lambda: r)
        return alist[:sub_len], size_list[:sub_len]
    return alist[:sub_len]
