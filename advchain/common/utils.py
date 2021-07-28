import os
import torch
import contextlib
import numpy as np
import SimpleITK as sitk
import random

from models.unet import UNet


def load_image_label(image_path, label_path, slice_id=0, crop_size=(192, 192)):

    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))[slice_id]
    label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))[slice_id]

    print('image size:', image.shape)
    print('label size:', label.shape)
    h_diff = (image.shape[0]-crop_size[0])//2
    w_diff = (image.shape[1]-crop_size[1])//2

    cropped_image = image[h_diff:crop_size[0] +
                          h_diff, w_diff:crop_size[1]+w_diff]
    cropped_label = label[h_diff:crop_size[0] +
                          h_diff, w_diff:crop_size[1]+w_diff]

    # rescale image intensities to 0-1
    cropped_image = (cropped_image-cropped_image.min()) / \
        (cropped_image.max()-cropped_image.min()+1e-10)
    return cropped_image, cropped_label


def rescale_intensity(data, new_min=0, new_max=1, eps=1e-20):
    '''
    rescale pytorch batch data
    :param data: N*1*H*W
    :return: data with intensity ranging from 0 to 1
    '''
    bs, c, h, w = data.size(0), data.size(1), data.size(2), data.size(3)
    data = data.view(bs*c, -1)
    old_max = torch.max(data, dim=1, keepdim=True).values
    old_min = torch.min(data, dim=1, keepdim=True).values
    new_data = (data - old_min) / (old_max - old_min + eps) * \
        (new_max-new_min)+new_min
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
        old_states = {}
        for name, module in model.named_children():
            if hasattr(module, 'track_running_stats'):
                old_state = module.track_running_stats
                if hist_states is not None:
                    module.track_running_stats = hist_states[name]
                else:
                    if new_state is not None:
                        module.track_running_stats = new_state
                old_states[name] = old_state
        return old_states

    old_states = switch_attr(model, False)
    yield
    switch_attr(model, old_states)


def set_grad(module, requires_grad=False):
    for p in module.parameters():  # reset requires_grad
        p.requires_grad = requires_grad


def get_unet_model(model_path, num_classes=2, device=None, model_arch='UNet_16'):
    '''
    init model and load the trained parameters from the disk.
    model path: string. path to the model checkpoint
    device: torch device
    return pytorch nn.module model
    '''
    assert check_dir(model_path) == 1, model_path+' does not exists'
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_arch == 'UNet_16':
        model = UNet(input_channel=1, num_classes=num_classes, feature_scale=4)
    elif model_arch == 'UNet_64':
        model = UNet(input_channel=1, num_classes=num_classes, feature_scale=1)
    else:
        raise NotImplementedError
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    return model


def random_chain(alist, *args):
    """[select a sub list from  a list and chained them in a random order]

    Args:
        list ([type]): [description]
    """
    length = len(alist)
    assert length >= 1, "input list must contains at least one element"
    if length == 1:
        results = [alist]
        if len(args) > 0:
            for arg in args:
                assert len(arg) == 1, 'must share equal size'
                results.append(arg)

            return results
        else:
            return results

    sub_len = np.random.randint(low=1, high=length+1)

    r = random.random()            # randomly generating a real in [0,1)
    # lambda : r is an unary function which returns r
    random.shuffle(alist, lambda: r)
    # using the same function as used in prev line so that shuffling order is the same
    results = []
    results.append(alist[:sub_len])
    if len(args) >= 0:
        for arg in args:
            random.shuffle(arg, lambda: r)
            results.append(arg[:sub_len])

    return results
