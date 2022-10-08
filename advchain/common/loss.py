import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def calc_segmentation_consistency(output, reference, divergence_types=['kl', 'contour'],
                                  divergence_weights=[1.0, 0.5], class_weights=None, scales=[0],
                                  mask=None, is_gt=False):
    """
    measuring the difference between two predictions (network logits before softmax)
    Args:
        output (torch tensor 4d): network predicts: NCHW (after perturbation)
        reference (torch tensor 4d): network references: NCHW (before perturbation)
        divergence_types (list, string): specifying loss types. Defaults to ['kl','contour'].
        divergence_weights (list, float): specifying coefficients for each loss above. Defaults to [1.0,0.5].
        class_weights (list of scalars):  specifying class weights for loss computation
        scales (list of int): specifying a list of downsampling rates so that losses will be calculated on different scales. Defaults to [0].
        mask ([tensor], 0-1 onehotmap): [N*1*H*W]. disable  loss computation on corresponding elements with mask=0. Defaults to None.
        is_gt: bool, if true, will use one-hot encoded `reference' instead of probabilities maps after appying softmax to compute the consistency loss
    Raises:
        NotImplementedError: when loss name is not in ['kl','mse','contour']
    Returns:
        loss (tensor float): 
    """
    if class_weights is not None:
        raise NotImplemented
    dist = 0.
    num_classes = reference.size(1)
    spatial_dims = len(output.size())-2
    assert spatial_dims == 2 or spatial_dims == 3, 'only support 2d or 3d segmentation'
    assert len(output.size()) == len(reference.size()), 'scales and weights should have the same length'
    if mask is None:
        # apply masks so that only gradients on non-zero regions  will be backpropagated.
        mask = torch.ones_like(output).float().to(reference.device)
    for scale in scales:
        if scale > 0:
            if spatial_dims == 2:
               down_sample_pool_fn  = torch.nn.AvgPool2d
            else:down_sample_pool_fn  = torch.nn.AvgPool3d
            output_reference = down_sample_pool_fn(2 ** scale)(reference)
            output_new = down_sample_pool_fn(2 ** scale)(output)
        else:
            output_reference = reference
            output_new = output
        for divergence_type, d_weight in zip(divergence_types, divergence_weights):
            loss = 0.
            if divergence_type == 'kl':
                '''
                standard kl loss 
                '''
                loss = kl_divergence(
                    pred=output_new, reference=output_reference, mask=mask, is_gt=is_gt)
            elif divergence_type == 'mse':
                
                if not is_gt:
                    target_pred = torch.softmax(output_reference, dim=1)
                else:
                    target_pred = output_reference
                input_pred = torch.softmax(output_new, dim=1)
                loss = torch.nn.MSELoss(reduction='mean')(
                    target=target_pred*mask, input=input_pred*mask)
                loss = loss/(torch.numel(mask)/num_classes)
            elif divergence_type == 'contour':  # contour-based loss
                if not is_gt:
                    target_pred = torch.softmax(output_reference, dim=1)
                else:
                    target_pred = output_reference
                input_pred = torch.softmax(output_new, dim=1)
                cnt = 0
                use_gpu = input_pred.device!=torch.device("cpu")

                for i in range(1, num_classes):
                    cnt += 1
                    loss += contour_loss(input=input_pred[:, [i], ], target=(target_pred[:, [i]]), ignore_background=False, mask=mask,
                                         one_hot_target=False,use_gpu=use_gpu,device = input_pred.device)
                if cnt > 0:
                    loss /= cnt

            else:
                raise NotImplementedError

            # print ('{}:{}'.format(divergence_type,loss.item()))

            dist += 2 ** scale*(d_weight * loss)
    return dist / (1.0 * len(scales))


def calc_segmentation_mse_consistency(input, target):
    loss = calc_segmentation_consistency(output=input, reference=target, divergence_types=[
                                         'mse'], divergence_weights=[1.0], class_weights=None, mask=None)
    return loss


def calc_segmentation_kl_consistency(input, target):
    loss = calc_segmentation_consistency(output=input, reference=target, divergence_types=[
                                         'kl'], divergence_weights=[1.0], class_weights=None, mask=None)
    return loss


def contour_loss(input, target,  use_gpu=True, ignore_background=True, one_hot_target=True, mask=None,device=torch.device("cuda")):
    '''
    calc the contour loss across object boundaries (WITHOUT background class)
    :param input: NDArray. N*num_classes*H*W : pixelwise probs. for each class e.g. the softmax output from a neural network
    :param target: ground truth labels (NHW) or one-hot ground truth maps N*C*H*W
    :param use_gpu:boolean. default: True, use GPU.
    :param ignore_background:boolean, ignore the background class. default: True
    :param one_hot_target: boolean. if true, will first convert the target from NHW to NCHW. Default: True.
    :return:
    '''
    n, num_classes, h, w = input.size(0), input.size(
        1), input.size(2), input.size(3)
    spatial_dims = len(input.size()) - 2
    if one_hot_target:
        onehot_mapper = One_Hot(depth=num_classes, use_gpu=use_gpu, device=device)
        target = target.long()
        onehot_target = onehot_mapper(target).contiguous().view(
             input.size())
    else:
        onehot_target = target
    assert onehot_target.size() == input.size(), 'pred size: {} must match target size: {}'.format(
        str(input.size()), str(onehot_target.size()))

    if mask is None:
        # apply masks so that only gradients on certain regions will be backpropagated.
        mask = torch.ones_like(input).long().to(input.device)
        mask.requires_grad = False
    else:
        pass
        # print ('mask applied')

    if ignore_background:
        object_classes = num_classes - 1
        target_object_maps = onehot_target[:, 1:].float()
        input = input[:, 1:]
    else:
        target_object_maps = onehot_target
        object_classes = num_classes
    ## 2D 
    if spatial_dims == 2:
        x_filter = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]]).reshape(1, 1, 3, 3)

        x_filter = np.repeat(x_filter, axis=1, repeats=object_classes)
        x_filter = np.repeat(x_filter, axis=0, repeats=object_classes)
        conv_x = nn.Conv2d(in_channels=object_classes, out_channels=object_classes, kernel_size=3, stride=1, padding=1,
                        dilation=1, bias=False)

        conv_x.weight = nn.Parameter(torch.from_numpy(x_filter).float())

        y_filter = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]]).reshape(1, 1, 3, 3)
        y_filter = np.repeat(y_filter, axis=1, repeats=object_classes)
        y_filter = np.repeat(y_filter, axis=0, repeats=object_classes)
        conv_y = nn.Conv2d(in_channels=object_classes, out_channels=object_classes, kernel_size=3, stride=1, padding=1,
                        bias=False)
        conv_y.weight = nn.Parameter(torch.from_numpy(y_filter).float())
        if use_gpu:
            conv_y = conv_y.to(input.device)
            conv_x = conv_x.to(input.device)
        for param in conv_y.parameters():
            param.requires_grad = False
        for param in conv_x.parameters():
            param.requires_grad = False
    elif spatial_dims == 3:
        hx, hy, hz = np.array([[1, 2, 1]]), np.array([[1, 2, 1]]), np.array([[1, 2, 1]])
        hpx, hpy, hpz = np.array([[1, 0, -1]]), np.array([[1, 0, -1]]), np.array([[1, 0, -1]])
        # make 3D kernel
        gx = (hpx*hy.T).reshape(3,3,1)*hz ## 3x3*3 matrix
        gz = (hx*hpy.T).reshape(3,3,1)*hz
        gz = (hx*hy.T).reshape(3,3,1)*hpz
        gx = gx.reshape(1,1,3,3,3)
        gy = gx.reshape(1,1,3,3,3)
        gz = gz.reshape(1,1,3,3,3)
        gx.repeat(object_classes, axis=0)
        gy.repeat(object_classes, axis=0)
        gz.repeat(object_classes, axis=0)
        gx = gx.repeat(object_classes, axis=1)
        gy = gy.repeat(object_classes, axis=1)
        gz = gz.repeat(object_classes, axis=1)
        conv_x = nn.Conv3d(in_channels=object_classes, out_channels=object_classes, kernel_size=3, stride=1, padding=1,
                        dilation=1, bias=False)

        conv_y = nn.Conv3d(in_channels=object_classes, out_channels=object_classes, kernel_size=3, stride=1, padding=1,
                        dilation=1, bias=False)
        conv_z = nn.Conv3d(in_channels=object_classes, out_channels=object_classes, kernel_size=3, stride=1, padding=1,
                        dilation=1, bias=False)
        conv_x.weight = nn.Parameter(torch.from_numpy(gx).float())
        conv_y.weight = nn.Parameter(torch.from_numpy(gy).float())
        conv_z.weight = nn.Parameter(torch.from_numpy(gz).float())
        if use_gpu:
            conv_x = conv_x.to(input.device)
            conv_y = conv_y.to(input.device)
            conv_z = conv_z.to(input.device)
        for param in conv_x.parameters():
            param.requires_grad = False
        for param in conv_y.parameters():
            param.requires_grad = False
        for param in conv_z.parameters():
            param.requires_grad = False
    else: raise NotImplementedError
    g_x_pred = conv_x(input)*mask[:, :object_classes]
    g_y_pred = conv_y(input)*mask[:, :object_classes]
    g_y_truth = conv_y(target_object_maps)*mask[:, :object_classes]
    g_x_truth = conv_x(target_object_maps)*mask[:, :object_classes]

    # mse loss
    if spatial_dims == 2:
        loss = 0.5*(torch.nn.MSELoss(reduction='mean')(input=g_x_pred, target=g_x_truth) +
                    torch.nn.MSELoss(reduction='mean')(input=g_y_pred, target=g_y_truth))
    elif spatial_dims == 3:
        g_z_pred = conv_z(input)*mask[:, :object_classes]
        g_z_truth = conv_z(target_object_maps)*mask[:, :object_classes]
        loss = 1/3*(torch.nn.MSELoss(reduction='mean')(input=g_x_pred, target=g_x_truth) +
                    torch.nn.MSELoss(reduction='mean')(input=g_y_pred, target=g_y_truth)+ 
                    torch.nn.MSELoss(reduction='mean')(input=g_z_pred, target=g_z_truth))
    return loss


def kl_divergence(reference, pred, mask=None, is_gt=False):
    '''
    support 2D and 3D
    calc the kl div distance between two outputs p and q from a network/model: p(y1|x1).p(y2|x2).
    :param reference p: directly output from network using origin input without softmax
    :param output q: approximate output: directly output from network using perturbed input without softmax
    :param is_gt: is onehot maps
    :return: kl divergence: DKL(P||Q) = mean(\sum_1 \to C (p^c log (p^c|q^c)))

    '''
    q = pred

    if mask is None:
        mask = torch.ones_like(q, device=q.device)
        mask.requires_grad = False
    if not is_gt:
        p = F.softmax(reference, dim=1)
        log_p = F.log_softmax(reference, dim=1)
    else:
        p = torch.where(reference == 0, 1e-8, 1-1e-8)
        log_p = torch.log(p)  # avoid NAN when log(0)
    cls_plogp = mask*(p * log_p)
    cls_plogq = mask*(p * F.log_softmax(q, dim=1))
    plogp = torch.sum(cls_plogp, dim=1, keepdim=False)
    plogq = torch.sum(cls_plogq, dim=1, keepdim=False)
    kl_loss = torch.mean(plogp - plogq)
    return kl_loss


class One_Hot(nn.Module):
    def __init__(self, depth, use_gpu=True, device=torch.device("cuda")):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.device = device if use_gpu else torch.device("cpu")
        if use_gpu:
            self.ones = torch.sparse.torch.eye(depth).to(self.device)
        else:
            self.ones = torch.sparse.torch.eye(depth)

    def forward(self, X_in):
        n_dim = X_in.dim()
        output_size = X_in.size() + torch.Size([self.depth])
        num_element = X_in.numel()
        X_in = X_in.data.long().view(num_element)
        out = Variable(self.ones.index_select(0, X_in)).view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


def cross_entropy_2D(input, target, weight=None, size_average=True):
    """[summary]
    calc cross entropy loss computed on 2D images 
    Args:
        input ([torch tensor]): [4d logit] in the format of NCHW
        target ([torch tensor]): 3D labelmap or 4d logit (before softmax), in the format of NCHW
        weight ([type], optional): weights for classes. Defaults to None.
        size_average (bool, optional): take the average across the spatial domain. Defaults to True.
    Raises:
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)

    if len(target.size()) == 3:
        target = target.view(target.numel())
        if not weight is None:
            # sum(weight) =C,  for numerical stability.
            weight = weight/weight.sum()*c
        loss_vector = F.nll_loss(
            log_p, target, weight=weight, reduction="none")
        loss = torch.sum(loss_vector)
        if size_average:
            loss /= (n*h*w)
    elif len(target.size()) == 4:
        # ce loss=-qlog(p)
        reference = target
        reference = reference.transpose(1, 2).transpose(
            2, 3).contiguous().view(-1, c)  # M,C
        if weight is None:
            plogq = torch.sum(reference * log_p, dim=1)
            plogq = torch.sum(plogq)
            if size_average:
                plogq /= (n*h*w)
        else:
            weight = np.array(weight)
            # sum(weight) =C
            weight = weight/weight.sum()*c
            plogq_class_wise = reference * log_p
            plogq_sum_class = 0.
            for i in range(c):
                plogq_sum_class += torch.sum(plogq_class_wise[:, i]*weight[i])
            plogq = plogq_sum_class
            if size_average:
                # only average loss on the mask entries with value =1
                plogq /= (n*h*w)
        loss = -1*plogq
    else:
        raise NotImplementedError
    return loss
