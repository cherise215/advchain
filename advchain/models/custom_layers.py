import math
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch
from torch.nn.utils import spectral_norm


class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in::

    Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507
    """

    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=False)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        input_rank = len(input_tensor.shape)
        squeeze_tensor = input_tensor.view(
            batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialSELayer(nn.Module):
    """
    Implementation of SE block -- squeezing spatially
    and exciting channel-wise described in::
    Roy et al., Concurrent Spatial and Channel Squeeze & Excitation
    in Fully Convolutional Networks, MICCAI 2018
    Roy et al., Recalibrating Fully Convolutional Networks with Spatial
    and Channel'Squeeze & Excitation'Blocks, IEEE TMI 2018
    """

    def __init__(self, num_channels):
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        # spatial squeeze
        batch_size, _, a, b = input_tensor.size()
        squeeze_tensor = self.sigmoid(self.conv(input_tensor))

        # spatial excitation
        output_tensor = torch.mul(
            input_tensor, squeeze_tensor.view(batch_size, 1, a, b))

        return output_tensor


class ChannelSpatialSELayer(nn.Module):
    """
    Implementation of concurrent spatial and channel
    squeeze & excitation: with Max-out aggregation
    Roy et al., Recalibrating Fully Convolutional Networks with Spatial
    and Channel'Squeeze & Excitation'Blocks, IEEE TMI 2018
    """

    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        output_tensor = torch.max(
            self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


def bilinear_additive_upsampling(x, output_channel_num):
    """
    pytorch implementation of Bilinear Additive Upsampling
    ref: @MikiBear_
    Tensorflow Implementation of Bilinear Additive Upsampling.
    Reference : https://arxiv.org/abs/1707.05847
    https://gist.github.com/mikigom/bad72795c5e87e3caa9464e64952b524
    """

    input_channel = x.size(1)
    assert input_channel > output_channel_num, 'the number of output channels should not be greater than the number of  input channels'
    assert input_channel % output_channel_num == 0, 'input channels must could be equally divided by output_channel_num '
    channel_split = int(input_channel / output_channel_num)

    # print (channel_split)

    new_h = x.size(2)*2
    new_w = x.size(3)*2
    upsampled_op = torch.nn.Upsample(scale_factor=2, mode='bilinear')
    upsampled_x = upsampled_op(x)

    # print(upsampled_x.size())

    result = torch.zeros(x.size(0), output_channel_num, new_h, new_w)
    for i in range(0, output_channel_num):
        splited_upsampled_x = upsampled_x.narrow(
            1, start=i * channel_split, length=channel_split)
        result[:, i, :, :] = torch.sum(splited_upsampled_x, 1)

    # by default, should be cuda tensor.
    result = result.cuda()
    return result


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation, factor=8, if_SN=False):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // factor, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // factor, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        if if_SN:
            self.query_conv = spectral_norm(self.query_conv)
            self.key_conv = spectral_norm(self.key_conv)
            self.value_conv = spectral_norm(self.value_conv)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        weighted_out = self.gamma * out
        final = weighted_out + x
        return final, weighted_out, attention

##################################################################################
# Normalization layers
##################################################################################


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


##################################################################################
# Normalization layers
##################################################################################
class AdaptiveBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        size = list(num_features.size())
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(size))
        self.register_buffer('running_var', torch.ones_like(size))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply batchNorm
        #x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        # nn.BatchNorm2d
        out = F.batch_norm(
            x, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

# Domain specific Batch Normalisation layer:


# Batch Instance Normalization
# credit to : https://github.com/hyeonseob-nam/Batch-Instance-Normalization/blob/master/models/batchinstancenorm.py
class _BatchInstanceNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(_BatchInstanceNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        if self.affine:
            self.gate = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('gate', None)

        self.gate.data.fill_(1)

    def forward(self, input):
        self._check_input_dim(input)

        # Batch norm
        if self.affine:
            bn_w = self.weight * self.gate
        else:
            bn_w = self.gate
        # print ('bn_weight',bn_w)
        # print ('self.gate',self.gate)

        out_bn = F.batch_norm(
            input, self.running_mean, self.running_var, bn_w, self.bias,
            self.training, self.momentum, self.eps)

        # Instance norm
        b, c = input.size(0), input.size(1)
        if self.affine:
            in_w = self.weight * (1 - self.gate)
        else:
            in_w = 1 - self.gate
        input = input.view(1, b * c, *input.size()[2:])
        out_in = F.batch_norm(
            input, None, None, None, None,
            True, self.momentum, self.eps)
        out_in = out_in.view(b, c, *input.size()[2:])
        out_in.mul_(in_w[None, :, None, None])

        return out_bn + out_in


class BatchInstanceNorm1d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                'expected 2D or 3D input (got {}D input)'.format(input.dim()))


class BatchInstanceNorm2d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(
                'expected 4D input (got {}D input)'.format(input.dim()))


class BatchInstanceNorm3d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError(
                'expected 5D input (got {}D input)'.format(input.dim()))


def spatial_pyramid_pool(previous_conv, batch_size, previous_conv_size, out_bin_sizes):
    '''
    ref: Spatial Pyramid Pooling in Deep ConvolutionalNetworks for Visual Recognition
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''
    # print(previous_conv.size())
    for i in range(0, len(out_bin_sizes)):
        # print(previous_conv_size)
        #assert  previous_conv_size[0] % out_bin_sizes[i]==0, 'please make sure feature size can be devided by bins'
        h_wid = int(math.ceil(previous_conv_size[0] / out_bin_sizes[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_bin_sizes[i]))
        # h_stride = int(math.floor(previous_conv_size[0] / out_bin_sizes[i]))
        # w_stride = int(math.floor(previous_conv_size[1] / out_bin_sizes[i]))
        h_pad = (h_wid * out_bin_sizes[i] - previous_conv_size[0] + 1) // 2
        w_pad = (w_wid * out_bin_sizes[i] - previous_conv_size[1] + 1) // 2
        maxpool = nn.MaxPool2d(kernel_size=(h_wid, w_wid), stride=(
            h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if (i == 0):
            spp = x.view(batch_size, -1)
        else:
            spp = torch.cat((spp, x.view(batch_size, -1)), dim=1)
    return spp
