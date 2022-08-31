# Created by cc215 at 17/03/19
# Enter feature description here
# Enter scenario name here
# Enter steps here

import math
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
# noqa
from advchain.models.custom_layers import BatchInstanceNorm2d
from advchain.models.custom_layers import Self_Attn
from advchain.models.unet_parts import *
from advchain.common.utils import check_dir


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
    model.load_state_dict(torch.load(model_path,map_location=device))
    model = model.to(device)
    return model


class UNet(nn.Module):
    def __init__(self, input_channel, num_classes, feature_scale=1, encoder_dropout=None, decoder_dropout=None, norm=nn.BatchNorm2d, self_attention=False, if_SN=False, last_layer_act=None):
        super(UNet, self).__init__()
        self.inc = inconv(input_channel, 64//feature_scale,
                          norm=norm, dropout=encoder_dropout)
        self.down1 = down(64//feature_scale, 128//feature_scale,
                          norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down2 = down(128//feature_scale, 256//feature_scale,
                          norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down3 = down(256//feature_scale, 512//feature_scale,
                          norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down4 = down(512//feature_scale, 512//feature_scale,
                          norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.up1 = up(512//feature_scale, 512//feature_scale, 256 //
                      feature_scale, norm=norm, dropout=decoder_dropout, if_SN=if_SN)
        self.up2 = up(256//feature_scale, 256//feature_scale, 128 //
                      feature_scale, norm=norm, dropout=decoder_dropout, if_SN=if_SN)
        self.up3 = up(128//feature_scale, 128//feature_scale, 64 //
                      feature_scale, norm=norm, dropout=decoder_dropout, if_SN=if_SN)
        self.up4 = up(64//feature_scale, 64//feature_scale, 64 //
                      feature_scale, norm=norm, dropout=decoder_dropout, if_SN=if_SN)
        if self_attention:
            self.self_atn = Self_Attn(512//feature_scale, 'relu', if_SN=False)
        self.self_attention = self_attention
        self.outc = outconv(64//feature_scale, num_classes)
        self.n_classes = num_classes
        self.attention_map = None
        self.last_act = last_layer_act

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        self.hidden_feature = x5
        if self.self_attention:
            x5, w_out, attention = self.self_atn(x5)
            self.attention_map = attention
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        if not self.last_act is None:
            x = self.last_act(x)

        return x

    def predict(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.self_attention:
            x5, w_out, attention = self.self_atn(x5)
            self.attention_map = attention
        x = self.up1(x5, x4)
        x = self.up2(x, x3,)
        x = self.up3(x, x2,)
        x = self.up4(x, x1,)
        x = self.outc(x)
        if self.self_attention:
            return x, w_out, attention

        return x

    def get_net_name(self):
        return 'unet'

    def adaptive_bn(self, if_enable=False):
        if if_enable:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, BatchInstanceNorm2d):
                    module.train()
                    module.track_running_stats = True

    def init_bn(self):
        for name, module in self.named_modules():
           # print(name, module)
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, BatchInstanceNorm2d):
                # print (module)
                module.running_mean.zero_()
                module.running_var.fill_(1)

    def fix_conv_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Conv2d):
                # print(name)
                for k in module.parameters():  # fix all conv layers
                    k.requires_grad = False

            else:
                for k in module.parameters():
                    k.requires_grad = True

    def activate_conv_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Conv2d):
                # print(name)
                for k in module.parameters():
                    k.requires_grad = True

    def print_bn(self):
        for name, module in self.named_modules():
            # print(name, module)
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, BatchInstanceNorm2d):
                print(module.running_mean)
                print(module.running_var)

    def fix_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                for k in module.parameters():  # fix all conv layers
                    k.requires_grad = False
            elif 'outc' in name:
                if isinstance(module, nn.Conv2d):
                    for k in module.parameters():  # except last layers
                        k.requires_grad = True
            else:
                for k in module.parameters():  # fix all conv layers
                    k.requires_grad = False

    def get_adapted_params(self):
        for name, module in self.named_modules():
            # if isinstance(module,nn.BatchNorm2d):
            #     for p in module.parameters():
            #         yield p
            # if 'outc' in name:
            #     if isinstance(module,nn.Conv2d):
            #        for p in module.parameters(): ##fix all conv layers
            #            yield p
            for k in module.parameters():  # fix all conv layers
                if k.requires_grad:
                    yield k

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.inc)
        b.append(self.down1)
        b.append(self.down2)
        b.append(self.down3)
        b.append(self.down4)
        b.append(self.up1)
        b.append(self.up2)
        b.append(self.up3)
        b.append(self.up4)
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.outc.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]

    def cal_num_conv_parameters(self):
        cnt = 0

        for module_name, module in self.named_modules():
            if isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Conv2d):
                # print(module_name)
                for name, param in module.named_parameters():
                    if param.requires_grad:
                        if 'weight' in name:
                            print(name, param.data)
                            param = param.view(-1, 1)
                            param.squeeze()
                            cnt += len(param.data)
        return cnt


class DeeplySupervisedUNet(nn.Module):
    def __init__(self, input_channel, num_classes, base_n_filters=64, dropout=None, activation=nn.ReLU):
        super(DeeplySupervisedUNet, self).__init__()
        self.inc = inconv(input_channel, base_n_filters, activation=activation)
        self.down1 = down(base_n_filters, base_n_filters *
                          2, activation=activation)
        self.down2 = down(base_n_filters*2, base_n_filters *
                          4, activation=activation)
        self.down3 = down(base_n_filters*4, base_n_filters *
                          8, activation=activation)
        self.down4 = down(base_n_filters*8, base_n_filters *
                          8, activation=activation)
        self.up1 = up(base_n_filters*8, base_n_filters*8,
                      base_n_filters*4, activation=activation, dropout=dropout)
        self.up2 = up(base_n_filters*4, base_n_filters*4,
                      base_n_filters*2, activation=activation, dropout=dropout)
        self.up3 = up(base_n_filters*2, base_n_filters*2,
                      base_n_filters, activation=activation, dropout=dropout)
        self.up4 = up(base_n_filters, base_n_filters,
                      base_n_filters, activation=activation)
        self.up2_conv1 = outconv_relu(
            base_n_filters*2, num_classes, activation=None)
        self.up2_up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.up3_conv1 = outconv_relu(
            base_n_filters, num_classes, activation=None)
        self.up3_up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.outc = outconv(base_n_filters, num_classes)
        self.n_classes = num_classes
        self.dropout = dropout
        if dropout is not None:
            self.dropoutlayer = nn.Dropout2d(p=dropout)
        else:
            self.dropoutlayer = nn.Dropout2d(p=0)

    def forward(self, x, multi_out=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.dropoutlayer(self.down2(x2))  # tail it after pooling
        x4 = self.dropoutlayer(self.down3(x3))
        x5 = self.dropoutlayer(self.down4(x4))

        x = self.up1(x5, x4)
        x_2 = self.up2(x, x3)  # insert dropout after concat
        dsv_x_2 = self.up2_conv1(x_2)
        dsv_x_2_up = self.up2_up(dsv_x_2)

        x_3 = self.up3(x_2, x2)
        dsv_x_3 = self.up3_conv1(x_3)
        dsv_mixed = dsv_x_2_up+dsv_x_3
        dsv_mixed_up = self.up3_up(dsv_mixed)

        x_4 = self.up4(x_3, x1)
        out = self.outc(x_4)
        final_output = torch.add(out, dsv_mixed_up)
        if multi_out:
            return out, dsv_mixed_up, final_output

        return final_output

    def predict(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x_2 = self.up2(x, x3)
        dsv_x_2 = self.up2_conv1(x_2)
        dsv_x_2_up = self.up2_up(dsv_x_2)

        x_3 = self.up3(x_2, x2)
        dsv_x_3 = self.up3_conv1(x_3)
        dsv_mixed = dsv_x_2_up + dsv_x_3
        dsv_mixed_up = self.up3_up(dsv_mixed)

        x_4 = self.up4(x_3, x1)
        out = self.outc(x_4)
        final_output = torch.add(out, dsv_mixed_up)

        return final_output

    def get_net_name(self):
        return 'dsv_unet'

    def adaptive_bn(self, if_enable=False):
        if if_enable:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, BatchInstanceNorm2d):
                    # if 'down' in name or 'up' in name or 'inc' in name:
                    # print (module.name)
                    module.train()
                    module.track_running_stats = True

    def init_bn(self):
        for name, module in self.named_modules():
            # print(name, module)
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, BatchInstanceNorm2d):
                # print(module)
                module.running_mean.zero_()
                module.running_var.fill_(1)

    def fix_params(self):
        for name, param in self.named_parameters():
            if 'outc' in name:
                # initialize
                if 'conv' in name and 'weight' in name:
                    n = param.size(0) * param.size(2) * param.size(3)
                    param.data.normal_().mul_(math.sqrt(2. / n))
            else:
                param.requires_grad = False

    def cal_num_conv_parameters(self):
        cnt = 0

        for module_name, module in self.named_modules():
            if isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Conv2d):
                # print (module_name)
                for name, param in module.named_parameters():
                    if param.requires_grad:
                        if 'weight' in name:
                            print(name, param.data)
                            param = param.view(-1, 1)
                            param.squeeze()
                            cnt += len(param.data)
        return cnt


class UNetv2(nn.Module):
    def __init__(self, input_channel, num_classes, feature_scale=1, encoder_dropout=None, decoder_dropout=None, norm=nn.BatchNorm2d, self_attention=False, if_SN=False, last_layer_act=None):
        super(UNetv2, self).__init__()
        self.inc = inconv(input_channel, 64//feature_scale,
                          norm=norm, dropout=encoder_dropout)
        self.down1 = down(64//feature_scale, 128//feature_scale,
                          norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down2 = down(128//feature_scale, 256//feature_scale,
                          norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down3 = down(256//feature_scale, 512//feature_scale,
                          norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down4 = down(512//feature_scale, 1024//feature_scale,
                          norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.up1 = up(1024//feature_scale, 512//feature_scale, 256 //
                      feature_scale, norm=norm, dropout=decoder_dropout, if_SN=if_SN)
        self.up2 = up(256//feature_scale, 256//feature_scale, 128 //
                      feature_scale, norm=norm, dropout=decoder_dropout, if_SN=if_SN)
        self.up3 = up(128//feature_scale, 128//feature_scale, 64 //
                      feature_scale, norm=norm, dropout=decoder_dropout, if_SN=if_SN)
        self.up4 = up(64//feature_scale, 64//feature_scale, 64 //
                      feature_scale, norm=norm, dropout=decoder_dropout, if_SN=if_SN)
        if self_attention:
            self.self_atn = Self_Attn(512//feature_scale, 'relu', if_SN=False)
        self.self_attention = self_attention
        self.outc = outconv(64//feature_scale, num_classes)
        self.n_classes = num_classes
        self.attention_map = None
        self.last_act = last_layer_act

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        self.hidden_feature = x5
        if self.self_attention:
            x5, w_out, attention = self.self_atn(x5)
            self.attention_map = attention
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        if not self.last_act is None:
            x = self.last_act(x)

        return x

    def predict(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.self_attention:
            x5, w_out, attention = self.self_atn(x5)
            self.attention_map = attention
        x = self.up1(x5, x4)
        x = self.up2(x, x3,)
        x = self.up3(x, x2,)
        x = self.up4(x, x1,)
        x = self.outc(x)
        if self.self_attention:
            return x, w_out, attention

        return x

    def get_net_name(self):
        return 'unet'

    def adaptive_bn(self, if_enable=False):
        if if_enable:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, BatchInstanceNorm2d):
                    module.train()
                    module.track_running_stats = True

    def init_bn(self):
        for name, module in self.named_modules():
           # print(name, module)
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, BatchInstanceNorm2d):
                # print(module)
                module.running_mean.zero_()
                module.running_var.fill_(1)


if __name__ == '__main__':
    model = UNet(input_channel=1, feature_scale=1,
                 num_classes=4, encoder_dropout=0.3)
    model.train()
    image = torch.autograd.Variable(torch.randn(2, 1, 224, 224), volatile=True)
    result = model(image)
    print(model.hidden_feature.size())
    print(result.size())
