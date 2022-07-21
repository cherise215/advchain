import torch.nn as nn
import torch
from torch.nn import functional as F

class Fixable2DDropout(nn.Module):
    """
    _summary_method = torch.nn.Dropout2d.__init__
     based on 2D pytorch mask, supports lazy load with last generated mask
    """
    def __init__(self, p: float = 0.5,inplace=False,lazy_load: bool = False,training=True):
        super(Fixable2DDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.inplace = inplace
        self.seed  = None
        self.lazy_load = lazy_load
        self.training=training

    def forward(self, X):
        if self.training:
            if self.lazy_load:
                if not self.seed is None:
                    seed  = self.seed
                else:
                    seed = torch.seed()
            else:seed = torch.seed()
        else:
            seed = torch.seed()
        self.seed=seed
        torch.manual_seed(seed)
        X = F.dropout2d(X, p=self.p, training=self.training, inplace=self.inplace)
        return X

class Fixable3DDropout(nn.Module):
    """
    _summary_method = torch.nn.Dropout2d.__init__
     based on 2D pytorch mask, supports lazy load with last generated mask
    """
    def __init__(self, p: float = 0.5,inplace=False,lazy_load: bool = False,training=True):
        super(Fixable3DDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.inplace = inplace
        self.seed  = None
        self.lazy_load = lazy_load
        self.training=training

    def forward(self, X):
        if self.training:
            if self.lazy_load:
                if not self.seed is None:
                    seed  = self.seed
                else:
                    seed = torch.seed()
            else:seed = torch.seed()
        else:
            seed = torch.seed()
        self.seed=seed
        torch.manual_seed(seed)
        X = F.dropout3d(X, p=self.p, training=self.training, inplace=self.inplace)
        return X