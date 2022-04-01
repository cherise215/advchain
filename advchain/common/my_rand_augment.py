from typing import List, Tuple, Optional, Dict
from torch import Tensor
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as tv_func
from torchvision.transforms import  InterpolationMode

import math

### add random seed to RandAugment for reproducible augmentations
def _apply_op(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]],seed:Optional[int]):
    if seed is not None:
        torch.manual_seed(seed=seed)
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = tv_func.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = tv_func.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = tv_func.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = tv_func.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = tv_func.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = tv_func.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = tv_func.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = tv_func.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = tv_func.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = tv_func.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = tv_func.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = tv_func.autocontrast(img)
    elif op_name == "Equalize":
        img = tv_func.equalize(img)
    elif op_name == "Invert":
        img = tv_func.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img

class MyRandAugment(torchvision.transforms.RandAugment):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

        ## save intermediate opts for reproducible rand augment
        self.op_mega = None
        self.op_name = None
        self.magnitude_state = None
        self.generator = torch.Generator()
        self.seed=None



    def forward(self, img: Tensor, reuse_param=False, interpolation=None) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if interpolation is None:
            interpolation = self.interpolation 
        channels, height, width = img.size(1), img.size(2),img.size(3)  
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        if not reuse_param:
            seed =self.generator.seed()
            torch.manual_seed(seed)
            op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
            for _ in range(self.num_ops):
                op_index = int(torch.randint(len(op_meta), (1,)).item())
                op_name = list(op_meta.keys())[op_index]
                magnitudes, signed = op_meta[op_name]
                magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
                if signed and torch.randint(2, (1,)):
                    magnitude *= -1.0
                img = _apply_op(img, op_name, magnitude, interpolation=interpolation, fill=fill,seed=seed)
            self.seed = seed
            self.magnitude_state = magnitude
            self.op_name = op_name
            self.op_mega = op_meta
            
        else:
            if self.seed is not None:
                seed  = self.seed
            else:  
                seed =self.generator.seed()
            torch.manual_seed(seed)

            op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
            for _ in range(self.num_ops):
                if self.op_name is not None:
                    op_name = self.op_name
                else:
                    op_index = int(torch.randint(len(op_meta), (1,)).item())
                    op_name =list(op_meta.keys())[op_index]
                if self.magnitude_state is not None: magnitude  =self.magnitude_state
                else:
                    magnitudes, signed = op_meta[op_name]
                    magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
                    if signed and torch.randint(2, (1,)):
                        magnitude *= -1.0
                img = _apply_op(img, op_name, magnitude, interpolation=interpolation, fill=fill,seed=seed)
                self.seed = seed
            self.magnitude_state = magnitude
            self.op_name = op_name
            self.op_mega = op_meta
        return img

