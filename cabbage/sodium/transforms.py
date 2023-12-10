#!/usr/bin/env python3
from typing import List

from torchvision import disable_beta_transforms_warning

try:
    disable_beta_transforms_warning()
    from torchvision.transforms.v2 import (AutoAugment, CenterCrop,
                                           Compose, Grayscale,
                                           Normalize, Resize)

    from torchvision.transforms.v2.functional import (to_grayscale, to_pil_image,
                                                      to_tensor, crop)

except ImportError:
    from torchvision.transforms import (AutoAugment, CenterCrop,
                                        Compose, Grayscale,
                                        Normalize, Resize)

    from torchvision.transforms.functional import (to_grayscale, to_pil_image,
                                                   to_tensor, crop)


mean: List[float] = [0.485, 0.456, 0.406]
std: List[float] = [0.229, 0.224, 0.225]

# Default Transforms To 224x224.
transforms_224x = Compose([
    # CenterCrop(size=(224, 244)),
    Resize(size=(224, 224), antialias=False),
    Normalize(mean=mean, std=std),
])

# Binding module 'torchvision.transforms' into 'sodium.torchy.transforms'
AutoAugment = AutoAugment
CenterCrop = CenterCrop
Compose = Compose
Grayscale = Grayscale
Normalize = Normalize
Resize = Resize
# ToImage = ToImage  # ToImage()
# ToDtype = ToDtype  # ToDtype(torch.float32, scale=True)

# Binding module 'torchvision.transforms.functional' into 'sodium.torchy.transforms.functional'
to_grayscale = to_grayscale
to_pil_image = to_pil_image
to_tensor = to_tensor
crop = crop