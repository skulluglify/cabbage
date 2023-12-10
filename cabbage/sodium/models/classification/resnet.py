#!/usr/bin/env python3

from torch import nn
from torchvision.models import (resnet101,
                                ResNet101_Weights)
from torchvision.models import (resnet152,
                                ResNet152_Weights)
from torchvision.models import (resnet18,
                                ResNet18_Weights, ResNet)
from torchvision.models import (resnet34,
                                ResNet34_Weights)
from torchvision.models import (resnet50,
                                ResNet50_Weights)
from torchvision.models import (wide_resnet101_2,
                                Wide_ResNet101_2_Weights)
from torchvision.models import (wide_resnet50_2,
                                Wide_ResNet50_2_Weights)

from sodium.x.runtime.utils.refs import refs_obj_set_val


class _resnet_pretrained:

    @property
    def resnet18(self) -> ResNet:
        return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1,
                        progress=False)

    @property
    def resnet34(self) -> ResNet:
        return resnet34(weights=ResNet34_Weights.IMAGENET1K_V1,
                        progress=False)

    @property
    def resnet50(self) -> ResNet:
        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2,
                        progress=False)

    @property
    def resnet101(self) -> ResNet:
        return resnet101(weights=ResNet101_Weights.IMAGENET1K_V2,
                         progress=False)

    @property
    def resnet152(self) -> ResNet:
        return resnet152(weights=ResNet152_Weights.IMAGENET1K_V2,
                         progress=False)

    @property
    def wide_resnet50_2(self) -> ResNet:
        return wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2,
                               progress=False)

    @property
    def wide_resnet101_2(self) -> ResNet:
        return wide_resnet101_2(weights=Wide_ResNet101_2_Weights.IMAGENET1K_V2,
                                progress=False)

    default = resnet50  # set default.


models = _resnet_pretrained()  # initial methods.


def create_cnn_model_resnet_trainable(num_classes: int, model: ResNet = models.resnet50) -> nn.Module:
    """
        Create FasterRCNN Model ResNet (ResNet50),
        Fast Training, And Trainable.
    :param num_classes:
    :param model:
    :return:
    """
    # Make it Usable General Purposes.
    # Set value 'in_features' and 'out_features' in fully connected layers.

    key: str
    module: nn.Linear
    key, module = tuple(model.named_modules())[-1]
    in_features = module.in_features

    # In the context of PyTorch and neural network models,
    # the `bias` parameter in `nn.Linear` refers to whether you want
    # that layer to have a bias or not. Bias is a constant value that is added
    # to each output feature. Bias allows the linear transformation to shift
    # the output values, making it more flexible and capable of adjusting
    # to various data ranges.
    # If you set `bias=False` when defining your `nn.Linear` layer,
    # then that layer will not learn an additive bias.
    # This means that there is no constant value added
    # to the output from that layer. Conversely, if you set `bias=True`
    # (which is the default value), then that layer will learn an additive bias.
    # In other words, that layer will add a constant value to its output,
    # which can help the model adjust to the data.
    # In the context of replacing the last module of the ResNet model
    # with `nn.Linear`, setting `bias=False` would mean that the final linear layer
    # will not have a bias. This could affect your model's performance,
    # depending on your specific data and task. Typically, it's good to let
    # the model learn a bias, but in some cases, you might want to disable it
    # for certain reasons.

    module = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    # Replace Linear Module.
    if not refs_obj_set_val(model, key, module):
        raise Exception("Couldn't modified fully connected layer with 'num_classes' value")

    return model
