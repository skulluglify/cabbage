#!/usr/bin/env python3
from typing import Tuple

from torch import nn
from torchvision.models import ResNet50_Weights, MobileNet_V3_Large_Weights, ResNet
from torchvision.models import WeightsEnum
from torchvision.models.detection import (fasterrcnn_mobilenet_v3_large_320_fpn,
                                          FasterRCNN_MobileNet_V3_Large_320_FPN_Weights)
from torchvision.models.detection import (fasterrcnn_mobilenet_v3_large_fpn,
                                          FasterRCNN_MobileNet_V3_Large_FPN_Weights)
from torchvision.models.detection import (fasterrcnn_resnet50_fpn,
                                          FasterRCNN_ResNet50_FPN_Weights)
from torchvision.models.detection import (fasterrcnn_resnet50_fpn_v2,
                                          FasterRCNN_ResNet50_FPN_V2_Weights)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from sodium.models.classification.resnet import models, create_cnn_model_resnet_trainable


def faster_rcnn_resnet50_fpn_eval() -> Tuple[FasterRCNN, WeightsEnum]:
    weights: WeightsEnum = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    num_classes = len(weights.meta["categories"])
    model: FasterRCNN = fasterrcnn_resnet50_fpn(weights=weights,
                                                progress=False,
                                                num_classes=num_classes,
                                                weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
                                                box_score_thresh=0.9)
    model.eval()
    return model, weights


def faster_rcnn_resnet50_fpn_v2_eval() -> Tuple[FasterRCNN, WeightsEnum]:
    weights: WeightsEnum = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    num_classes = len(weights.meta["categories"])
    model: FasterRCNN = fasterrcnn_resnet50_fpn_v2(weights=weights,
                                                   progress=False,
                                                   num_classes=num_classes,
                                                   weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
                                                   box_score_thresh=0.9)
    model.eval()
    return model, weights


def faster_rcnn_resnet50_fpn_v2_train(num_classes: int) -> FasterRCNN:
    """
        FasterRCNN ResNet50 FPN v2,
        Configuration for Training, but Slow.
    :param num_classes:
    :return:
    """

    weights: WeightsEnum = ResNet50_Weights.IMAGENET1K_V2
    model: FasterRCNN = fasterrcnn_resnet50_fpn_v2(weights=None, progress=False,
                                                   num_classes=num_classes, weights_backbone=weights,
                                                   trainable_backbone_layers=5,  # train all layers
                                                   box_score_thresh=0.0)

    model.train()
    return model


def faster_rcnn_mobilenet_v3_large_fpn_eval() -> Tuple[FasterRCNN, WeightsEnum]:
    weights: WeightsEnum = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    num_classes = len(weights.meta["categories"])
    model: FasterRCNN = fasterrcnn_mobilenet_v3_large_fpn(weights=weights,
                                                          progress=False,
                                                          num_classes=num_classes,
                                                          weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
                                                          box_score_thresh=0.9)
    model.eval()
    return model, weights


def faster_rcnn_mobilenet_v3_large_320_fpn_eval() -> Tuple[FasterRCNN, WeightsEnum]:
    weights: WeightsEnum = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    num_classes = len(weights.meta["categories"])
    model: FasterRCNN = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights,
                                                              progress=False,
                                                              num_classes=num_classes,
                                                              weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
                                                              box_score_thresh=0.9)
    model.eval()
    return model, weights


def create_faster_rcnn_model_resnet_trainable(num_classes: int,
                                              model: ResNet = models.resnet50) -> nn.Module:
    """
        Create FasterRCNN Model ResNet (ResNet50),
        Fast Training, And Trainable.
    :param num_classes:
    :param model:
    :return:
    """
    # Faster RCNN with ResNet50 (FPN).

    # Make it Usable General Purposes.
    # Set value 'in_features' and 'out_features' in fully connected layers.
    model = create_cnn_model_resnet_trainable(num_classes=num_classes, model=model)

    # return model

    # End of modified fully connected layers.

    # In the context of deep learning,
    # a backbone is a pre-trained network (like ResNet50)
    # that is used as a feature extractor for the main task,
    # which in this case is object detection with Faster R-CNN.

    # When they "remove the last 2 children from ResNet",
    # they are likely removing the last two layers of the ResNet50 model.
    # The last two layers of ResNet50 are usually the global average pooling layer
    # and the fully connected layer. These layers are specific to the task that ResNet50
    # was originally trained on (ImageNet classification), and are not useful for
    # the object detection task.

    # After removing these layers, the modified ResNet50 is then
    # used as a feature extractor in the Faster R-CNN model. The features extracted
    # by this backbone are then passed to the other components of the Faster R-CNN
    # (like the Region Proposal Network and the detection head) to perform object detection.

    # Reduce Size With 'Conv2d'. (Common, No FPN)

    # out_channels = 512
    # children = tuple(model.children())[:-2]  # must be specific of `avg_pool2d`.
    # backbone = nn.Sequential(*children, nn.Conv2d(in_features,
    #                                               out_channels=out_channels,
    #                                               kernel_size=1))
    #
    # backbone.out_channels = out_channels

    # End of modified models.

    backbone = model

    # modified: values
    # anchor_sizes = ((32, 64, 128, 256, 512,),)
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))  # max: 5 anchors
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # default: values
    # box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

    # Backbone With FPN. (ResNet50 Only)
    norm_layer = None
    extra_blocks = LastLevelMaxPool()

    # skipping avg_pool_2d, fc.
    # layers = tuple(dict(tuple(backbone.named_children())).keys())[:-2]
    layers = ["layer1", "layer2", "layer3", "layer4"]

    # start at 1.
    returned_layers = [i + 1 for i in range(len(layers))]
    return_layers = dict((k, str(i + 1)) for i, k in enumerate(layers))

    # In the source code of PyTorch, the line `in_channels_stage2 = backbone.inplanes // 8`
    # is used to calculate the number of input channels for the second stage of
    # the **Feature Pyramid Network (FPN)** in a Faster R-CNN model
    # with a ResNet50 backbone.

    # The `backbone.inplanes` variable stores the number of input channels
    # for the first convolutional layer of the ResNet50 backbone.
    # In the original ResNet50 architecture, `inplanes` is set to 64.
    # By dividing `backbone.inplanes` by 8, the code calculates the number
    # of input channels for the second stage of the FPN, which is one-eighth
    # of the original value.

    # This division by 8 is a design choice made in the FPN architecture
    # to reduce the number of channels and improve computational efficiency
    # while maintaining the quality of the feature maps.

    channels_stage = backbone.inplanes // 8
    in_channels_list = [channels_stage * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256

    # a FPN on top of a model. Internally,
    # it uses `torchvision.models._utils.IntermediateLayerGetter`
    # to extract a submodules that returns the feature maps specified
    # in `return_layers`. The same limitations
    # of IntermediateLayerGetter apply here.

    backbone_fpn = BackboneWithFPN(backbone=backbone,
                                   return_layers=return_layers,
                                   in_channels_list=in_channels_list,
                                   out_channels=out_channels,
                                   extra_blocks=extra_blocks,
                                   norm_layer=norm_layer)

    # Adds a simple RPN Head with classification and regression heads.

    # rpn_head = RPNHead(in_channels=backbone_fpn.out_channels,
    #                    num_anchors=rpn_anchor_generator.num_anchors_per_location()[0],
    #                    conv_depth=2)

    faster_rcnn = FasterRCNN(backbone=backbone_fpn,
                             num_classes=num_classes,
                             rpn_anchor_generator=rpn_anchor_generator,
                             # box_roi_pool=box_roi_pool,
                             box_score_thresh=0.0,
                             # rpn_head=rpn_head,
                             max_size=2048,
                             min_size=256)

    return faster_rcnn
