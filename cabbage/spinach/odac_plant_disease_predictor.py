#!/usr/bin/env python3
import os
from tempfile import NamedTemporaryFile
from typing import Iterator, Tuple, IO, List, Dict, Any

from PIL import Image
from torch import Tensor

from sodium.models.classification.resnet import create_cnn_model_resnet_trainable
from sodium.models.detection.faster_rcnn import create_faster_rcnn_model_resnet_trainable
from sodium.torch_ex.extra.utils import nms, crop_by_predictions
from sodium.torch_ex.models.classification import LitAutoModel as LitAutoClassificationModel
from sodium.torch_ex.models.detection import LitAutoModel as LitAutoDetectionModel
from sodium.torch_ex.utils import cvt_u8_to_f32_tensor
from sodium.transforms import CenterCrop, Resize
from sodium.utils import (cvt_image_to_tensor, data_prediction_normalize_dict, cvt_tensor_to_image,
                          auto_resize_keep_resolution)
from sodium.utils_ex import draw_boxing_boxes_colored_ex


class ODACPlantDiseasePredictor:
    detection_classes: List[str] | Tuple[str]
    detection_model_path: str | os.PathLike[str]
    detection: LitAutoDetectionModel

    classification_classes: List[str] | Tuple[str]
    classification_model_path: str | os.PathLike[str]
    classification: LitAutoDetectionModel

    def __init__(
            self,
            detection_classes: List[str] | Tuple[str],
            detection_model_path: str | os.PathLike[str],
            classification_classes: List[str] | Tuple[str],
            classification_model_path: str | os.PathLike[str],
    ):
        # fixed start index class in detection model!
        if detection_classes[0] != '__background__':
            detection_classes = ['__background__'] + detection_classes

        self.detection_classes = detection_classes
        self.detection_model_path = detection_model_path
        self.classification_classes = classification_classes
        self.classification_model_path = classification_model_path

        self.init()

    def init(self):
        classification_num_classes = len(self.classification_classes)
        classification_model = create_cnn_model_resnet_trainable(num_classes=classification_num_classes)
        self.classification = LitAutoClassificationModel(classification_model)
        self.classification.try_checkpoint_load(self.classification_model_path)

        detection_num_classes = len(self.detection_classes)
        detection_model = create_faster_rcnn_model_resnet_trainable(num_classes=detection_num_classes)
        self.detection = LitAutoDetectionModel(detection_model)
        self.detection.try_checkpoint_load(self.detection_model_path)

    def oci(self, inp: Tensor) -> Iterator[Tuple[str, float]]:
        classes = self.classification_classes
        model = self.classification

        inp = inp.to(model.device)
        batch = [inp]

        predictions = model.evaluation(batch, 0)
        for prediction in predictions:
            idx, score = prediction
            yield classes[idx], float(score)

    def odi(self, inp: Tensor) -> Iterator[Tuple[Tensor, str, float]]:
        classes = self.detection_classes
        model = self.detection

        inp = inp.to(model.device)
        batch = [inp]

        predictions = model.evaluation(batch, 0)

        for prediction in predictions:
            prediction = nms(prediction, iou_threshold=0.8)

            labels: Tensor = prediction["labels"]
            scores: Tensor = prediction["scores"]
            boxes: Tensor = prediction["boxes"]

            if len(scores) > 0:
                predictions = data_prediction_normalize_dict(classes, labels, scores, boxes)
                yield predictions

    def predict(self, fp: str | bytes | os.PathLike[str] | IO[bytes]) -> List[Dict[str, Any]]:
        img = Image.open(fp)
        img = img.convert('RGB')

        inp = cvt_image_to_tensor(img)
        inp = cvt_u8_to_f32_tensor(inp)

        results = []

        for predictions in self.odi(inp):
            temp = {}

            results.append(temp)
            temp['image'] = img

            box = draw_boxing_boxes_colored_ex(inp, predictions, [])
            image_data_boxes = cvt_tensor_to_image(box)
            temp['image_data_boxes'] = image_data_boxes

            many_pred = []
            temp['predictions'] = many_pred

            for pred in crop_by_predictions(inp, predictions):
                out, label, confidence = pred
                image = cvt_tensor_to_image(out)

                if label.endswith('Harvest'):
                    many_pred.append(dict(
                        image=image,
                        label=label,
                        confidence=confidence,
                        diseases=[],
                        diseased=False,
                        harvest=True,
                    ))

                    continue

                if label.endswith('Diseased'):
                    size = min(image.width, image.height)

                    # fix transforms image on classification!
                    out = CenterCrop(size=(size, size))(out)
                    out = Resize(size=(224, 244), antialias=False)(out)

                    classify = []
                    for prediction in self.oci(out):
                        prediction = dict(zip(('label', 'confidence'), prediction))
                        classify.append(prediction)

                    many_pred.append(dict(
                        image=image,
                        label=label,
                        confidence=confidence,
                        diseases=classify,
                        diseased=True,
                        harvest=False,
                    ))

                    continue

        return results

    def predict_save_image(
            self,
            fp: str | bytes | os.PathLike[str] | IO[bytes],
            directory: str | os.PathLike[str] = 'data/odc/plant/prediction/outputs',
            image_max_width: int = 1080,
            image_max_height: int = 720,
    ) -> List[Dict[str, Any]]:
        """
        Example:
        .. highlight:: python
        .. code-block:: python
            schemas = [
                {
                    'image': 'pil.image',
                    'image_data_boxes': 'pil.image',
                    'predictions': [
                        {
                            'image': 'pil.image',
                            'label': 'string',
                            'confidence': 'float',
                            'diseases': [
                                {
                                    'label': 'string',
                                    'confidence': 'float',
                                }
                            ],
                            'diseased': '',
                            'harvest': '',
                        },
                    ],
                }
            ]
        ..
        :param fp:
        :param directory:
        :param image_max_width:
        :param image_max_height:
        :return:
        """

        # make directory if not exists!
        os.makedirs(directory, exist_ok=True)
        schemas = self.predict(fp)

        for i, schema in enumerate(schemas):
            with NamedTemporaryFile(mode='wb', prefix='odac-plant-disease-image-sample-', suffix='.png', dir=directory, delete=False) as stream:
                image_data_boxes: Image.Image

                del schema['image']

                image_data_boxes = schema.get('image_data_boxes')
                predictions = schema.get('predictions')

                stream.seek(0)
                stream.truncate(0)

                image_data_boxes.convert(mode='RGB')
                image_data_boxes = auto_resize_keep_resolution(
                    image_data_boxes, max_width=image_max_width,
                    max_height=image_max_height)
                image_data_boxes.save(stream, format='PNG', optimize=True)
                schema['image_data_boxes'] = os.path.relpath(stream.name, directory)

                prediction: Dict[str, Any]
                for j, prediction in enumerate(predictions):
                    del prediction['image']

                    label = prediction.get('label')
                    confidence = prediction.get('confidence')
                    diseases = prediction.get('diseases')
                    diseased = prediction.get('diseased')
                    harvest = prediction.get('harvest')

                    # buffer = io.BytesIO()
                    # image = auto_resize_keep_resolution(image, max_width=image_max_width, max_height=image_max_height)
                    # image = image.convert(mode='RGB')
                    # image.save(buffer, format='PNG', optimize=True)

                    prediction['label'] = label
                    prediction['confidence'] = confidence
                    prediction['diseases'] = diseases
                    prediction['diseased'] = diseased
                    prediction['harvest'] = harvest

                    predictions[j] = prediction
                schema['predictions'] = predictions
            schemas[i] = schema
        return schemas
