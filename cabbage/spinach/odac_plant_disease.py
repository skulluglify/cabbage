#!/usr/bin/env python3
import os
from typing import Any, Callable, Dict, List, IO

import matplotlib

import sodium
from .odac_plant_disease_predictor import ODACPlantDiseasePredictor


def odac_plant_disease_predictor(fn: Callable[..., Any]) -> Callable[..., List[Dict[str, Any]]]:
    cwd = os.path.dirname(os.path.abspath(__file__))
    sodium.init(debug=False)

    matplotlib.use('agg', force=True)

    classification_classes = [
        'Alternaria Leaf Spot', 'Angular Leaf Spot', 'Ascochyta Leaf Spot',
        'Cercospora Leaf Spot', 'Cladosporium Leaf Spot', 'Coniothyrium',
        'Corynespora Leaf Spot', 'Downy Mildew', 'Guignardia Leaf Spot',
        'Gynosporanium', 'Leptosphaeria', 'Leptosphaerulina',
        'Marssonina', 'Mycosphaerella', 'Pestalotiopsis',
        'Phacidium', 'Phloeospora', 'Phoma',
        'Phomopsis', 'Phyllosticta', 'Powdery Mildew',
        'Pseudocercospora', 'Puccinia', 'Rhabdocline',
        'Septoria', 'Stemphylium', 'Stigmina',
        'Tracheomycosis', 'Xanthomonas', 'Zonate Leaf Spot',
    ]

    detection_classes = [
        'Bok-Choy-Diseased',
        'Bok-Choy-Ready-to-Harvest',
        'Lettuce-Diseased',
        'Lettuce-Ready-to-Harvest',
        'Spinach-Diseased',
        'Spinach-Ready-to-Harvest',
    ]

    odac = ODACPlantDiseasePredictor(
        detection_classes=detection_classes,
        detection_model_path=os.path.join(cwd, '../torch/models/detection'),
        classification_classes=classification_classes,
        classification_model_path=os.path.join(cwd, '../torch/models/classification'))
    odac_outputs_dir = os.path.join(cwd, '../data/odac/plant/prediction/outputs')

    def wrapper(self, *args, fp: str | bytes | os.PathLike[str] | IO[bytes], **kwargs):
        return fn(self, *args, odac.predict_save_image(fp, directory=odac_outputs_dir), **kwargs)

    return wrapper
