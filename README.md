# retico-objectFeatures

A retico module for extracting visual features from detected objects using CLIP and other vision models from Hugging 
Face transformers.

## Overview

The `retico-objectFeatures` module provides functionality to extract visual features from detected objects from other 
modules like [YOLO](https://github.com/retico-team/retico-yolo.git) or [SAM](https://github.com/retico-team/retico-sam.git). 
It uses state-of-the-art vision models including CLIP, EfficientNet, ResNet, ViT, and other transformers compatible 
models to generate meaningful feature representations from extracted objects.

## Installation and Requirements

To use the object features module you first need to install the retico-objectfeatures package:
* Install retico-objectfeatures:
```pip install retico-objectfeatures```

  or, for the latest development version from GitHub:
```pip install git+https://github.com/retico-team/retico-objectFeatures.git```

## Modules

### `ObjectFeaturesExtractor`
Subscribes to `ExtractedObjectsIU` and outputs `ObjectFeaturesIU`. For each of the first `top_objects` extracted objects, the object's image is passed through the vision model, and its pooled features are flattened into a 1D list. The features are stored in the IU's `object_features`, a dictionary from the object's index to its features. The model is loaded when the module starts running.

#### Arguments:
* `model_name` (str): the Hugging Face model to extract features with; defaults to `"openai/clip-vit-base-patch32"`
* `top_objects` (int): the number of objects to extract features from; defaults to `1`
* `timeout` (float): how long in seconds to wait before checking for new input again; defaults to `0.5`

### Example
**Note:** The example uses `WebcamModule` and `ExtractObjectsModule` from [retico-vision](https://github.com/retico-team/retico-vision), `DebugModule` from `retico_core` and `ObjectDetectionModule` from [retico-yolo](https://github.com/retico-team/retico-yolo) (`pip install git+https://github.com/retico-team/retico-yolo.git`).

```python
from retico_core.debug import DebugModule
from retico_vision.vision import WebcamModule, ExtractObjectsModule
from retico_yolo import ObjectDetectionModule
from retico_objectfeatures import ObjectFeaturesExtractor


webcam = WebcamModule()
yolo = ObjectDetectionModule()
extractor = ExtractObjectsModule(num_obj_to_display=5)
feats = ObjectFeaturesExtractor(top_objects=5)
debug = DebugModule()

webcam.subscribe(yolo)
yolo.subscribe(extractor)
extractor.subscribe(feats)
feats.subscribe(debug)

webcam.run()
yolo.run()
extractor.run()
feats.run()
debug.run()

print("Network is running")
input()

webcam.stop()
yolo.stop()
extractor.stop()
feats.stop()
debug.stop()
```

## Related Projects

- [ReTiCo Core](https://github.com/retico-team/retico-core) - The core ReTiCo framework
- [ReTiCo Vision](https://github.com/retico-team/retico-vision) - Vision components for ReTiCo
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - The transformers library used for vision models
