from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch.nn as nn


def get_segformer(path_or_hub, out_channels=1):
    # load a pretrained Segformer model
    preprocessor = SegformerImageProcessor.from_pretrained(path_or_hub)
    model = SegformerForSemanticSegmentation.from_pretrained(path_or_hub)
    # change the number of output channels
    model.decode_head.classifier = nn.Conv2d(model.decode_head.classifier.in_channels, out_channels, kernel_size=1)
    return preprocessor, model
