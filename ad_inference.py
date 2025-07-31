# This is the demo code used for the inference of the artifact detector, based on the segformer backbone.
# DiffDoctor: Diagnosing Image Diffusion Models Before Treating

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch.nn as nn
import torch
import cv2
from torchvision import transforms
import numpy as np
import os


segformer_path = 'checkpoints/ad_pytorch_model.bin'
image_dir = 'asset/input'
device = 'cuda'

def get_segformer(path_or_hub, out_channels=1):
    # load a pretrained Segformer model
    preprocessor = SegformerImageProcessor.from_pretrained(path_or_hub)
    model = SegformerForSemanticSegmentation.from_pretrained(path_or_hub)
    # change the number of output channels
    model.decode_head.classifier = nn.Conv2d(model.decode_head.classifier.in_channels, out_channels, kernel_size=1)
    return preprocessor, model

if __name__ == '__main__':
    print('It\'s normal to have some warnings here. -- by author')
    seg_preprocessor, artifact_detector = get_segformer("nvidia/mit-b5", out_channels=1)
    artifact_detector.load_state_dict(torch.load(segformer_path))
    artifact_detector.to(device)
    artifact_detector.eval()

    image_paths = os.listdir(image_dir)
    image_paths = [path for path in image_paths if path.endswith('.jpg') or path.endswith('.png')]
    image_paths = [os.path.join(image_dir, path) for path in image_paths]



    for image_path in image_paths:
        image_name = os.path.basename(image_path).split('.')[0]
        image = cv2.imread(image_path)
        image = cv2.resize(image, (512, 512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        visualizable_image = image.copy()
        with torch.no_grad():
            image = transforms.ToTensor()(image).to(device)
            processed_images = seg_preprocessor(image, return_tensors='pt',do_rescale=False)['pixel_values'].to(device)
            pred = artifact_detector(processed_images)
            pred = nn.functional.interpolate(
                pred.logits, size=processed_images.shape[-2:], mode="bilinear", align_corners=False
            )
            normed_preds = torch.sigmoid(pred)

        mask = (normed_preds[0].detach().cpu().numpy().transpose(-2, -1, -3) * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        alpha = 0.6  # the transparency of the heatmap
        masked_img = cv2.addWeighted(heatmap_color, alpha, visualizable_image, 1 - alpha, 0)

        cv2.imwrite(f'asset/output/result_{image_name}.png', masked_img)