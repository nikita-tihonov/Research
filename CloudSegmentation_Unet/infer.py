import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import segmentation_models_pytorch as smp
from pathlib import Path
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=4,
    classes=2
).to(device)

model.load_state_dict(torch.load("weights/unet_cloud_25epochs.pth", map_location=device))
model.eval()
print("Модель загружена и готова к инференсу")


def predict_single_image(model, r_dir, g_dir, b_dir, nir_dir, device, invert=True):

    image_r = np.array(Image.open(r_dir))
    image_g = np.array(Image.open(g_dir))
    image_b = np.array(Image.open(b_dir))
    image_nir = np.array(Image.open(nir_dir))

    raw_rgbi = np.stack([image_r,
                        image_g,
                        image_b,
                        image_nir
                        ], axis=2)

    if invert:
        raw_rgbi = raw_rgbi.transpose((2,0,1))

    input_tensor = torch.tensor((raw_rgbi / np.iinfo(raw_rgbi.dtype).max), dtype=torch.float32)

    input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    return image_r, image_g, image_b, image_nir, pred_mask


def visualize_prediction(r_img, g_img, b_img, nir_img, predicted_mask, save_path=None):

    plt.figure(figsize=(25, 5))

    plt.subplot(1, 5, 1)
    plt.imshow(r_img)
    plt.title("Red band image")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(g_img)
    plt.title("Blue band image")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(b_img)
    plt.title("Green band image")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(nir_img)
    plt.title("NIR band image")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Результат сохранён: {save_path}")
    plt.show()


base_path = Path('/root/.cache/kagglehub/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images/versions/4/38-Cloud_test')

r_dir=base_path/'test_red/red_patch_5_1_by_5_LC08_L1TP_003052_20160120_20170405_01_T1.TIF'
g_dir=base_path/'test_green/green_patch_5_1_by_5_LC08_L1TP_003052_20160120_20170405_01_T1.TIF'
b_dir=base_path/'test_blue/blue_patch_5_1_by_5_LC08_L1TP_003052_20160120_20170405_01_T1.TIF'
nir_dir=base_path/'test_nir/nir_patch_5_1_by_5_LC08_L1TP_003052_20160120_20170405_01_T1.TIF'

r_img, g_img, b_img, nir_img, pred_mask = predict_single_image(
    model=model,
    r_dir=r_dir,
    g_dir=g_dir,
    b_dir=b_dir,
    nir_dir=nir_dir,
    device=device
)


visualize_prediction(r_img, g_img, b_img, nir_img, pred_mask, save_path="results/prediction.png")