import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)

model.load_state_dict(torch.load("weights/best_segmentation_model.pth", map_location=device))
model.eval()
print("Модель загружена и готова к инференсу")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_single_image(model, image_path, device, transform, threshold=0.5):

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)

    pred_mask = output.squeeze().cpu().numpy()
    pred_mask = (pred_mask > threshold).astype(np.uint8)
    pred_mask = cv2.resize(pred_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

    return image, pred_mask

def visualize_prediction(original_image, predicted_mask, save_path=None):

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    overlay = original_image.copy()
    overlay[predicted_mask == 1] = [255, 0, 0]
    alpha = 0.6
    blended = cv2.addWeighted(original_image, 1 - alpha, overlay, alpha, 0)
    plt.imshow(blended)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Результат сохранён: {save_path}")
    plt.show()

image_path = "data/IRSTD1k_Img/XDU500.png"

original_img, pred_mask = predict_single_image(
    model=model,
    image_path=image_path,
    device=device,
    transform=transform,
    threshold=0.5
)

visualize_prediction(original_img, pred_mask, save_path="results/prediction_result_500.png")