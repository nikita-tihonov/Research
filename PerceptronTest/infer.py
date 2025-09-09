import torch
from torch import nn
from torchvision.transforms import ToTensor, transforms
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class Perceptron(nn.Module):
    def __init__(self, num_classes):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(784, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, num_classes)

    def forward(self, x):
        out = x.reshape(x.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        return out


def predict_image(model, image_path, device, transform):

    image = cv2.imread(image_path, 0)
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

    image = transform(image)

    with torch.no_grad():
        image = image.to(device)
        output = model(image)
    
    return output

transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(1., ), std =(0.5, ))
                                ])


model = torch.load('root/perceptron.pth', weights_only=False)
model.eval() 
print("Модель загружена и готова к инференсу")

image_path = 'mnist_7567_label_2.png'

result = predict_image(model, image_path, device, transform)
print(f"Обнаружена цифра {torch.argmax(result).item()}")