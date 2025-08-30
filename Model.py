import torch
from torchvision import transforms, models
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()
from torchvision.models import MobileNet_V2_Weights

# Load the model
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = torch.nn.Linear(model.last_channel, 1)
model.load_state_dict(torch.load('mobilenetv2_prescription_label.pth', map_location='cpu'))
model.eval()

# Define the same transforms as during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        prob_no_label = torch.sigmoid(output).item()
        prob_label = 1 - prob_no_label  # Invert the probability
        pred = (prob_label > 0.5)
    return pred, prob_label

# Example usage:
image_path = 'RxCameraApp-new/assets/IMG_7914.jpg'
pred, prob = predict(image_path)
if pred:
    print(f"Prediction: Label present (probability: {prob:.2f})")
else:
    print(f"Prediction: No label (probability: {1-prob:.2f})")
