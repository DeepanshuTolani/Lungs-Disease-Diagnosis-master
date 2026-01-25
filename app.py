# # fast api application for x-ray image classification

# import torch
# from xray.ml.model.arch import Net  # Ensure this path is correct
# import torchvision.transforms as transforms
# from PIL import Image
# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles


# app = FastAPI()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Initialize and load model weights
# model = Net().to(device)
# model.load_state_dict(torch.load("xray_model.pth", map_location=device))
# model.eval()

# # Transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# # Prediction label mapping
# label_map = {
#     0: "Normal",
#     1: "Pneumonia"
# }

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     image = Image.open(file.file).convert("RGB")
#     input_tensor = transform(image).unsqueeze(0).to(device)
    
#     # with torch.no_grad():
#     #     output = model(input_tensor)
#     #     prediction_index = torch.argmax(output, dim=1).item()
#     #     prediction_label = label_map.get(prediction_index, "Unknown")
#     with torch.no_grad():
#         output = model(input_tensor)
#         probabilities = torch.softmax(output, dim=1)
#         prediction_index = torch.argmax(probabilities, dim=1).item()
#         confidence = probabilities[0][prediction_index].item()
#         prediction_label = label_map.get(prediction_index, "Unknown")

    
#     # return {
#     #     "prediction_index": prediction_index,
#     #     "prediction_label": prediction_label
#     # }
#     return {
#         "prediction_index": prediction_index,
#         "prediction_label": prediction_label,
#         "confidence": round(confidence * 100, 2)
# }
import torch
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Import your specific model architecture
from xray.ml.model.arch import Net 

app = FastAPI()

# Setup templates and static files (CSS/JS/Images)
# Make sure you have a folder named 'templates' with index.html in it
# templates = Jinja2Templates(directory="templates")
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Configuration
MODEL_PATH = "xray_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load model weights
model = Net().to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded successfully on {device}")
except FileNotFoundError:
    print(f"Warning: {MODEL_PATH} not found. Ensure the weight file is in the root directory.")

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Optional: Add normalization if your model was trained with it
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Prediction label mapping
label_map = {
    0: "Normal",
    1: "Pneumonia"
}

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Renders the upload page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Handles image upload and returns classification results."""
    # 1. Load and process image
    image = Image.open(file.file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 2. Run Inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        
        # Get result details
        prediction_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction_index].item()
        prediction_label = label_map.get(prediction_index, "Unknown")

    # 3. Return results
    return {
        "prediction_index": prediction_index,
        "prediction_label": prediction_label,
        "confidence": f"{round(confidence * 100, 2)}%"
    }
