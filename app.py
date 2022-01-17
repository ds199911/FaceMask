from flask import Flask, jsonify, request
import io
import json
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

#load model
model = models.detection.fasterrcnn_resnet50_fpn(False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
model.load_state_dict(torch.load('model.pt', map_location='cpu'))
model.eval()
# model.to(device)




def allowed_file(filename):
    return '.' in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    return 'Mask Detection'

def image_transformation(image_bytes):
    image_transformations = transforms.Compose([transforms.Resize(255),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485,0.456,0.406),
                            (0.229, 0.224, 0.225))])
    upload_image = Image.open(io.BytesIO(image_bytes))
    return image_transformations(upload_image).unsqueeze(0)


def prediction(image_bytes):
    tensor = image_transformation(image_bytes)
    model_output = model.forward(tensor)
    pred = model_output
    print(pred)
    return pred

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            image_bytes = file.read()
            pred = prediction(image_bytes)
            return jsonify({'class id': pred })
        
        return "Could not Predict"

if __name__ == '__main__':
    app.run()
