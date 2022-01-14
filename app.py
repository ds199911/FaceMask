from flask import Flask, jsonify, request, render_template
import io
import json
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
imagenet_index = json.load(open('imagenet_class_index.json'))


model = models.googlenet(True)
def allowed_file(filename):
    return '.' in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template('index.html')

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
    pred = model_output.max(1)[1]
    return imagenet_index[str(pred.item())]

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            image_bytes = file.read()
            class_id, class_name = prediction(image_bytes)
            return jsonify({'class id': class_id, 'class name': class_name})
        
        return "Could not Predict"

if __name__ == '__main__':
    app.run()
