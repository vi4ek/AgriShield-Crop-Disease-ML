import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, render_template, request, jsonify, url_for
from torchvision import transforms

from src.models.model import CropDiseaseModel
from src.models.gradcam import GradCAM

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ------------------------------------------------------------------------------
# 1. LOAD MODEL CHECKPOINT & CLASS NAMES
# ------------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'crop_disease_model (1).pth')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'src', 'models', 'crop_disease_model.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    CLASS_NAMES = checkpoint.get('class_names', [])
    model = CropDiseaseModel(num_classes=len(CLASS_NAMES), num_severities=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    grad_cam = GradCAM(model)
    print(f"Loaded AgriShield PyTorch Model from {MODEL_PATH} ({len(CLASS_NAMES)} classes, Val Acc: {checkpoint.get('val_acc', 0):.2f}%)")
else:
    model = None
    CLASS_NAMES = []
    grad_cam = None
    print("WARNING: Model weight file not found!")

# Build Crop Species to Class Indices Mapping for Species-Constrained Softmax Masking
CROP_SPECIES_MAP = {
    "All": list(range(len(CLASS_NAMES))),
    "Apple": [i for i, name in enumerate(CLASS_NAMES) if name.startswith("Apple")],
    "Blueberry": [i for i, name in enumerate(CLASS_NAMES) if name.startswith("Blueberry")],
    "Cherry": [i for i, name in enumerate(CLASS_NAMES) if name.startswith("Cherry")],
    "Corn (Maize)": [i for i, name in enumerate(CLASS_NAMES) if name.startswith("Corn")],
    "Grape": [i for i, name in enumerate(CLASS_NAMES) if name.startswith("Grape")],
    "Orange": [i for i, name in enumerate(CLASS_NAMES) if name.startswith("Orange")],
    "Peach": [i for i, name in enumerate(CLASS_NAMES) if name.startswith("Peach")],
    "Pepper": [i for i, name in enumerate(CLASS_NAMES) if name.startswith("Pepper")],
    "Potato": [i for i, name in enumerate(CLASS_NAMES) if name.startswith("Potato")],
    "Raspberry": [i for i, name in enumerate(CLASS_NAMES) if name.startswith("Raspberry")],
    "Soybean": [i for i, name in enumerate(CLASS_NAMES) if name.startswith("Soybean")],
    "Squash": [i for i, name in enumerate(CLASS_NAMES) if name.startswith("Squash")],
    "Strawberry": [i for i, name in enumerate(CLASS_NAMES) if name.startswith("Strawberry")],
    "Tomato": [i for i, name in enumerate(CLASS_NAMES) if name.startswith("Tomato")]
}

SEVERITY_LABELS = ["Early", "Moderate", "Severe"]

# Image Preprocessing Transform
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_species_from_class(class_name):
    raw_species = class_name.split("___")[0].replace("_", " ").strip()
    if "Pepper" in raw_species:
        return "Pepper"
    elif "Corn" in raw_species:
        return "Corn (Maize)"
    elif "Cherry" in raw_species:
        return "Cherry"
    return raw_species

def parse_crop_and_disease(class_name):
    """
    Parses class label (e.g. 'Pepper,_bell___Bacterial_spot') into clean, 
    separate crop name and disease name strings.
    """
    if "___" in class_name:
        crop_part, disease_part = class_name.split("___", 1)
    else:
        crop_part, disease_part = class_name, "Unknown"

    # Clean Crop Name
    crop_name = crop_part.replace("_", " ").strip()
    if crop_name == "Pepper, bell":
        crop_name = "Bell Pepper"
    elif crop_name == "Corn (maize)":
        crop_name = "Corn (Maize)"
    elif crop_name == "Cherry (including sour)":
        crop_name = "Cherry"
    
    # Clean Disease Name
    disease_name = disease_part.replace("_", " ").strip()
    if disease_name.endswith(" "):
        disease_name = disease_name.strip()

    if disease_name.lower() == "healthy":
        disease_name = "Healthy (No Disease)"
    elif "Cercospora" in disease_name and "Gray" in disease_name:
        disease_name = "Cercospora Leaf Spot / Gray Leaf Spot"
    elif "Spider mites" in disease_name:
        disease_name = "Spider Mites (Two-Spotted)"
    else:
        # Title case disease words (e.g., 'Apple scab' -> 'Apple Scab')
        disease_name = " ".join([word.capitalize() if not word.startswith("(") else word for word in disease_name.split()])
    
    return crop_name, disease_name

# ------------------------------------------------------------------------------
# 3. ROUTES
# ------------------------------------------------------------------------------
@app.route('/')
def index():
    crops = list(CROP_SPECIES_MAP.keys())
    return render_template('index.html', crops=crops)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please ensure crop_disease_model.pth exists.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    selected_species = request.form.get('crop_species', 'All')

    try:
        # 1. Save uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)
        pil_img = Image.open(image_path).convert('RGB')

        # 2. Preprocess
        input_tensor = eval_transform(pil_img).unsqueeze(0).to(device)

        # 3. Forward Pass
        disease_logits, severity_logits = model(input_tensor)

        # Global (Unconstrained) Top-1 Prediction
        global_probs = F.softmax(disease_logits, dim=1)[0]
        global_top1_idx = torch.argmax(global_probs).item()
        global_top1_class = CLASS_NAMES[global_top1_idx]
        global_confidence = global_probs[global_top1_idx].item()
        global_species = extract_species_from_class(global_top1_class)

        # 4. Objective 3: Species-Constrained Softmax Masking
        valid_indices = CROP_SPECIES_MAP.get(selected_species, list(range(len(CLASS_NAMES))))
        
        masked_logits = disease_logits.clone()
        if selected_species != "All":
            mask = torch.full_like(masked_logits, -1e9)
            mask[0, valid_indices] = masked_logits[0, valid_indices]
            masked_logits = mask

        constrained_probs = F.softmax(masked_logits, dim=1)[0]
        constrained_top1_idx = torch.argmax(constrained_probs).item()
        predicted_class = CLASS_NAMES[constrained_top1_idx]
        confidence = constrained_probs[constrained_top1_idx].item()

        # Parse crop name and disease name separately
        crop_name, disease_name = parse_crop_and_disease(predicted_class)

        # 5. Objective 4: Confidence-Based Crop Mismatch Alert Engine (45% Threshold)
        mismatch_alert = False
        suggested_species = None

        if selected_species != "All":
            if confidence < 0.45 or global_species != selected_species:
                mismatch_alert = True
                suggested_species = global_species

        # 6. Severity Estimation
        severity_probs = F.softmax(severity_logits, dim=1)[0]
        severity_idx = torch.argmax(severity_probs).item()
        severity_stage = SEVERITY_LABELS[severity_idx]

        # 7. Objective 5: Grad-CAM Explainable AI Visual Heatmap Overlay
        cam_map = grad_cam.generate_heatmap(input_tensor, target_class=constrained_top1_idx)
        heatmap_pil = grad_cam.overlay_heatmap(pil_img, cam_map, alpha=0.5)

        heatmap_filename = f"heatmap_{file.filename}"
        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
        heatmap_pil.save(heatmap_path)

        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'crop_name': crop_name,
            'disease_name': disease_name,
            'confidence': round(confidence * 100, 2),
            'severity_stage': severity_stage,
            'severity_confidence': round(severity_probs[severity_idx].item() * 100, 2),
            'selected_species': selected_species,
            'detected_species': global_species,
            'mismatch_alert': mismatch_alert,
            'suggested_species': suggested_species,
            'original_image_url': url_for('static', filename=f'uploads/{file.filename}'),
            'heatmap_image_url': url_for('static', filename=f'uploads/{heatmap_filename}')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print(f"\n=======================================================")
    print(f" AgriShield Web App is Running!")
    print(f" Local Access:   http://127.0.0.1:{port}")
    print(f" Network (LAN):  http://0.0.0.0:{port}")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=True)


