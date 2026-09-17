import os
import numpy as np
import tensorflow as tf
import cv2
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
from io import BytesIO
from PIL import Image

from src import config
from src.gradcam import GradCAM
from src.models import MODEL_BUILDERS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Modelleri onbellege alalım (Hız için)
loaded_models = {}

@app.route('/', methods=['GET'])
def index():
    return send_from_directory('web', 'index.html')

def get_model(model_name):
    if model_name not in loaded_models:
        print(f"[*] Model dosyasi kontrol ediliyor: {model_name}")
        ckpt_path = config.CHECKPOINT_DIR / f"{model_name}_best.keras"
        try:
            # Rebuild strategy
            model = MODEL_BUILDERS[model_name]()
            print(f"[*] Mimari insa edildi. Agirliklar yukleniyor...")
            model.load_weights(ckpt_path)
            
            # Tahmin sirasinda BatchNormalization katmanlarinin rastgele davranmasini onlemek icin
            # model.trainable = False yapmak bazen inference kararliligini artirir.
            model.trainable = False
            
            loaded_models[model_name] = model
            print(f"[OK] {model_name} basariyla yuklendi ve donduruldu.")
        except Exception as e:
            print(f"[!] HATA: {model_name} yuklenemedi: {e}")
            import traceback
            traceback.print_exc()
            return None
    return loaded_models[model_name]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Resim gonderilmedi'}), 400
    
    file = request.files['image']
    model_name = request.form.get('model', 'CustomCNN')
    
    # Resmi oku
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Model girdisi icin hazirla
    img_resized = cv2.resize(img_rgb, (config.IMG_SIZE, config.IMG_SIZE))
    img_tensor = img_resized.astype(np.float32) / 255.0
    
    # Model yukle ve tahmin et
    model = get_model(model_name)
    if model is None:
        return jsonify({'error': 'Model yuklenemedi'}), 500
        
    preds = model.predict(np.expand_dims(img_tensor, axis=0), verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx])
    label = config.CLASS_NAMES[pred_idx]
    
    # Grad-CAM Heatmap
    gradcam = GradCAM(model)
    heatmap, _, _ = gradcam.compute_heatmap(img_tensor, class_idx=pred_idx)
    overlayed = gradcam.overlay(img_tensor, heatmap)
    
    # Gorselleri Base64'e cevir (Frontend'e gondermek icin)
    def to_base64(img_array):
        # Image is 0-1 float, convert to 0-255 uint8
        img_uint8 = (img_array * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8)
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    results = {
        'label': label,
        'confidence': f"{confidence:.2%}",
        'original': to_base64(img_tensor),
        'heatmap': to_base64(cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET) / 255.0),
        'overlay': to_base64(overlayed / 255.0 if overlayed.max() > 1.0 else overlayed),
        'all_probs': {config.CLASS_NAMES[i]: f"{float(preds[i]):.2%}" for i in range(len(preds))}
    }
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
