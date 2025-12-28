"""
Flask Backend cho Email Phishing Detection Extension
Load models sẵn khi server start để tối ưu tốc độ
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

# Thêm project root vào path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import XAI modules
from notebooks.XAI.model_loader import ModelLoader
from notebooks.XAI.lime_explainer import LIMEExplainer
from notebooks.XAI.shap_explainer import SHAPExplainer

app = Flask(__name__)
CORS(app)  # Bật CORS cho extension

# Biến toàn cục để lưu models và explainers
model_loader = None
lime_explainer = None
shap_explainer = None
models_loaded = False

# Model paths
MODELS_BASE_PATH = os.path.join(project_root, "output", "models")


def load_models():
    """Load tất cả models khi server start"""
    global model_loader, lime_explainer, shap_explainer, models_loaded
    
    if models_loaded:
        return
    
    try:
        print("="*80)
        print("ĐANG LOAD MODELS...")
        print("="*80)
        print(f"📁 Project root: {project_root}")
        print(f"📁 Models base path: {MODELS_BASE_PATH}")
        print(f"📁 Models base path (absolute): {os.path.abspath(MODELS_BASE_PATH)}")
        print(f"📁 Models base path exists: {os.path.exists(MODELS_BASE_PATH)}")
        print()
        
        # Load model loader
        model_loader = ModelLoader(base_path=MODELS_BASE_PATH)
        models = model_loader.load_all_models()
        
        print(f"\nĐã load {len(models)} models: {', '.join(models.keys())}")
        
        # Khởi tạo LIME explainer cho Keras models
        lime_explainer = LIMEExplainer(model_loader)
        print("LIME explainer đã được khởi tạo")
        
        # Khởi tạo SHAP explainer cho BERT (English)
        if "BERT" in models:
            bert_path = os.path.join(MODELS_BASE_PATH, "BERT", "bert_base_email_model")
            if os.path.exists(bert_path):
                try:
                    shap_explainer = SHAPExplainer(bert_path)
                    print("✓ SHAP explainer đã được khởi tạo cho BERT")
                    # Warm-up để lần gọi thực tế đầu tiên nhanh hơn
                    shap_explainer.warmup()
                except Exception as e:
                    print(f"⚠ Không thể khởi tạo SHAP explainer cho BERT: {e}")
                    shap_explainer = None
            else:
                print(f"⚠ Không tìm thấy BERT model tại {bert_path}")
                shap_explainer = None
        else:
            shap_explainer = None
        
        models_loaded = True
        
        print("="*80)
        print("✓ TẤT CẢ MODELS ĐÃ SẴN SÀNG!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ LỖI KHI LOAD MODELS: {e}")
        traceback.print_exc()
        raise


@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint kiểm tra trạng thái server"""
    return jsonify({
        'status': 'ok',
        'models_loaded': models_loaded,
        'models': list(model_loader.models.keys()) if model_loader else []
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint để dự đoán email với tất cả models đang được load
    
    Request body:
    {
        "email_text": "subject + body của email"
    }
    
    Response:
    {
        "language": "unknown",
        "predictions": {
            "GRU": {"label": "phishing", "probability": 0.95},
            "CNN": {"label": "benign", "probability": 0.87},
            ...
        }
    }
    """
    if not models_loaded:
        return jsonify({
            'error': 'Models chưa được load. Vui lòng đợi...'
        }), 503
    
    try:
        data = request.get_json()
        email_text = data.get('email_text', '').strip()
        
        if not email_text:
            return jsonify({
                'error': 'email_text không được để trống'
            }), 400

        # Không tự phát hiện ngôn ngữ; chạy tất cả models đang có
        selected_models = list(model_loader.models.keys())
        print(f"📌 Chọn models: {selected_models}")
        
        # Dự đoán với các models đã chọn
        predictions = {}
        
        for model_name in selected_models:
            if model_name not in model_loader.models:
                print(f"⚠ Model {model_name} không có sẵn, bỏ qua...")
                continue
                
            try:
                label, probability = model_loader.predict(model_name, email_text)
                predictions[model_name] = {
                    'label': label,
                    'probability': float(probability)
                }
            except Exception as e:
                print(f"Lỗi khi predict với {model_name}: {e}")
                predictions[model_name] = {
                    'label': 'error',
                    'probability': 0.0,
                    'error': str(e)
                }
        
        return jsonify({
            'language': 'unknown',
            'predictions': predictions,
            'email_text': email_text[:100] + '...' if len(email_text) > 100 else email_text
        })
        
    except Exception as e:
        print(f"Lỗi trong /api/predict: {e}")
        traceback.print_exc()
        return jsonify({
            'error': f'Lỗi khi dự đoán: {str(e)}'
        }), 500


@app.route('/api/explain', methods=['POST'])
def explain():
    """
    API endpoint để tạo XAI explanation
    
    Request body:
    {
        "model_name": "GRU" | "CNN" | "BiLSTM" | "Hybrid_CNN_BiLSTM" | "BERT",
        "email_text": "subject + body của email"
    }
    
    Response:
    {
        "model_name": "GRU",
        "prediction_label": "phishing",
        "prediction_probability": 0.95,
        "important_tokens": [
            {"token": "verify", "weight": 0.123},
            ...
        ]
    }
    """
    if not models_loaded:
        return jsonify({
            'error': 'Models chưa được load. Vui lòng đợi...'
        }), 503
    
    try:
        data = request.get_json()
        model_name = data.get('model_name', '').strip()
        email_text = data.get('email_text', '').strip()
        mode = data.get('mode', 'quick').strip().lower()  # 'quick' hoặc 'full'
        
        if not model_name or not email_text:
            return jsonify({
                'error': 'model_name và email_text không được để trống'
            }), 400
        
        if model_name not in model_loader.models:
            return jsonify({
                'error': f'Model {model_name} không tồn tại. Các models có sẵn: {list(model_loader.models.keys())}'
            }), 400
        
        # Tạo giải thích
        if model_name == "BERT":
            # BERT (English) sử dụng SHAP để giải thích
            if shap_explainer is None:
                return jsonify({
                    'error': 'SHAP explainer chưa được khởi tạo cho BERT'
                }), 503
            
            try:
                # Quick mode dùng explain_with_shap_fast, full mode dùng explain_with_shap đầy đủ
                if mode == 'full':
                    result = shap_explainer.explain_with_shap(
                        email_text,
                        max_features=15
                    )
                else:
                    result = shap_explainer.explain_with_shap_fast(
                        email_text,
                        max_features=15
                    )
                
                return jsonify({
                    'model_name': model_name,
                    'prediction_label': result['prediction_label'],
                    'prediction_probability': result['prediction_probability'],
                    'important_tokens': result['important_tokens'],
                    'method': 'SHAP'
                })
            except Exception as e:
                print(f"Lỗi khi tạo SHAP explanation: {e}")
                traceback.print_exc()
                return jsonify({
                    'error': f'Lỗi khi tạo SHAP explanation: {str(e)}'
                }), 500
        else:
            # Các models khác sử dụng LIME để giải thích
            if lime_explainer is None:
                return jsonify({
                    'error': 'LIME explainer chưa được khởi tạo'
                }), 503
            
            try:
                # Quick mode dùng ít samples, full mode dùng nhiều samples hơn
                num_samples = 1000 if mode != 'full' else 3000
                result = lime_explainer.explain_with_lime(
                    model_name,
                    email_text,
                    num_features=15,
                    num_samples=num_samples
                )
                
                return jsonify({
                    'model_name': model_name,
                    'prediction_label': result['prediction_label'],
                    'prediction_probability': result['prediction_probability'],
                    'important_tokens': result['important_tokens'],
                    'method': 'LIME'
                })
            except Exception as e:
                print(f"Lỗi khi tạo LIME explanation: {e}")
                traceback.print_exc()
                return jsonify({
                    'error': f'Lỗi khi tạo LIME explanation: {str(e)}'
                }), 500
        
    except Exception as e:
        print(f"Lỗi trong /api/explain: {e}")
        traceback.print_exc()
        return jsonify({
            'error': f'Lỗi khi tạo explanation: {str(e)}'
        }), 500


if __name__ == '__main__':
    # Load models trước khi khởi động server
    print("\n🚀 Khởi động Flask Backend...")
    load_models()
    
    # Khởi động Flask server
    print("\n🌐 Server đang chạy tại http://localhost:5000")
    print("📡 API endpoints:")
    print("   - GET  /api/health  - Health check")
    print("   - POST /api/predict - Dự đoán email với tất cả models")
    print("   - POST /api/explain  - Tạo XAI explanation\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

