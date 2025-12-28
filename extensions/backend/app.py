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
from langdetect import detect, LangDetectException

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
shap_explainer_vietnamese = None
models_loaded = False

# Model paths
MODELS_BASE_PATH = os.path.join(project_root, "output", "models")


def load_models():
    """Load tất cả models khi server start"""
    global model_loader, lime_explainer, shap_explainer, shap_explainer_vietnamese, models_loaded
    
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
        
        print(f"\n✓ Đã load {len(models)} models: {', '.join(models.keys())}")
        
        # Khởi tạo LIME explainer cho Keras models
        lime_explainer = LIMEExplainer(model_loader)
        print("✓ LIME explainer đã được khởi tạo")
        
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
        
        # Khởi tạo SHAP explainer cho BERT_Vietnamese
        if "BERT_Vietnamese" in models:
            bert_vietnamese_path = os.path.join(MODELS_BASE_PATH, "BERT_Vietnamese", "phobert_vietnamese_email_model")
            if os.path.exists(bert_vietnamese_path):
                try:
                    shap_explainer_vietnamese = SHAPExplainer(bert_vietnamese_path)
                    print("✓ SHAP explainer đã được khởi tạo cho BERT_Vietnamese")
                    # Warm-up PhoBERT luôn khi server start để tránh chờ lâu lần đầu
                    shap_explainer_vietnamese.warmup()
                except Exception as e:
                    print(f"⚠ Không thể khởi tạo SHAP explainer cho BERT_Vietnamese: {e}")
                    shap_explainer_vietnamese = None
            else:
                print(f"⚠ Không tìm thấy BERT_Vietnamese model tại {bert_vietnamese_path}")
                shap_explainer_vietnamese = None
        else:
            shap_explainer_vietnamese = None
        
        models_loaded = True
        
        print("="*80)
        print("✓ TẤT CẢ MODELS ĐÃ SẴN SÀNG!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ LỖI KHI LOAD MODELS: {e}")
        traceback.print_exc()
        raise


def detect_language(text: str) -> str:
    """
    Phát hiện ngôn ngữ của email text
    
    Args:
        text: Nội dung email cần phát hiện ngôn ngữ
        
    Returns:
        'vi' nếu là tiếng Việt, 'en' nếu là tiếng Anh, 'unknown' nếu không xác định được
    """
    if not text or len(text.strip()) == 0:
        return 'unknown'
    
    try:
        # Sử dụng langdetect để phát hiện ngôn ngữ
        # Lấy sample đầu tiên 500 ký tự để tăng tốc độ
        sample_text = text[:500] if len(text) > 500 else text
        detected_lang = detect(sample_text)
        
        # Chuyển đổi mã ngôn ngữ về định dạng chuẩn
        if detected_lang == 'vi':
            return 'vi'
        elif detected_lang == 'en':
            return 'en'
        else:
            # Nếu không phải vi hoặc en, kiểm tra thêm bằng cách đếm ký tự đặc biệt tiếng Việt
            vietnamese_chars = set('àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ')
            text_lower = text.lower()
            vietnamese_char_count = sum(1 for char in text_lower if char in vietnamese_chars)
            
            # Nếu có nhiều ký tự tiếng Việt, coi như tiếng Việt
            if vietnamese_char_count > 5:
                return 'vi'
            else:
                return 'en'  # Mặc định là tiếng Anh nếu không xác định được
                
    except LangDetectException:
        # Nếu langdetect không phát hiện được, kiểm tra ký tự tiếng Việt
        vietnamese_chars = set('àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ')
        text_lower = text.lower()
        vietnamese_char_count = sum(1 for char in text_lower if char in vietnamese_chars)
        
        if vietnamese_char_count > 5:
            return 'vi'
        else:
            return 'en'
    except Exception as e:
        print(f"Lỗi khi phát hiện ngôn ngữ: {e}")
        return 'unknown'


@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint kiểm tra trạng thái server"""
    return jsonify({
        'status': 'ok',
        'models_loaded': models_loaded,
        'models': list(model_loader.models.keys()) if model_loader else []
    })


@app.route('/api/detect-language', methods=['POST'])
def detect_language_endpoint():
    """
    API endpoint để phát hiện ngôn ngữ của email
    
    Request body:
    {
        "email_text": "subject + body của email"
    }
    
    Response:
    {
        "language": "vi" | "en" | "unknown"
    }
    """
    try:
        data = request.get_json()
        email_text = data.get('email_text', '').strip()
        
        if not email_text:
            return jsonify({
                'error': 'email_text không được để trống'
            }), 400
        
        language = detect_language(email_text)
        
        return jsonify({
            'language': language
        })
        
    except Exception as e:
        print(f"Lỗi trong /api/detect-language: {e}")
        traceback.print_exc()
        return jsonify({
            'error': f'Lỗi khi phát hiện ngôn ngữ: {str(e)}'
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint để dự đoán email với models dựa trên ngôn ngữ
    
    Request body:
    {
        "email_text": "subject + body của email"
    }
    
    Response:
    {
        "language": "vi" | "en",
        "predictions": {
            "GRU": {"label": "phishing", "probability": 0.95},
            "CNN": {"label": "benign", "probability": 0.87},
            ...
        }
    }
    
    Logic:
    - Nếu email là tiếng Việt: chỉ dùng BERT_Vietnamese
    - Nếu email là tiếng Anh: dùng BERT, BiLSTM, CNN, GRU, Hybrid_CNN_BiLSTM
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
        
        # Phát hiện ngôn ngữ
        language = detect_language(email_text)
        print(f"🌐 Phát hiện ngôn ngữ: {language}")
        
        # Chọn models dựa trên ngôn ngữ
        if language == 'vi':
            # Tiếng Việt: chỉ dùng BERT_Vietnamese
            selected_models = ['BERT_Vietnamese']
            print(f"📌 Chọn models cho tiếng Việt: {selected_models}")
        else:
            # Tiếng Anh: dùng 5 models
            selected_models = ['BERT', 'BiLSTM', 'CNN', 'GRU', 'Hybrid_CNN_BiLSTM']
            print(f"📌 Chọn models cho tiếng Anh: {selected_models}")
        
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
            'language': language,
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
        "model_name": "GRU" | "CNN" | "BiLSTM" | "Hybrid_CNN_BiLSTM" | "BERT" | "BERT_Vietnamese",
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
        elif model_name == "BERT_Vietnamese":
            # BERT_Vietnamese sử dụng SHAP để giải thích
            if shap_explainer_vietnamese is None:
                return jsonify({
                    'error': 'SHAP explainer chưa được khởi tạo cho BERT_Vietnamese'
                }), 503
            
            try:
                if mode == 'full':
                    result = shap_explainer_vietnamese.explain_with_shap(
                        email_text,
                        max_features=15
                    )
                else:
                    result = shap_explainer_vietnamese.explain_with_shap_fast(
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
                print(f"Lỗi khi tạo SHAP explanation cho BERT_Vietnamese: {e}")
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

