# Kiến trúc hệ thống Email Phishing Detection Extension

## Tổng quan

Extension sử dụng kiến trúc **Client-Server** với 2 thành phần chính:

1. **Browser Extension** (Client): Đọc email, hiển thị UI, gọi API
2. **Flask Backend** (Server): Load models, dự đoán, tạo XAI explanations

## Kiến trúc chi tiết

```
┌─────────────────────────────────────────────────────────┐
│                    Browser Extension                     │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │  Content     │    │   Popup      │    │ Background│ │
│  │  Script      │───▶│   UI/JS      │◀───│  Worker   │ │
│  │  (content.js)│    │ (popup.html) │    │(background)│ │
│  └──────────────┘    └──────────────┘    └───────────┘ │
│         │                    │                           │
│         │                    │                           │
│         ▼                    ▼                           │
│  ┌──────────────────────────────────────┐                │
│  │      Chrome Storage (localStorage)    │                │
│  └──────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
                          │
                          │ HTTP Requests
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    Flask Backend                         │
│                  (http://localhost:5000)                 │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │         Model Loader (Preloaded on Start)        │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐    │   │
│  │  │ GRU  │ │ CNN  │ │BiLSTM│ │Hybrid│ │ BERT │    │   │
│  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘    │   │
│  └──────────────────────────────────────────────────┘   │
│                                                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │              XAI Explainers                       │   │
│  │  ┌──────────────┐      ┌──────────────┐          │   │
│  │  │ LIME         │      │ SHAP         │          │   │
│  │  │ (Keras)      │      │ (BERT)       │          │   │
│  │  └──────────────┘      └──────────────┘          │   │
│  └──────────────────────────────────────────────────┘   │
│                                                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │              API Endpoints                        │   │
│  │  • GET  /api/health                              │   │
│  │  • POST /api/predict                             │   │
│  │  • POST /api/explain                             │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Luồng hoạt động

### 1. Đọc Email (Content Script)

```
User mở email trong Gmail/Outlook
    ↓
Content Script (content.js) detect DOM changes
    ↓
Extract subject + body từ DOM
    ↓
Lưu vào Chrome Storage (localStorage)
    ↓
Gửi message đến Background Worker
```

### 2. Phân tích Email (Popup → Backend)

```
User click "Phân tích Email" trong popup
    ↓
Popup.js gọi API: POST /api/predict
    ↓
Backend nhận email_text
    ↓
Chạy prediction với 5 models song song:
    • GRU.predict()
    • CNN.predict()
    • BiLSTM.predict()
    • Hybrid_CNN_BiLSTM.predict()
    • BERT.predict()
    ↓
Trả về JSON với predictions của tất cả models
    ↓
Popup hiển thị kết quả:
    • Summary card (majority vote)
    • 5 model cards (label, confidence, progress bar)
```

### 3. XAI Explanation (Popup → Backend)

```
User click "Giải thích dự đoán" trên model card
    ↓
Popup.js gọi API: POST /api/explain
    ↓
Backend nhận model_name + email_text
    ↓
Nếu model == BERT:
    → SHAPExplainer.explain_with_shap_fast()
Nếu model != BERT:
    → LIMEExplainer.explain_with_lime()
    ↓
Trả về JSON với important_tokens
    ↓
Popup hiển thị modal với:
    • Prediction label + confidence
    • List important tokens với weights
```

## Các thành phần

### Browser Extension

#### 1. manifest.json
- Định nghĩa permissions, content scripts, popup
- Manifest V3 (Chrome/Edge)

#### 2. content.js
- Chạy trên Gmail/Outlook pages
- Monitor DOM changes để detect email mới
- Extract email content (subject + body)
- Lưu vào Chrome Storage

#### 3. popup.html/css/js
- UI chính của extension
- Hiển thị email preview
- Gọi API backend
- Hiển thị kết quả predictions
- Modal XAI explanations

#### 4. background.js
- Service worker (Manifest V3)
- Listen messages từ content script
- Có thể thêm logic khác nếu cần

### Flask Backend

#### 1. app.py
- Flask application
- Load models khi start (preload)
- 3 API endpoints:
  - `/api/health`: Health check
  - `/api/predict`: Predict với tất cả models
  - `/api/explain`: XAI explanation

#### 2. Model Loader
- Sử dụng `notebooks/XAI/model_loader.py`
- Load 5 models:
  - GRU, CNN, BiLSTM, Hybrid_CNN_BiLSTM: Keras (.h5/.keras)
  - BERT: PyTorch/HuggingFace

#### 3. XAI Explainers
- LIME: Cho Keras models (GRU, CNN, BiLSTM, Hybrid)
- SHAP: Cho BERT model

## Tối ưu hóa

### 1. Preload Models
- Models được load khi server start
- Không delay khi predict (models đã sẵn sàng)

### 2. Optimized XAI
- LIME: 3000 samples (cân bằng tốc độ/accuracy)
- SHAP: Fast mode (~5 giây cho BERT)

### 3. Efficient Content Script
- MutationObserver để detect changes
- Check interval 2 giây
- Chỉ update khi email thay đổi

### 4. Caching
- Email được cache trong Chrome Storage
- Tránh re-extract không cần thiết

## Security

1. **No User Data Access**: Extension chỉ đọc DOM, không truy cập tài khoản
2. **Local Backend**: Backend chạy localhost, không gửi data ra ngoài
3. **CORS**: Chỉ cho phép extension gọi API
4. **No Storage of Sensitive Data**: Email chỉ lưu tạm trong localStorage

## Mở rộng

### Thêm model mới:
1. Thêm vào `model_loader.load_all_models()`
2. Update UI nếu cần
3. Thêm vào API response

### Thêm email provider:
1. Thêm selectors trong `content.js`
2. Update `extractEmail()` function
3. Thêm vào `manifest.json` host_permissions

### Thêm XAI method:
1. Tạo explainer mới (ví dụ: Integrated Gradients)
2. Thêm vào `app.py` explain endpoint
3. Update UI để hiển thị

## Performance

- **Predict time**: ~1-3 giây (5 models song song)
- **LIME time**: ~10-20 giây (3000 samples)
- **SHAP time**: ~5 giây (BERT, fast mode)
- **Total analyze time**: ~2-5 giây (chỉ predict)
- **Total with XAI**: ~15-25 giây (predict + 1 explanation)

## Dependencies

### Extension:
- Chrome/Edge browser (Manifest V3)
- No external dependencies (vanilla JS)

### Backend:
- Flask, Flask-CORS
- TensorFlow (Keras models)
- PyTorch (BERT)
- Transformers (BERT tokenizer)
- LIME (XAI)
- SHAP (XAI)

