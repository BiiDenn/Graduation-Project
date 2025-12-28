# Bảng Tổng Hợp Tham Số Huấn Luyện Các Model

## 1. Model BERT

| Thành phần | Cấu hình |
|------------|----------|
| **Tokenizer** | BertTokenizerFast |
| **max_length** | 512 |
| **Model** | bert-base-uncased (HuggingFace) |
| **Embedding** | Pre-trained BERT embeddings (768 dimensions) |
| **Kiến trúc** | BertForSequenceClassification với 2 labels (binary classification) |
| **Loss** | CrossEntropyLoss (mặc định) |
| **Optimizer** | AdamW |
| **Learning Rate** | 2e-5 |
| **Metrics** | Accuracy, F1-score |
| **Epochs (max)** | 15 |
| **Batch size** | 8 (per_device_train_batch_size) |
| **Early Stopping** | monitor='f1', patience=3 |
| **Callbacks** | EarlyStoppingCallback, TensorBoard, ModelCheckpoint |
| **Epoch thực tế** | 6 (từ metrics) |
| **Test Accuracy** | 99.01% |
| **Test F1-Score** | 99.01% |
| **Test AUC-ROC** | 99.92% |

---

## 2. Model BiLSTM

| Thành phần | Cấu hình |
|------------|----------|
| **TextVectorization** | max_tokens=20000, output_sequence_length=512 |
| **Embedding** | input_dim=20000, output_dim=256 |
| **Kiến trúc** | BiLSTM(LSTM(128)) → Temporal Pooling → Dense(128, ReLU) → Dense(64, ReLU) → Dense(1, Sigmoid) |
| **Loss** | Binary Crossentropy |
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 |
| **Metrics** | Accuracy |
| **Epochs (max)** | 30 |
| **Batch size** | 32 (mặc định) |
| **Early Stopping** | monitor='val_loss', patience=3 |
| **Callbacks** | TensorBoard, EarlyStopping, ModelCheckpoint |
| **Epoch thực tế** | 8 (từ metrics) |
| **Test Accuracy** | 98.87% |
| **Test F1-Score** | 98.87% |
| **Test AUC-ROC** | 99.95% |

---

## 3. Model CNN

| Thành phần | Cấu hình |
|------------|----------|
| **TextVectorization** | max_tokens=20000, output_sequence_length=512 |
| **Embedding** | input_dim=20000, output_dim=256 |
| **Kiến trúc** | Conv1D(128, kernel=5) → BatchNorm → Dropout(0.3) → Conv1D(256, kernel=5) → BatchNorm → Dropout(0.3) → Conv1D(256, kernel=3) → BatchNorm → GlobalMaxPool1D → Dense(128, ReLU) → Dropout(0.2) → Dense(64, ReLU) → Dropout(0.2) → Dense(1, Sigmoid) |
| **Loss** | Binary Crossentropy |
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 |
| **Metrics** | Accuracy |
| **Epochs (max)** | 30 |
| **Batch size** | 32 (mặc định) |
| **Early Stopping** | monitor='val_loss', patience=3 |
| **Callbacks** | TensorBoard, EarlyStopping, ModelCheckpoint |
| **Epoch thực tế** | 19 (từ metrics) |
| **Test Accuracy** | 98.74% |
| **Test F1-Score** | 98.73% |
| **Test AUC-ROC** | 99.94% |

---

## 4. Model GRU

| Thành phần | Cấu hình |
|------------|----------|
| **TextVectorization** | max_tokens=20000, output_sequence_length=512 |
| **Embedding** | input_dim=20000, output_dim=256 |
| **Kiến trúc** | GRU(128, return_sequences=True) → BatchNorm → Dropout(0.4) → GRU(128, return_sequences=False) → BatchNorm → Dropout(0.4) → Dense(128, ReLU) → BatchNorm → Dropout(0.3) → Dense(64, ReLU) → Dense(1, Sigmoid) |
| **Loss** | Binary Crossentropy |
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 |
| **Metrics** | Accuracy |
| **Epochs (max)** | 30 |
| **Batch size** | 32 (mặc định) |
| **Early Stopping** | monitor='val_loss', patience=3 |
| **Callbacks** | TensorBoard, EarlyStopping, ModelCheckpoint |
| **Epoch thực tế** | 10 (từ metrics) |
| **Test Accuracy** | 98.56% |
| **Test F1-Score** | 98.56% |
| **Test AUC-ROC** | 99.76% |

---

## 5. Model Hybrid_CNN_BiLSTM

| Thành phần | Cấu hình |
|------------|----------|
| **TextVectorization** | max_tokens=20000, output_sequence_length=512 |
| **Embedding** | input_dim=20000, output_dim=256 |
| **Kiến trúc** | Conv1D(128, kernel=3) → Conv1D(128, kernel=5) → MaxPooling1D → Dropout(0.3) → BiLSTM(LSTM(128)) x2 → Dropout(0.3) → Dense(128, ReLU) → Dense(1, Sigmoid) |
| **Loss** | Binary Crossentropy |
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 |
| **Metrics** | Accuracy |
| **Epochs (max)** | 30 |
| **Batch size** | 32 (mặc định) |
| **Early Stopping** | monitor='val_loss', patience=3 |
| **Callbacks** | TensorBoard, EarlyStopping, ModelCheckpoint |
| **Epoch thực tế** | 4 (từ metrics) |
| **Test Accuracy** | 98.63% |
| **Test F1-Score** | 98.63% |
| **Test AUC-ROC** | 99.89% |

---

## So Sánh Tổng Quan

| Tham số | BERT | BiLSTM | CNN | GRU | Hybrid_CNN_BiLSTM |
|---------|------|-------|-----|-----|-------------------|
| **Framework** | PyTorch (HuggingFace) | TensorFlow/Keras | TensorFlow/Keras | TensorFlow/Keras | TensorFlow/Keras |
| **TextVectorization** | BertTokenizerFast | TextVectorization | TextVectorization | TextVectorization | TextVectorization |
| **max_tokens/max_length** | 512 | 20000 | 20000 | 20000 | 20000 |
| **output_sequence_length** | 512 | 512 | 512 | 512 | 512 |
| **Embedding dim** | 768 (pre-trained) | 256 | 256 | 256 | 256 |
| **Learning Rate** | 2e-5 | 0.001 | 0.001 | 0.001 | 0.001 |
| **Batch Size** | 8 | 32 | 32 | 32 | 32 |
| **Max Epochs** | 15 | 30 | 30 | 30 | 30 |
| **Epoch thực tế** | 6 | 8 | 19 | 10 | 4 |
| **Early Stopping Patience** | 3 | 3 | 3 | 3 | 3 |
| **Early Stopping Monitor** | f1 | val_loss | val_loss | val_loss | val_loss |
| **Test Accuracy** | 99.01% | 98.87% | 98.74% | 98.56% | 98.63% |
| **Test F1-Score** | 99.01% | 98.87% | 98.73% | 98.56% | 98.63% |
| **Test AUC-ROC** | 99.92% | 99.95% | 99.94% | 99.76% | 99.89% |

---

## Ghi Chú

1. **BERT**: 
   - Sử dụng pre-trained model từ HuggingFace, không cần TextVectorization riêng
   - Batch size nhỏ hơn (8) do model lớn hơn (~110M tham số)
   - Đạt kết quả tốt nhất (99.01% accuracy) với chỉ 6 epochs
   - Test AUC-ROC: 99.92%

2. **BiLSTM, CNN, GRU, Hybrid_CNN_BiLSTM**: 
   - Tất cả sử dụng cùng cấu hình TextVectorization (max_tokens=20000, sequence_length=512) và Embedding (256 dimensions)
   - Kiến trúc khác nhau ở các layer xử lý
   - BiLSTM đạt AUC-ROC cao nhất (99.95%)
   - Hybrid_CNN_BiLSTM hội tụ nhanh nhất (chỉ 4 epochs)

3. **Early Stopping**: 
   - Tất cả models đều sử dụng early stopping để tránh overfitting
   - Patience=3 cho tất cả models
   - BERT monitor 'f1', các model khác monitor 'val_loss'

4. **Epoch thực tế**: 
   - Số epochs thực tế nhỏ hơn max epochs do early stopping kích hoạt
   - Hybrid_CNN_BiLSTM: 4 epochs (nhanh nhất)
   - BERT: 6 epochs
   - BiLSTM: 8 epochs
   - GRU: 10 epochs
   - CNN: 19 epochs (lâu nhất)

5. **Kết quả Test**:
   - Tất cả models đạt accuracy >98.5%, chứng tỏ hiệu quả cao
   - BERT tốt nhất (99.01%), tiếp theo là BiLSTM (98.87%)
   - AUC-ROC của tất cả models đều >99.7%, chứng tỏ khả năng phân biệt xuất sắc

