# Tài Liệu Mô Hình BiLSTM (Bidirectional Long Short-Term Memory)

## 1. Khái Niệm Cơ Bản về BiLSTM

### 1.1. BiLSTM là gì?

**BiLSTM (Bidirectional Long Short-Term Memory)** là một mô hình mạng nơ-ron hồi quy hai chiều, kết hợp hai mạng LSTM chạy ngược chiều nhau trên cùng một chuỗi dữ liệu. BiLSTM được thiết kế để nắm bắt thông tin từ cả hai hướng: quá khứ (forward) và tương lai (backward), giúp mô hình hiểu được ngữ cảnh đầy đủ hơn so với LSTM một chiều.

### 1.2. Đặc điểm chính của BiLSTM

- **Hai chiều xử lý**: 
  - **Forward LSTM**: Đọc chuỗi từ đầu đến cuối (từ trái sang phải)
  - **Backward LSTM**: Đọc chuỗi từ cuối đến đầu (từ phải sang trái)
  - Kết quả của hai hướng được ghép lại (concatenate)

- **Nắm bắt ngữ cảnh đầy đủ**: 
  - Với mỗi từ trong chuỗi, BiLSTM có thông tin về cả phần trước và phần sau
  - Giúp hiểu rõ hơn ý nghĩa của từ trong ngữ cảnh

- **Giải quyết vấn đề phụ thuộc dài hạn**: 
  - Kế thừa khả năng của LSTM trong việc học các phụ thuộc dài hạn
  - Cải thiện khả năng hiểu ngữ nghĩa nhờ thông tin hai chiều

### 1.3. Cấu trúc BiLSTM

BiLSTM bao gồm:

1. **Forward LSTM**: 
   - Đọc chuỗi từ x_1 → x_2 → ... → x_T
   - Tạo ra hidden states: h_1^f, h_2^f, ..., h_T^f

2. **Backward LSTM**: 
   - Đọc chuỗi từ x_T → x_{T-1} → ... → x_1
   - Tạo ra hidden states: h_T^b, h_{T-1}^b, ..., h_1^b

3. **Concatenation**: 
   - Với mỗi vị trí t: h_t = [h_t^f; h_t^b]
   - Kích thước hidden state tăng gấp đôi so với LSTM một chiều

**Công thức toán học**:

```
Forward LSTM:
h_t^f = LSTM_forward(x_t, h_{t-1}^f, c_{t-1}^f)

Backward LSTM:
h_t^b = LSTM_backward(x_t, h_{t+1}^b, c_{t+1}^b)

BiLSTM output:
h_t = [h_t^f; h_t^b]  (concatenation)
```

### 1.4. So sánh với LSTM một chiều

| Đặc điểm | LSTM một chiều | BiLSTM |
|---------|----------------|--------|
| Hướng xử lý | Chỉ từ trái sang phải | Cả hai chiều |
| Thông tin ngữ cảnh | Chỉ biết phần trước | Biết cả trước và sau |
| Kích thước hidden state | d | 2d (gấp đôi) |
| Số tham số | N | ~2N |
| Độ chính xác | Tốt | Tốt hơn (thường) |
| Tốc độ | Nhanh | Chậm hơn (gấp đôi tính toán) |

---

## 2. Ứng Dụng của BiLSTM trong Project Phát Hiện Email Phishing

### 2.1. Bài toán

Trong project này, BiLSTM được sử dụng để giải quyết bài toán **phân loại nhị phân email**:
- **Nhãn 0**: Benign (email bình thường, hợp pháp)
- **Nhãn 1**: Phishing (email lừa đảo, giả mạo)

### 2.2. Tại sao chọn BiLSTM?

1. **Hiểu ngữ cảnh đầy đủ**: 
   - Email có thể chứa các câu phủ định hoặc điều kiện
   - Ví dụ: "Do NOT click suspicious links" vs "Click here to verify"
   - BiLSTM có thể phân biệt nhờ thông tin hai chiều

2. **Phát hiện pattern phức tạp**: 
   - Email phishing thường có cấu trúc: lời chào → lý do → yêu cầu → link
   - BiLSTM nắm bắt được cả cấu trúc tổng thể và chi tiết cục bộ

3. **Xử lý phụ thuộc dài hạn**: 
   - Email có thể dài, thông tin quan trọng có thể ở đầu hoặc cuối
   - BiLSTM đảm bảo không bỏ sót thông tin quan trọng

4. **Hiệu quả cao**: 
   - Trong thực tế, BiLSTM thường đạt kết quả tốt hơn LSTM một chiều
   - Đặc biệt phù hợp với bài toán phân loại văn bản

### 2.3. Dữ liệu sử dụng

- **Tập train**: `final_train.csv` với ~13,648 email (sau khi xử lý mất cân bằng)
- **Tập validation**: `final_val.csv` với ~2,925 email (hoặc chia 30% từ train)
- **Tập test**: `final_test.csv` với ~2,925 email
- **Chia dữ liệu**: 70% train, 15% validation, 15% test (stratified split)
- **Lưu ý**: Trong quá trình training, có thể chia tiếp từ `final_train.csv` thành train/val với `stratify=y` để đảm bảo phân phối lớp đồng đều

---

## 3. Kiến Trúc Mô Hình BiLSTM

### 3.1. Tổng quan kiến trúc

```
Input (String) 
    ↓
TextVectorization (max_tokens=20000, sequence_length=512)
    ↓
Embedding (20000 → 128 dimensions)
    ↓
Bidirectional LSTM Layer (64 units forward + 64 units backward = 128 total)
    ↓
Dense Layer (32 units, ReLU activation)
    ↓
Dense Layer (1 unit, Sigmoid activation)
    ↓
Output (Probability: 0 = Benign, 1 = Phishing)
```

### 3.2. Chi tiết từng lớp

#### 3.2.1. Input Layer và Tiền Xử Lý

**Cách Model Nhận Input**:

1. **Input ban đầu**: Email text dạng string
   ```python
   email_text = "Your account has been suspended. Please verify..."
   ```

2. **TextVectorization** (tích hợp trong model):
   ```python
   text_vectorizer = layers.TextVectorization(
       max_tokens=20000,        # Vocabulary size
       output_mode='int',       # Token IDs
       output_sequence_length=512,
   )
   text_vectorizer.adapt(X_train.astype(str))
   ```
   - Email text → Token IDs `[id_1, id_2, ..., id_512]`
   - Từ không có trong vocab → `[UNK]`
   - Email ngắn → Padding, email dài → Truncate

3. **Input Layer**:
   ```python
   inputs = layers.Input(shape=(1,), dtype="string")
   ```
   - Nhận đầu vào là chuỗi text gốc (string)
   - TextVectorization layer sẽ tự động xử lý trong model

#### 3.2.2. TextVectorization Layer
```python
text_vectorizer = layers.TextVectorization(
    max_tokens=20000,
    output_mode='int',
    output_sequence_length=512,
    name='text_vectorization'
)
text_vectorizer.adapt(X_train.astype(str))
```
- Chuyển đổi text thành chuỗi token IDs
- Mỗi email → vector `[id_1, id_2, ..., id_512]`

#### 3.2.3. Embedding Layer
```python
bidirectonal_embedding = layers.Embedding(
    input_dim=20000,
    output_dim=256,
    input_length=512,
    name="embedding_4"
)
```
- Chuyển mỗi token ID thành vector 256 chiều
- Kết quả: Tensor `(batch_size, 512, 256)`

#### 3.2.4. Bidirectional LSTM Layer
```python
x = layers.Bidirectional(
    layers.LSTM(128, return_sequences=True),
    name="bilstm"
)(x)
```

**Cơ chế hoạt động**:

1. **Forward LSTM (128 units)**:
   - Đọc từ token 1 → 512
   - Tạo ra forward hidden states: h_1^f, h_2^f, ..., h_512^f
   - Mỗi h_t^f có kích thước 128

2. **Backward LSTM (128 units)**:
   - Đọc từ token 512 → 1
   - Tạo ra backward hidden states: h_512^b, h_511^b, ..., h_1^b
   - Mỗi h_t^b có kích thước 128

3. **Concatenation**:
   - Với mỗi vị trí t: h_t = [h_t^f; h_t^b] (256 chiều)
   - Với `return_sequences=True`: Trả về hidden states cho tất cả vị trí
   - Kết quả: Tensor `(batch_size, 512, 256)` (256 = 128 forward + 128 backward)

4. **Temporal Pooling**:
   - GlobalMaxPooling1D + GlobalAveragePooling1D (concatenated)
   - Kết quả: Vector 512 chiều (256 từ max pooling + 256 từ average pooling)

**Ví dụ minh họa**:

```
Email: "Do not click suspicious links"

Forward LSTM đọc: "Do" → "not" → "click" → "suspicious" → "links"
  h_1^f: thông tin về "Do"
  h_2^f: thông tin về "Do not"
  h_3^f: thông tin về "Do not click"
  ...

Backward LSTM đọc: "links" → "suspicious" → "click" → "not" → "Do"
  h_5^b: thông tin về "links"
  h_4^b: thông tin về "suspicious links"
  h_3^b: thông tin về "click suspicious links"
  ...

Khi xử lý từ "click":
  - Forward biết: "Do not" (trước đó)
  - Backward biết: "suspicious links" (sau đó)
  → h_3 = [h_3^f; h_3^b] có thông tin đầy đủ về ngữ cảnh
```

#### 3.2.5. Dense Layers
```python
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
```
- **Dense(128, ReLU)**: Kết hợp các đặc trưng từ BiLSTM
- **Dense(64, ReLU)**: Tiếp tục kết hợp đặc trưng
- **Dense(1, Sigmoid)**: Trả về xác suất email là phishing

### 3.3. Tổng số tham số

Mô hình BiLSTM có khoảng **10 triệu tham số**, chủ yếu từ:
- Embedding: 20000 × 256 = 5,120,000
- BiLSTM: ~800,000 (gấp đôi LSTM một chiều)
- Dense layers: ~8,000

---

## 4. Trích Xuất Đặc Trưng (Feature Extraction)

### 4.1. Quá trình trích xuất đặc trưng của BiLSTM

BiLSTM trích xuất đặc trưng qua các bước sau:

1. **Embedding Layer**: Chuyển token IDs thành vector embedding
   - Input: `[id_1, id_2, ..., id_512]` (token IDs)
   - Output: `(batch_size, 512, 128)` (mỗi token → vector 128 chiều)
   - **Ý nghĩa**: Tạo không gian đặc trưng ngữ nghĩa

2. **Bidirectional LSTM**: Nắm bắt ngữ cảnh hai chiều
   - **Forward LSTM**: Đọc từ trái sang phải, duy trì hidden state
     - Với mỗi từ, cập nhật hidden state dựa trên từ hiện tại và hidden state trước
     - Hidden state chứa thông tin về tất cả từ đã đọc trước đó
   - **Backward LSTM**: Đọc từ phải sang trái
     - Cung cấp thông tin về các từ "sau" từ hiện tại
   - **Concatenation**: Ghép hidden states từ hai chiều
     - Output: `(batch_size, 128)` (64 từ forward + 64 từ backward)
   - **Ý nghĩa**: Mỗi từ được hiểu trong ngữ cảnh đầy đủ (trước + sau)

3. **Dense Layers**: Kết hợp đặc trưng để phân loại
   - Dense(32) → Kết hợp đặc trưng từ BiLSTM
   - Dense(1, sigmoid) → Xác suất phishing

### 4.2. Ví dụ minh họa

**Email**: "Do NOT click suspicious links. This is a security warning."

**Quá trình**:
1. TextVectorization → Token IDs
2. Embedding → Vectors (512 × 128)
3. BiLSTM xử lý:
   - Forward đọc: "Do" → "NOT" → "click" → ...
     - Tại "click": Forward biết "Do NOT" (phủ định)
   - Backward đọc: "warning" → "security" → "links" → "suspicious" → "click" → ...
     - Tại "click": Backward biết "suspicious links" (nguy hiểm)
   - Hidden state cuối: Vector 128 chiều chứa thông tin "đây là cảnh báo, không phải phishing"
4. Dense → Xác suất phishing = 0.15 (thấp, đúng)

### 4.3. Đặc điểm của đặc trưng BiLSTM

- **Ngữ cảnh đầy đủ**: Biết cả phần trước và sau của mỗi từ
- **Phụ thuộc dài hạn**: Có thể nắm bắt mối quan hệ giữa các từ xa nhau
- **Hiểu cấu trúc**: Phân biệt được "Do NOT click" vs "Click here"
- **Hạn chế**: Chậm hơn CNN (phải đọc tuần tự), nhiều tham số hơn

---

## 5. Quy Trình Hoạt Động của BiLSTM để Nhận Diện Email

### 4.1. Quy trình tổng thể

Khi một email mới được đưa vào mô hình BiLSTM, quy trình xử lý diễn ra như sau:

#### Bước 1: Tiền xử lý Text
```
Email gốc: "Your account has been compromised. Please verify immediately."
    ↓
TextVectorization
    ↓
Token IDs: [1234, 567, 890, 234, 456, 789, 345, 678, ...] (512 số)
```

#### Bước 2: Embedding
```
Token IDs (512 số)
    ↓
Embedding Layer
    ↓
Embedding Vectors: (512, 128) - mỗi từ là vector 128 chiều
```

#### Bước 3: BiLSTM xử lý hai chiều

**Forward LSTM (từ trái sang phải)**:
```
Từ 1: "Your" → Forward LSTM tính h_1^f
Từ 2: "account" → Forward LSTM tính h_2^f (dựa trên h_1^f và "account")
Từ 3: "has" → Forward LSTM tính h_3^f (dựa trên h_2^f và "has")
...
Từ 512: → Forward LSTM tính h_512^f
```

**Backward LSTM (từ phải sang trái)**:
```
Từ 512: (padding hoặc từ cuối) → Backward LSTM tính h_512^b
Từ 511: → Backward LSTM tính h_511^b (dựa trên h_512^b và từ 511)
...
Từ 1: "Your" → Backward LSTM tính h_1^b (dựa trên h_2^b và "Your")
```

**Ghép kết quả**:
```
Với mỗi vị trí t:
  h_t = [h_t^f; h_t^b]  (concatenation, 128 chiều)

Với return_sequences=False:
  Chỉ lấy h_512 (hoặc h_T) - hidden state cuối cùng
  → Vector 128 chiều
```

**Ví dụ cụ thể với email phishing**:

```
Email: "Your account has been compromised. Please verify immediately."

Forward LSTM đọc:
  "Your" → h_1^f: [0.1, 0.2, ...] (64 chiều)
  "account" → h_2^f: [0.15, 0.25, ...] (tích lũy "Your account")
  "has" → h_3^f: [0.2, 0.3, ...] (tích lũy "Your account has")
  "been" → h_4^f: [0.25, 0.35, ...]
  "compromised" → h_5^f: [0.5, 0.6, ...] (dấu hiệu nguy hiểm tăng mạnh)
  "Please" → h_6^f: [0.55, 0.65, ...]
  "verify" → h_7^f: [0.7, 0.8, ...] (dấu hiệu phishing rõ ràng)
  "immediately" → h_8^f: [0.75, 0.85, ...] (tăng mức độ gấp gáp)
  ...

Backward LSTM đọc:
  "immediately" → h_8^b: [0.1, 0.15, ...] (64 chiều)
  "verify" → h_7^b: [0.2, 0.25, ...] (tích lũy "verify immediately")
  "Please" → h_6^b: [0.3, 0.35, ...] (tích lũy "Please verify immediately")
  "compromised" → h_5^b: [0.6, 0.7, ...] (biết sẽ có "verify" sau đó)
  "been" → h_4^b: [0.65, 0.75, ...]
  "has" → h_3^b: [0.7, 0.8, ...]
  "account" → h_2^b: [0.75, 0.85, ...] (biết sẽ có "compromised" và "verify")
  "Your" → h_1^b: [0.8, 0.9, ...] (có thông tin đầy đủ về toàn bộ email)
  ...

Ghép lại:
  h_8 = [h_8^f; h_8^b] = [[0.75, 0.85, ...]; [0.1, 0.15, ...]] (128 chiều)
  → Hidden state cuối có thông tin đầy đủ từ cả hai chiều
```

#### Bước 4: Trích xuất đặc trưng cuối cùng
```
h_512 (hidden state cuối, 128 chiều)
    ↓
Dense(32, ReLU) → kết hợp đặc trưng
    ↓
Dense(1, Sigmoid) → xác suất phishing
```

#### Bước 5: Quyết định phân loại
```
Xác suất = 0.92 (> 0.5)
    ↓
Kết luận: Phishing (Nhãn 1)
```

### 4.2. Ví dụ so sánh với LSTM một chiều

**Email: "This is NOT a phishing email. Do not click any links."**

**LSTM một chiều**:
- Khi đọc đến "click", chỉ biết phần trước: "This is NOT a phishing email. Do not"
- Có thể hiểu nhầm là yêu cầu click (vì chưa thấy "any links" phía sau)

**BiLSTM**:
- Forward LSTM: Khi đọc "click", biết "Do not" trước đó → hiểu là phủ định
- Backward LSTM: Khi đọc "click", biết "any links" sau đó → hiểu rõ hơn ngữ cảnh
- Kết hợp: Hiểu đúng là "Do not click any links" (cảnh báo, không phải phishing)

### 4.3. Đặc điểm BiLSTM học được

1. **Pattern từ vựng với ngữ cảnh**:
   - "Click here" trong email cảnh báo vs trong email phishing
   - "Verify your account" với ngữ cảnh phủ định vs khẳng định

2. **Cấu trúc email**:
   - Phishing: Lời chào → Lý do giả mạo → Yêu cầu → Link
   - Benign: Lời chào → Thông tin → Yêu cầu hợp lý

3. **Tông giọng và ngữ cảnh**:
   - Phishing: Gấp gáp, đe dọa, hứa hẹn
   - Benign: Thân thiện, chuyên nghiệp, thông tin

---

## 5. Quy Trình Xây Dựng Mô Hình BiLSTM

### 5.1. Chuẩn bị dữ liệu

#### Bước 1: Load dữ liệu
```python
df_train = pd.read_csv('data/final/final_train.csv')
X = df_train['text']
y = df_train['label']
```

#### Bước 2: Phân tích độ dài email
```python
text_length = [len(str(text).split(' ')) for text in df_train['text']]
print(f'Email dài nhất: {max(text_length)} từ')
print(f'95% email < {np.percentile(text_length, 95)} từ')
```
**Kết quả**: Email dài nhất 568 từ, 95% < 467 từ → chọn `sequence_length=512`

#### Bước 3: Chia train/validation

**Lưu ý về chia dữ liệu**:
- Dataset đã được chia sẵn thành `final_train.csv`, `final_val.csv`, `final_test.csv`
- Trong training, có thể sử dụng trực tiếp hoặc chia tiếp từ `final_train.csv`

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.3,      # 30% validation từ train
    stratify=y,         # Giữ tỷ lệ phishing/benign đồng đều
    random_state=42     # Đảm bảo tái tạo kết quả
)
```

### 5.2. Xây dựng TextVectorization

```python
text_vectorizer = layers.TextVectorization(
    max_tokens=20000,
    output_mode='int',
    output_sequence_length=512,
    name='text_vectorization'
)
text_vectorizer.adapt(X_train.astype(str))
```

### 5.3. Xây dựng Embedding Layer

```python
bidirectonal_embedding = layers.Embedding(
    input_dim=20000,
    output_dim=128,
    input_length=512,
    name="embedding_4"
)
```

### 5.4. Xây dựng mô hình BiLSTM

```python
# Build model
inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = bidirectonal_embedding(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
x = layers.Dense(32, activation='relu')(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

bidirectonal_model = tf.keras.Model(inputs, outputs, name="model_4_Bidirectional")
```

**Lý do chọn tham số**:
- `Bidirectional(LSTM(64))`: 64 units mỗi chiều → 128 units tổng
- `Dense(32)`: Giảm chiều để tránh overfitting

### 5.5. Compile mô hình

```python
bidirectonal_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### 5.6. Chuẩn bị callbacks

```python
# Model Checkpoint
bidirectonal_model_checkpoint_callback = create_checkpoint_callback(
    file_path='outputs/models/BiLSTM/checkpoints/bidirectonal_model.keras'
)

# TensorBoard
bidirectonal_model_tensorboard = create_tensorboard_callback(
    file_path='outputs/models/BiLSTM/logs/bidirectonal_model_tensorboard'
)

# Early Stopping
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',     # Theo dõi validation accuracy
    patience=10,                 # Dừng nếu không cải thiện sau 10 epochs
    verbose=1,                   # Hiển thị thông báo
    mode='max',                  # Tối đa hóa accuracy (tăng)
    restore_best_weights=True    # Khôi phục weights tốt nhất
)
```

**Giải thích Early Stopping**:
- **Monitor `val_accuracy`**: Theo dõi độ chính xác trên validation set
- **Patience=10**: Nếu val_accuracy không tăng trong 10 epochs → dừng
- **Mode='max'**: Tối đa hóa metric (accuracy càng cao càng tốt)
- **Lý do**: Tránh overfitting, dừng khi model không còn học được gì mới
```

### 5.7. Huấn luyện mô hình

```python
# Convert to tensors
X_train_tensor = tf.convert_to_tensor(
    X_train.fillna('').astype(str), 
    dtype=tf.string
)
X_val_tensor = tf.convert_to_tensor(
    X_val.fillna('').astype(str), 
    dtype=tf.string
)

# Train
bidirectonal_model_history = bidirectonal_model.fit(
    X_train_tensor,
    y_train_tensor,
    validation_data=(X_val_tensor, y_val_tensor),
    epochs=30,
    callbacks=[
        bidirectonal_model_checkpoint_callback,
        bidirectonal_model_tensorboard,
        early_stopping_callback
    ]
)
```

**Quá trình huấn luyện** (thực tế: 8 epochs):
- Epoch 1: Accuracy ~0.90, Loss ~0.25
- Epoch 2: Accuracy ~0.98, Loss ~0.05
- Epoch 3: Accuracy ~0.99, Loss ~0.02
- ...
- Epoch 8: Accuracy ~0.9988, Loss ~0.003 (Early stopping kích hoạt)

### 5.8. Đánh giá mô hình

```python
# Predict on validation set
pred_labels_bilstm = bidirectonal_model.predict(X_val_tensor)
avg_pred_labels_bilstm = np.round(pred_labels_bilstm.flatten())

# Calculate metrics
bilstm_model_results = calculate_results(y_val.values, avg_pred_labels_bilstm)
```

### 5.9. Lưu mô hình và metrics

```python
# Save model
bidirectonal_model.save('outputs/models/BiLSTM/BiLSTM.h5')

# Save metrics
bidirectional_metrics = {
    'model_config': {...},
    'validation': {...},
    'other_validation_metrics': {...}
}
with open('outputs/models/BiLSTM/BiLSTM_metrics.json', 'w') as f:
    json.dump(bidirectional_metrics, f, indent=4)
```

### 5.10. Lưu embedding để visualization

```python
# Get embedding weights
embedding_weights = bidirectonal_model.get_layer('embedding_4').get_weights()[0]

# Save for TensorFlow Projector
out_v = io.open('outputs/models/BiLSTM/bilstm_embedding_vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('outputs/models/BiLSTM/bilstm_embedding_metadata.tsv', 'w', encoding='utf-8')

for num, word in enumerate(words_in_vocab):
    if num == 0:
        continue  # skip padding token
    vec = embedding_weights[num]
    out_m.write(word + "\n")
    out_v.write("\t".join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()
```

---

## 6. Kết Quả của Mô Hình BiLSTM

### 6.1. Kết quả trên tập test

Sau khi huấn luyện và đánh giá trên tập test (2,925 email), mô hình BiLSTM đạt được các chỉ số sau:

| Metric | Giá trị |
|--------|---------|
| **Accuracy** | 0.9887 (98.87%) |
| **Precision** | 0.9887 (98.87%) |
| **Recall** | 0.9887 (98.87%) |
| **F1-Score** | 0.9887 (98.87%) |
| **AUC-ROC** | 0.9995 (99.95%) |

### 6.2. Kết quả trên tập validation

| Metric | Giá trị |
|--------|---------|
| **Accuracy** | 0.988 (98.8%) |
| **Precision** | 0.988 (98.8%) |
| **Recall** | 0.988 (98.8%) |
| **F1-Score** | 0.988 (98.8%) |
| **AUC-ROC** | 0.9994 (99.94%) |

### 6.3. Các chỉ số khác (Test)

| Metric | Giá trị |
|--------|---------|
| **False Discovery Rate (FDR)** | 0.0126 (1.26%) |
| **False Negative Rate (FNR)** | 0.0105 (1.05%) |
| **False Omission Rate (FOR)** | 0.01 (1.0%) |
| **False Positive Rate (FPR)** | 0.012 (1.2%) |
| **Negative Prediction Value (NPV)** | 0.99 (99.0%) |

### 6.4. Phân tích kết quả

**Điểm mạnh**:
- **Accuracy rất cao (98.87% trên test)**: Mô hình phân loại cực kỳ chính xác
- **F1-Score rất cao (98.87%)**: Cân bằng hoàn hảo giữa Precision và Recall
- **FNR rất thấp (1.05%)**: Rất ít email phishing bị bỏ sót (quan trọng trong bảo mật)
- **FPR rất thấp (1.2%)**: Rất ít email benign bị nhận nhầm
- **NPV rất cao (99.0%)**: Khi mô hình dự đoán Benign, khả năng đúng rất cao
- **AUC-ROC gần hoàn hảo (99.95%)**: Khả năng phân biệt giữa phishing và benign rất xuất sắc

**So sánh với các mô hình khác**:

| Mô hình | Accuracy (Test) | F1-Score | FNR | AUC-ROC |
|---------|----------------|----------|-----|---------|
| BiLSTM | 98.87% | 98.87% | 1.05% | 99.95% |
| BERT | 99.01% | 99.01% | 0.98% | 99.92% |
| CNN-BiLSTM | 98.63% | 98.63% | 1.61% | 99.89% |
| CNN | 98.74% | 98.73% | 2.18% | 99.94% |
| GRU | 98.56% | 98.56% | 1.61% | 99.76% |

**Nhận xét**: BiLSTM đạt kết quả tốt thứ hai (sau BERT), nhờ khả năng nắm bắt ngữ cảnh hai chiều. AUC-ROC cao nhất (99.95%) chứng tỏ khả năng phân biệt xuất sắc.

### 6.5. Confusion Matrix (Test)

Dựa trên kết quả test với 2,925 email:
- **True Positive (TP)**: ~1,447 email phishing được phát hiện đúng
- **True Negative (TN)**: ~1,444 email benign được phân loại đúng
- **False Positive (FP)**: ~18 email benign bị nhận nhầm là phishing (FPR = 1.2%)
- **False Negative (FN)**: ~16 email phishing bị bỏ sót (FNR = 1.05%, rất ít!)

### 6.6. Ví dụ dự đoán

**Email được phát hiện đúng là Phishing**:
```
Input: "Your bank account has been compromised. Please verify your details immediately at this link: http://malicious-site.com/verify"
Prediction: Phishing (xác suất: 0.98) ✓
```

**Email được phát hiện đúng là Benign**:
```
Input: "Hi team, just a quick reminder about our meeting tomorrow at 10 AM. Please prepare your reports."
Prediction: Benign (xác suất: 0.12) ✓
```

**Email có ngữ cảnh phủ định (phát hiện đúng)**:
```
Input: "This is an example of a phishing attempt. Do NOT click any suspicious links."
Prediction: Benign (xác suất: 0.25) ✓
(LSTM một chiều có thể nhầm, nhưng BiLSTM hiểu đúng nhờ thông tin hai chiều)
```

---

## 7. Giải Thích Dự Đoán (Explainable AI - XAI)

### 7.1. Cách Model "Biết" Từ Nào Quan Trọng

BiLSTM sử dụng **LIME (Local Interpretable Model-Agnostic Explanations)** để giải thích dự đoán. Cơ chế tương tự như CNN (xem phần 7 của TAI_LIEU_CNN.md).

**Điểm khác biệt với CNN**:
- BiLSTM nắm bắt ngữ cảnh hai chiều tốt hơn
- LIME có thể phát hiện các từ quan trọng dựa trên ngữ cảnh đầy đủ
- Ví dụ: "Do NOT click" → từ "NOT" có trọng số âm mạnh (giảm xác suất phishing)

### 7.2. Ví Dụ Giải Thích LIME cho BiLSTM

**Email**: "Do NOT click suspicious links. This is a security warning from your bank."

**Kết quả LIME** (top 5 từ quan trọng):
```
1. "NOT"         → weight: -0.145  (giảm xác suất phishing mạnh)
2. "security"    → weight: -0.098  (tăng xác suất benign)
3. "warning"     → weight: -0.087  (tăng xác suất benign)
4. "suspicious"  → weight: +0.065  (tăng xác suất phishing)
5. "click"       → weight: +0.032  (tăng xác suất phishing)
```

**Giải thích**: 
- BiLSTM hiểu được ngữ cảnh "Do NOT click" → từ "NOT" có trọng số âm mạnh
- Từ "security warning" → trọng số âm (benign)
- Tổng hợp: Email này là cảnh báo hợp pháp, không phải phishing

---

## 8. Kết Luận

Mô hình BiLSTM đã được áp dụng thành công trong bài toán phát hiện email phishing với các đặc điểm:

1. **Hiệu quả rất cao**: Đạt accuracy 98.87% trên test set, tốt thứ hai sau BERT
2. **AUC-ROC cao nhất (99.95%)**: Khả năng phân biệt giữa phishing và benign xuất sắc nhất
3. **FNR rất thấp (1.05%)**: Rất ít email phishing bị bỏ sót - quan trọng trong bảo mật
4. **Nắm bắt ngữ cảnh đầy đủ**: Nhờ xử lý hai chiều, BiLSTM hiểu rõ hơn ý nghĩa của từ trong ngữ cảnh
5. **Phát hiện pattern phức tạp**: Có thể phân biệt email cảnh báo vs email phishing thật
6. **Cân bằng tốt**: FNR và FPR đều rất thấp, phù hợp với ứng dụng thực tế

BiLSTM là lựa chọn tốt cho bài toán phát hiện email phishing khi cần độ chính xác cao, AUC-ROC xuất sắc và có đủ tài nguyên tính toán. Đặc biệt phù hợp khi cần cân bằng giữa độ chính xác và tốc độ xử lý.

