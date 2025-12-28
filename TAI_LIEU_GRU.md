# Tài Liệu Mô Hình GRU (Gated Recurrent Unit)

## 1. Khái Niệm Cơ Bản về GRU

### 1.1. GRU là gì?

**GRU (Gated Recurrent Unit)** là một loại mạng nơ-ron hồi quy (RNN) được thiết kế để xử lý dữ liệu tuần tự. GRU được đề xuất vào năm 2014 như một biến thể đơn giản hóa của LSTM (Long Short-Term Memory), với mục tiêu giảm độ phức tạp tính toán trong khi vẫn duy trì khả năng học các phụ thuộc dài hạn.

### 1.2. Đặc điểm chính của GRU

- **Cơ chế cổng (Gating Mechanism)**: GRU sử dụng các cổng để điều khiển luồng thông tin, giúp mô hình quyết định thông tin nào cần giữ lại và thông tin nào cần quên đi.
- **Giải quyết vấn đề Vanishing Gradient**: Nhờ cơ chế cổng, GRU có thể học các phụ thuộc dài hạn mà không bị mất gradient trong quá trình lan truyền ngược.
- **Đơn giản hơn LSTM**: GRU chỉ có 2 cổng (Update gate và Reset gate) so với 3 cổng của LSTM, làm cho nó nhanh hơn và dễ huấn luyện hơn.

### 1.3. Cấu trúc GRU

GRU bao gồm hai cổng chính:

1. **Update Gate (z_t)**: Quyết định bao nhiêu thông tin từ trạng thái ẩn trước đó (h_{t-1}) được giữ lại và bao nhiêu thông tin mới từ đầu vào hiện tại (x_t) được cập nhật.
2. **Reset Gate (r_t)**: Quyết định bao nhiêu thông tin từ trạng thái ẩn trước đó cần được "quên" khi tính toán trạng thái ẩn mới.

Công thức toán học:

```
r_t = σ(W_r · [h_{t-1}, x_t] + b_r)  # Reset gate
z_t = σ(W_z · [h_{t-1}, x_t] + b_z)  # Update gate
h̃_t = tanh(W_h · [r_t * h_{t-1}, x_t] + b_h)  # Candidate hidden state
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t  # Final hidden state
```

Trong đó:
- σ là hàm sigmoid
- * là phép nhân element-wise
- W_r, W_z, W_h là ma trận trọng số
- b_r, b_z, b_h là bias

---

## 2. Ứng Dụng của GRU trong Project Phát Hiện Email Phishing

### 2.1. Bài toán

Trong project này, GRU được sử dụng để giải quyết bài toán **phân loại nhị phân email**:
- **Nhãn 0**: Benign (email bình thường, hợp pháp)
- **Nhãn 1**: Phishing (email lừa đảo, giả mạo)

### 2.2. Tại sao chọn GRU?

1. **Xử lý dữ liệu tuần tự**: Email là chuỗi từ được sắp xếp theo thứ tự, GRU có thể nắm bắt được ngữ cảnh và mối quan hệ giữa các từ trong email.

2. **Học phụ thuộc dài hạn**: Email có thể dài và chứa nhiều câu, GRU có khả năng "nhớ" thông tin từ đầu email đến cuối email, giúp hiểu được cấu trúc và ngữ nghĩa tổng thể.

3. **Hiệu quả tính toán**: So với LSTM, GRU nhanh hơn và ít tham số hơn, phù hợp với dữ liệu email có độ dài vừa phải (512 từ).

4. **Phát hiện pattern phức tạp**: GRU có thể học các pattern như:
   - Cấu trúc email phishing: lời chào → lý do giả mạo → yêu cầu hành động → link nguy hiểm
   - Ngữ cảnh xung quanh từ khóa: "click here" trong email cảnh báo vs trong email phishing thật

### 2.3. Dữ liệu sử dụng

- **Tập train**: `final_train.csv` với ~13,648 email (sau khi xử lý mất cân bằng)
- **Tập validation**: `final_val.csv` với ~2,925 email (hoặc chia 30% từ train)
- **Tập test**: `final_test.csv` với ~2,925 email
- **Chia dữ liệu**: 70% train, 15% validation, 15% test (stratified split)

---

## 3. Kiến Trúc Mô Hình GRU

### 3.1. Tổng quan kiến trúc

```
Input (String) 
    ↓
TextVectorization (max_tokens=20000, sequence_length=512)
    ↓
Embedding (20000 → 128 dimensions)
    ↓
GRU Layer (64 units, return_sequences=False)
    ↓
Dense Layer (64 units, ReLU activation)
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
   email_text = "Your bank account has been compromised. Click here to verify..."
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
```
- **Chức năng**: Chuyển đổi text thành chuỗi số (token IDs)
- **max_tokens=20000**: Chỉ giữ 20,000 từ phổ biến nhất, từ ít gặp → `[UNK]`
- **output_sequence_length=512**: Cắt hoặc pad mỗi email thành đúng 512 token
- **Kết quả**: Vector `[id_1, id_2, ..., id_512]`

#### 3.2.3. Embedding Layer
```python
gru_model_embedding = layers.Embedding(
    input_dim=20000,
    output_dim=256,
    embeddings_initializer="uniform",
    input_length=512,
    name="embedding_3"
)
```
- **Chức năng**: Chuyển mỗi token ID thành vector 256 chiều
- **Học ngữ nghĩa**: Từ có nghĩa tương tự → vector gần nhau
- **Kết quả**: Tensor `(batch_size, 512, 256)`

#### 3.2.4. GRU Layers (2 layers)
```python
# GRU Layer 1
x = layers.GRU(128, return_sequences=True, name="gru_layer_1")(x)
x = layers.BatchNormalization(name="bn_gru_1")(x)
x = layers.Dropout(0.4, name="dropout_gru_1")(x)

# GRU Layer 2
x = layers.GRU(128, return_sequences=False, name="gru_layer_2")(x)
x = layers.BatchNormalization(name="bn_gru_2")(x)
x = layers.Dropout(0.4, name="dropout_gru_2")(x)
```
- **GRU Layer 1 (128 units, return_sequences=True)**: 
  - Đọc email từ trái sang phải, từng từ một
  - Trả về hidden states cho tất cả vị trí
  - BatchNormalization + Dropout(0.4) để ổn định và tránh overfitting

- **GRU Layer 2 (128 units, return_sequences=False)**: 
  - Xử lý output từ layer 1
  - Chỉ trả về hidden state cuối cùng (sau khi đọc hết 512 từ)
  - Hidden state cuối cùng là tóm tắt toàn bộ email
  - BatchNormalization + Dropout(0.4)

- **Kết quả**: Vector 128 chiều

#### 3.2.5. Dense Layers
```python
x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01), name="dense_hidden_1")(x)
x = layers.BatchNormalization(name="bn_dense")(x)
x = layers.Dropout(0.3, name="dropout_dense")(x)
x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01), name="dense_hidden_2")(x)
outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
```
- **Dense(128, ReLU)**: Kết hợp các đặc trưng từ GRU thành biểu diễn phân loại
  - L2 regularization (0.01) để tránh overfitting
  - BatchNormalization + Dropout(0.3) để ổn định
- **Dense(64, ReLU)**: Tiếp tục kết hợp đặc trưng
  - L2 regularization (0.01)
- **Dense(1, Sigmoid)**: Trả về xác suất email là phishing (0-1)

### 3.3. Tổng số tham số

Mô hình GRU có khoảng **5.1 triệu tham số**, chủ yếu từ:
- Embedding: 20000 × 256 = 5,120,000
- GRU: ~400,000
- Dense layers: ~8,000

---

## 4. Trích Xuất Đặc Trưng (Feature Extraction)

### 4.1. Quá trình trích xuất đặc trưng của GRU

GRU trích xuất đặc trưng qua các bước sau:

1. **Embedding Layer**: Chuyển token IDs thành vector embedding
   - Input: `[id_1, id_2, ..., id_512]` (token IDs)
   - Output: `(batch_size, 512, 128)` (mỗi token → vector 128 chiều)

2. **GRU Layer**: Đọc tuần tự và duy trì hidden state
   - Đọc email từ trái sang phải, từng từ một
   - Với mỗi từ, cập nhật hidden state dựa trên:
     - Từ hiện tại (x_t)
     - Hidden state trước (h_{t-1})
     - Reset gate: Quyết định "quên" bao nhiêu thông tin cũ
     - Update gate: Quyết định "cập nhật" bao nhiêu thông tin mới
   - Hidden state cuối cùng (h_512) là tóm tắt toàn bộ email
   - Output: `(batch_size, 64)` (vector 64 chiều)
   - **Ý nghĩa**: Hidden state chứa ngữ cảnh và thông tin quan trọng của toàn bộ email

3. **Dense Layers**: Kết hợp đặc trưng để phân loại
   - Dense(64) → Kết hợp đặc trưng từ GRU
   - Dense(1, sigmoid) → Xác suất phishing

### 4.2. Đặc điểm của đặc trưng GRU

- **Ngữ cảnh tuần tự**: Nắm bắt thứ tự và mối quan hệ giữa các từ
- **Phụ thuộc dài hạn**: Có thể "nhớ" thông tin từ đầu email đến cuối
- **Hiệu quả**: Nhanh hơn LSTM, ít tham số hơn
- **Hạn chế**: Chỉ đọc một chiều (không như BiLSTM)

---

## 5. Quy Trình Hoạt Động của GRU để Nhận Diện Email

### 4.1. Quy trình tổng thể

Khi một email mới được đưa vào mô hình GRU, quy trình xử lý diễn ra như sau:

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

#### Bước 3: GRU xử lý tuần tự
```
Từ 1: "Your" → GRU tính h_1 (hidden state 1)
Từ 2: "account" → GRU tính h_2 (dựa trên h_1 và "account")
Từ 3: "has" → GRU tính h_3 (dựa trên h_2 và "has")
...
Từ 512: (từ cuối hoặc padding) → GRU tính h_512
```

**Cơ chế hoạt động của GRU tại mỗi bước thời gian:**

1. **Tính Reset Gate (r_t)**:
   - Quyết định bao nhiêu thông tin từ hidden state trước (h_{t-1}) cần "quên"
   - Nếu r_t ≈ 0: quên hầu hết thông tin cũ
   - Nếu r_t ≈ 1: giữ lại thông tin cũ

2. **Tính Candidate Hidden State (h̃_t)**:
   - Kết hợp thông tin từ từ hiện tại (x_t) và hidden state đã được reset (r_t * h_{t-1})
   - Đây là "ứng viên" cho hidden state mới

3. **Tính Update Gate (z_t)**:
   - Quyết định bao nhiêu thông tin từ hidden state cũ (h_{t-1}) và bao nhiêu từ candidate mới (h̃_t) được giữ lại

4. **Cập nhật Hidden State (h_t)**:
   - h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
   - Kết hợp thông tin cũ và mới theo tỷ lệ được điều khiển bởi update gate

**Ví dụ trong ngữ cảnh email phishing:**

- Khi GRU đọc "Your account has been compromised":
  - Reset gate có thể "quên" bớt thông tin từ phần lời chào trước đó
  - Update gate cho phép thông tin quan trọng về "compromised" chi phối hidden state
  - Hidden state phản ánh mức độ nguy hiểm tăng lên

- Khi GRU đọc "Please verify immediately":
  - Hidden state tích lũy thêm thông tin về sự gấp gáp và yêu cầu hành động
  - Đây là pattern điển hình của phishing

#### Bước 4: Trích xuất đặc trưng cuối cùng
```
h_512 (hidden state cuối, 64 chiều)
    ↓
Dense(64, ReLU) → kết hợp đặc trưng
    ↓
Dense(1, Sigmoid) → xác suất phishing
```

#### Bước 5: Quyết định phân loại
```
Xác suất = 0.85 (> 0.5)
    ↓
Kết luận: Phishing (Nhãn 1)
```

### 4.2. Ví dụ cụ thể

**Email 1: Phishing**
```
Input: "Congratulations! You've won a free iPhone. Click here to claim your prize!"

Quá trình xử lý:
- Từ "Congratulations" → h_1: [0.1, 0.2, ...] (64 chiều)
- Từ "won" → h_2: [0.15, 0.25, ...] (tích lũy thông tin về giải thưởng)
- Từ "free" → h_3: [0.2, 0.3, ...] (tăng mức độ nghi ngờ)
- Từ "iPhone" → h_4: [0.25, 0.35, ...] (tiếp tục tích lũy)
- Từ "Click" → h_5: [0.4, 0.5, ...] (dấu hiệu phishing rõ ràng)
- ...
- h_512: [0.7, 0.8, 0.6, ...] (hidden state cuối)

→ Dense layers → xác suất = 0.92 → Phishing ✓
```

**Email 2: Benign**
```
Input: "Hi team, just a quick reminder about our meeting tomorrow at 10 AM."

Quá trình xử lý:
- Từ "Hi" → h_1: [0.05, 0.1, ...] (lời chào thân thiện)
- Từ "team" → h_2: [0.08, 0.12, ...] (ngữ cảnh công việc)
- Từ "meeting" → h_3: [0.1, 0.15, ...] (thông tin công việc)
- ...
- h_512: [0.1, 0.15, 0.12, ...] (hidden state cuối, không có dấu hiệu nguy hiểm)

→ Dense layers → xác suất = 0.15 → Benign ✓
```

### 4.3. Đặc điểm GRU học được

GRU tự động học các pattern sau từ dữ liệu:

1. **Pattern từ vựng phishing**:
   - "verify your account", "update payment", "account compromised"
   - "congratulations", "winner", "free", "urgent"

2. **Ngữ cảnh xung quanh từ khóa**:
   - "Do NOT click suspicious links" (cảnh báo, benign)
   - "Click here to verify" (phishing)

3. **Cấu trúc email**:
   - Phishing: Lời chào → Lý do giả mạo → Yêu cầu hành động → Link
   - Benign: Lời chào → Thông tin công việc → Yêu cầu hợp lý

4. **Tông giọng**:
   - Phishing: Gấp gáp, đe dọa, hứa hẹn
   - Benign: Thân thiện, chuyên nghiệp, thông tin

---

## 6. Quy Trình Xây Dựng Mô Hình GRU

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

**Lý do chọn tham số**:
- `max_tokens=20000`: Đủ để bao phủ từ vựng phổ biến, không quá lớn
- `output_sequence_length=512`: Đủ để chứa 95% email, không quá dài

### 5.3. Xây dựng Embedding Layer

```python
gru_model_embedding = layers.Embedding(
    input_dim=20000,
    output_dim=128,
    embeddings_initializer="uniform",
    input_length=512,
    name="embedding_3"
)
```

**Lý do chọn tham số**:
- `output_dim=128`: Đủ để biểu diễn ngữ nghĩa, không quá lớn

### 5.4. Xây dựng mô hình GRU

```python
# Set random seed để tái tạo kết quả
tf.random.set_seed(42)

# Build model
inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = gru_model_embedding(x)
x = layers.GRU(64, return_sequences=False)(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

gru_model = tf.keras.Model(inputs, outputs, name="GRU_model")
```

**Lý do chọn tham số**:
- `GRU(64)`: Đủ để nắm bắt ngữ cảnh, không quá lớn để tránh overfitting
- `return_sequences=False`: Chỉ cần hidden state cuối để phân loại
- `Dense(64)`: Kết hợp đặc trưng từ GRU

### 5.5. Compile mô hình

```python
gru_model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)
```

**Lý do chọn**:
- `binary_crossentropy`: Phù hợp với phân loại nhị phân
- `Adam`: Optimizer hiệu quả, tự điều chỉnh learning rate

### 5.6. Chuẩn bị callbacks

```python
# Early Stopping
gru_early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',     # Theo dõi validation accuracy
    patience=10,                 # Dừng nếu không cải thiện sau 10 epochs
    verbose=1,                   # Hiển thị thông báo
    restore_best_weights=True   # Khôi phục weights tốt nhất
)
```

**Giải thích Early Stopping**:
- **Monitor `val_accuracy`**: Theo dõi độ chính xác trên validation set
- **Patience=10**: Nếu val_accuracy không tăng trong 10 epochs → dừng
- **Lý do**: Tránh overfitting, dừng khi model không còn học được gì mới

# Model Checkpoint
checkpoint_callback = create_checkpoint_callback(
    file_path='outputs/models/GRU/checkpoints/gru_model.keras'
)

# TensorBoard
tensorboard_callback = create_tensorboard_callback(
    file_path='outputs/models/GRU/logs/gru_model_tensorboard'
)
```

### 5.7. Huấn luyện mô hình

```python
# Convert to tensors
X_train_tensor = tf.constant(
    X_train.fillna('').astype(str).values, 
    dtype=tf.string
)
X_val_tensor = tf.constant(
    X_val.fillna('').astype(str).values, 
    dtype=tf.string
)

# Train
gru_model_history = gru_model.fit(
    X_train_tensor, y_train,
    validation_data=(X_val_tensor, y_val),
    epochs=30,
    callbacks=[
        tensorboard_callback,
        gru_early_stopping_callback,
        checkpoint_callback
    ]
)
```

**Quá trình huấn luyện** (thực tế: 10 epochs):
- Epoch 1: Accuracy ~0.85, Loss ~0.35
- Epoch 2: Accuracy ~0.97, Loss ~0.08
- Epoch 3: Accuracy ~0.99, Loss ~0.03
- ...
- Epoch 10: Accuracy ~0.9979, Loss ~0.005 (Early stopping kích hoạt)

### 5.8. Đánh giá mô hình

```python
# Load test data
df_test = pd.read_csv('data/final/final_test.csv')
test_text = df_test['text']
test_labels = df_test['label']

# Preprocess
test_text_processed = test_text.fillna('').astype(str)
test_text_tensor = tf.constant(test_text_processed.values, dtype=tf.string)

# Predict
pred_labels_gru = gru_model.predict(test_text_tensor)
avg_pred_labels_gru = np.round(pred_labels_gru.flatten())
```

### 5.9. Lưu mô hình và metrics

```python
# Save model
gru_model.save('outputs/models/GRU/gru_model.h5')

# Save metrics
gru_metrics = {
    'model_config': {...},
    'test_evaluation': {...},
    'other_test_metrics': {...}
}
with open('outputs/models/GRU/gru_metrics.json', 'w') as f:
    json.dump(gru_metrics, f, indent=4)
```

---

## 6. Kết Quả của Mô Hình GRU

### 6.1. Kết quả trên tập test

Sau khi huấn luyện và đánh giá trên tập test (2,925 email), mô hình GRU đạt được các chỉ số sau:

| Metric | Giá trị |
|--------|---------|
| **Accuracy** | 0.9856 (98.56%) |
| **Precision** | 0.9856 (98.56%) |
| **Recall** | 0.9856 (98.56%) |
| **F1-Score** | 0.9856 (98.56%) |
| **AUC-ROC** | 0.9976 (99.76%) |

### 6.2. Kết quả trên tập validation

| Metric | Giá trị |
|--------|---------|
| **Accuracy** | 0.9832 (98.32%) |
| **Precision** | 0.9833 (98.33%) |
| **Recall** | 0.9832 (98.32%) |
| **F1-Score** | 0.9832 (98.32%) |
| **AUC-ROC** | 0.9979 (99.79%) |

### 6.3. Các chỉ số khác (Test)

| Metric | Giá trị |
|--------|---------|
| **False Discovery Rate (FDR)** | 0.0134 (1.34%) |
| **False Negative Rate (FNR)** | 0.0161 (1.61%) |
| **False Omission Rate (FOR)** | 0.0153 (1.53%) |
| **False Positive Rate (FPR)** | 0.0127 (1.27%) |
| **Negative Prediction Value (NPV)** | 0.9847 (98.47%) |

### 6.4. Phân tích kết quả

**Điểm mạnh**:
- **Accuracy rất cao (98.56%)**: Mô hình phân loại cực kỳ chính xác
- **F1-Score rất cao (98.56%)**: Cân bằng hoàn hảo giữa Precision và Recall
- **FNR thấp (1.61%)**: Ít bỏ sót email phishing (quan trọng trong bảo mật)
- **FPR thấp (1.27%)**: Ít email benign bị nhận nhầm
- **NPV cao (98.47%)**: Khi mô hình dự đoán Benign, khả năng đúng rất cao
- **AUC-ROC rất tốt (99.76%)**: Khả năng phân biệt giữa phishing và benign tốt

### 6.5. Confusion Matrix (Test)

Dựa trên kết quả test với 2,925 email:
- **True Positive (TP)**: ~1,440 email phishing được phát hiện đúng
- **True Negative (TN)**: ~1,443 email benign được phân loại đúng
- **False Positive (FP)**: ~19 email benign bị nhận nhầm là phishing (FPR = 1.27%)
- **False Negative (FN)**: ~23 email phishing bị bỏ sót (FNR = 1.61%)

### 6.6. So sánh với các mô hình khác

| Mô hình | Accuracy (Test) | F1-Score | FNR | AUC-ROC |
|---------|----------------|----------|-----|---------|
| GRU | 98.56% | 98.56% | 1.61% | 99.76% |
| BERT | 99.01% | 99.01% | 0.98% | 99.92% |
| BiLSTM | 98.87% | 98.87% | 1.05% | 99.95% |
| CNN | 98.74% | 98.73% | 2.18% | 99.94% |
| CNN-BiLSTM | 98.63% | 98.63% | 1.61% | 99.89% |

**Nhận xét**: GRU đạt kết quả rất tốt (98.56%), chỉ thấp hơn một chút so với BERT và BiLSTM. Tuy nhiên, GRU nhanh hơn và ít tham số hơn, phù hợp khi cần cân bằng giữa độ chính xác và tốc độ.

### 6.6. Ví dụ dự đoán

**Email được phát hiện đúng là Phishing**:
```
Input: "Congratulations! You've won a free iPhone. Click here to claim your prize!"
Prediction: Phishing (xác suất: 0.92) ✓
```

**Email được phát hiện đúng là Benign**:
```
Input: "Hi team, just a quick reminder about our meeting tomorrow at 10 AM."
Prediction: Benign (xác suất: 0.15) ✓
```

**Email bị nhận nhầm (False Positive)**:
```
Input: "Limited time offer: Get 50% off all products today!"
Prediction: Phishing (xác suất: 0.65) ✗ (Thực tế là marketing hợp pháp)
```

**Email bị bỏ sót (False Negative)**:
```
Input: "Your account needs verification. Please click the link below."
Prediction: Benign (xác suất: 0.45) ✗ (Thực tế là phishing)
```

---

## 7. Giải Thích Dự Đoán (Explainable AI - XAI)

GRU sử dụng **LIME** để giải thích dự đoán, tương tự như CNN và BiLSTM. Xem chi tiết trong phần 7 của TAI_LIEU_CNN.md.

**Đặc điểm của GRU với LIME**:
- GRU nắm bắt ngữ cảnh tuần tự, LIME có thể phát hiện các từ quan trọng dựa trên thứ tự
- Ví dụ: "urgent action required" → các từ này có trọng số dương cao khi xuất hiện cùng nhau

---

## 8. Kết Luận

Mô hình GRU đã được áp dụng thành công trong bài toán phát hiện email phishing với các đặc điểm:

1. **Hiệu quả rất cao**: Đạt accuracy 98.56% trên test set, cải thiện đáng kể so với phiên bản trước
2. **FNR thấp (1.61%)**: Ít bỏ sót email phishing - quan trọng trong bảo mật
3. **AUC-ROC rất tốt (99.76%)**: Khả năng phân biệt giữa phishing và benign tốt
4. **Tốc độ**: Nhanh hơn LSTM và BERT, phù hợp với ứng dụng thời gian thực
5. **Đơn giản**: Kiến trúc đơn giản, dễ hiểu và triển khai
6. **Khả năng học**: Tự động học các pattern phức tạp từ dữ liệu mà không cần feature engineering

Mô hình GRU phù hợp cho các ứng dụng cần cân bằng giữa độ chính xác cao (98.56%) và tốc độ xử lý, đặc biệt trong môi trường có tài nguyên tính toán hạn chế hoặc cần xử lý thời gian thực.

