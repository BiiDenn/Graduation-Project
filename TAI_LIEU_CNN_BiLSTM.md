# Tài Liệu Mô Hình CNN-BiLSTM (Hybrid Convolutional Neural Network - Bidirectional LSTM)

## 1. Khái Niệm Cơ Bản về CNN-BiLSTM

### 1.1. CNN-BiLSTM là gì?

**CNN-BiLSTM** là một mô hình lai (hybrid) kết hợp sức mạnh của **CNN (Convolutional Neural Network)** và **BiLSTM (Bidirectional Long Short-Term Memory)**. Mô hình này tận dụng:
- **CNN**: Phát hiện các pattern cục bộ (n-gram) trong văn bản
- **BiLSTM**: Nắm bắt phụ thuộc dài hạn và ngữ cảnh hai chiều

Bằng cách kết hợp hai kiến trúc này, mô hình có thể vừa phát hiện các cụm từ đặc trưng (CNN) vừa hiểu được cấu trúc và ngữ nghĩa tổng thể của email (BiLSTM).

### 1.2. Đặc điểm chính của CNN-BiLSTM

- **Kết hợp ưu điểm của cả hai**:
  - CNN: Phát hiện pattern cục bộ, nhanh, bất biến vị trí
  - BiLSTM: Nắm bắt ngữ cảnh dài hạn, hiểu thứ tự và cấu trúc

- **Bổ sung cho nhau**:
  - CNN trích xuất đặc trưng cục bộ từ các cụm từ
  - BiLSTM xử lý các đặc trưng này để hiểu ngữ cảnh tổng thể

- **Hiệu quả cao**:
  - Thường đạt kết quả tốt hơn so với chỉ dùng CNN hoặc chỉ dùng BiLSTM
  - Có thể phát hiện cả pattern cục bộ và pattern toàn cục

### 1.3. Cấu trúc CNN-BiLSTM

Mô hình CNN-BiLSTM thường có cấu trúc:

1. **Embedding Layer**: Chuyển từ thành vector
2. **CNN Block**: Trích xuất đặc trưng cục bộ (n-gram patterns)
3. **BiLSTM Block**: Xử lý đặc trưng từ CNN để nắm bắt ngữ cảnh
4. **Dense Layers**: Phân loại dựa trên đặc trưng đã xử lý

**Luồng dữ liệu**:
```
Text → Embedding → CNN (trích xuất pattern) → BiLSTM (hiểu ngữ cảnh) → Dense → Output
```

---

## 2. Ứng Dụng của CNN-BiLSTM trong Project Phát Hiện Email Phishing

### 2.1. Bài toán

Trong project này, CNN-BiLSTM được sử dụng để giải quyết bài toán **phân loại nhị phân email**:
- **Nhãn 0**: Benign (email bình thường, hợp pháp)
- **Nhãn 1**: Phishing (email lừa đảo, giả mạo)

### 2.2. Tại sao chọn CNN-BiLSTM?

1. **Bổ sung điểm mạnh của nhau**:
   - **CNN**: Giỏi phát hiện các cụm từ đặc trưng như "verify your account", "click here", "urgent action"
   - **BiLSTM**: Giỏi hiểu ngữ cảnh và cấu trúc email, phân biệt "Do NOT click" vs "Click here"

2. **Phát hiện pattern phức tạp**:
   - Email phishing có thể có cấu trúc: lời chào → lý do giả mạo → yêu cầu → link
   - CNN phát hiện các cụm từ, BiLSTM hiểu cấu trúc tổng thể

3. **Hiệu quả cao**:
   - Thường đạt kết quả tốt hơn so với chỉ dùng CNN hoặc chỉ dùng BiLSTM
   - Có thể xử lý cả pattern cục bộ và pattern toàn cục

4. **Cân bằng tốt**:
   - CNN nhanh, BiLSTM chính xác → Kết hợp cho kết quả tốt và hợp lý về tốc độ

### 2.3. Dữ liệu sử dụng

- **Tập train**: `final_train.csv` với ~13,648 email (sau khi xử lý mất cân bằng)
- **Tập validation**: `final_val.csv` với ~2,925 email (hoặc chia 30% từ train)
- **Tập test**: `final_test.csv` với ~2,925 email
- **Chia dữ liệu**: 70% train, 15% validation, 15% test (stratified split)

---

## 3. Kiến Trúc Mô Hình CNN-BiLSTM

### 3.1. Tổng quan kiến trúc

```
Input (String) 
    ↓
TextVectorization (max_tokens=20000, sequence_length=512)
    ↓
Embedding (20000 → 128 dimensions)
    ↓
Conv1D (64 filters, kernel_size=3, ReLU)
    ↓
MaxPooling1D (pool_size=2)
    ↓
Dropout (0.3)
    ↓
Bidirectional LSTM (64 units forward + 64 units backward = 128 total)
    ↓
Dropout (0.3)
    ↓
Dense (64 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (32 units, ReLU)
    ↓
Dense (1 unit, Sigmoid)
    ↓
Output (Probability: 0 = Benign, 1 = Phishing)
```

### 3.2. Chi tiết từng lớp

#### 3.2.1. Input Layer
```python
inputs = layers.Input(shape=(1,), dtype=tf.string, name='text_input')
```
- Nhận đầu vào là chuỗi text gốc của email

#### 3.2.2. TextVectorization Layer
```python
text_vectorizer = layers.TextVectorization(
    max_tokens=MAX_TOKENS,  # 20000
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH  # 512
)
text_vectorizer.adapt(X_train)
```
- Chuyển đổi text thành chuỗi token IDs
- Mỗi email → vector `[id_1, id_2, ..., id_512]`

#### 3.2.3. Embedding Layer
```python
embedding_layer = layers.Embedding(
    input_dim=MAX_TOKENS,  # 20000
    output_dim=EMBEDDING_DIM,  # 256
    input_length=MAX_SEQUENCE_LENGTH,  # 512
    name='hybrid_embedding'
)
```
- Chuyển mỗi token ID thành vector 256 chiều
- Kết quả: Tensor `(batch_size, 512, 256)`

#### 3.2.4. CNN Block

```python
# CNN Layer 1
x = layers.Conv1D(
    filters=128,
    kernel_size=3,
    padding='same',
    activation='relu',
    name='conv1d_layer_3'
)(x)

# CNN Layer 2
x = layers.Conv1D(
    filters=128,
    kernel_size=5,
    padding='same',
    activation='relu',
    name='conv1d_layer_5'
)(x)
x = layers.MaxPooling1D(pool_size=2, name='maxpool')(x)
x = layers.Dropout(0.3, name='dropout_after_cnn')(x)
```

**Cơ chế hoạt động**:

1. **Conv1D Layer 1 (128 filters, kernel_size=3)**:
   - Phát hiện pattern 3-gram (cụm 3 từ liên tiếp)
   - 128 filters học 128 pattern khác nhau
   - `padding='same'`: Giữ nguyên độ dài chuỗi sau convolution
   - Kết quả: `(batch_size, 512, 128)`

2. **Conv1D Layer 2 (128 filters, kernel_size=5)**:
   - Phát hiện pattern 5-gram (cụm 5 từ liên tiếp)
   - Học các pattern phức tạp hơn từ output của layer 1
   - Kết quả: `(batch_size, 512, 128)`

3. **MaxPooling1D (pool_size=2)**:
   - Giảm độ dài chuỗi xuống còn một nửa
   - Lấy giá trị lớn nhất trong mỗi cửa sổ 2
   - Kết quả: `(batch_size, 256, 128)`
   - **Mục đích**: Giảm chiều, tăng tính khái quát, giảm tính toán cho BiLSTM

4. **Dropout (0.3)**:
   - Giảm overfitting

**Ví dụ minh họa**:

```
Email: "Your account has been compromised. Please verify immediately."

Sau Embedding: (512, 128)

Conv1D với kernel_size=3 quét:
  Vị trí 1-3: ["Your", "account", "has"]
    → Filter 1: Kích hoạt = 0.7 (pattern "account has")
  Vị trí 2-4: ["account", "has", "been"]
    → Filter 1: Kích hoạt = 0.8
  Vị trí 3-5: ["has", "been", "compromised"]
    → Filter 1: Kích hoạt = 0.9 (pattern "been compromised")
  Vị trí 6-8: ["Please", "verify", "immediately"]
    → Filter 2: Kích hoạt = 0.95 (pattern "verify immediately")

Kết quả Conv1D: (512, 64) - mỗi vị trí có 64 giá trị kích hoạt

MaxPooling1D (pool_size=2):
  Vị trí 1-2: max(0.7, 0.8) = 0.8
  Vị trí 3-4: max(0.9, 0.1) = 0.9
  Vị trí 5-6: max(0.2, 0.95) = 0.95
  ...

Kết quả: (256, 64) - giảm độ dài, giữ lại đặc trưng quan trọng
```

#### 3.2.5. BiLSTM Block

```python
# BiLSTM Layer 1
x = layers.Bidirectional(
    layers.LSTM(128, return_sequences=True),
    name='bilstm_layer_1'
)(x)

# BiLSTM Layer 2
x = layers.Bidirectional(
    layers.LSTM(128, return_sequences=False),
    name='bilstm_layer_2'
)(x)
x = layers.Dropout(0.3, name='dropout_after_bilstm')(x)
```

**Cơ chế hoạt động**:

1. **Bidirectional LSTM Layer 1 (128 units mỗi chiều, return_sequences=True)**:
   - **Forward LSTM**: Đọc từ đầu đến cuối (256 vị trí)
   - **Backward LSTM**: Đọc từ cuối đến đầu (256 vị trí)
   - Với `return_sequences=True`: Trả về hidden states cho tất cả vị trí
   - Kết quả: Tensor `(batch_size, 256, 256)` (256 = 128 forward + 128 backward)

2. **Bidirectional LSTM Layer 2 (128 units mỗi chiều, return_sequences=False)**:
   - Xử lý output từ layer 1
   - Với `return_sequences=False`: Chỉ lấy hidden state cuối cùng
   - Kết quả: Vector 256 chiều (128 từ forward + 128 từ backward)

3. **Ý nghĩa**:
   - CNN đã trích xuất các đặc trưng cục bộ (pattern n-gram)
   - BiLSTM xử lý các đặc trưng này để hiểu ngữ cảnh và cấu trúc tổng thể
   - Kết hợp: Vừa biết pattern cục bộ, vừa hiểu ngữ cảnh

**Ví dụ minh họa**:

```
Sau CNN: (256, 64) - mỗi vị trí có 64 đặc trưng từ CNN

BiLSTM xử lý:
  Forward LSTM đọc từ vị trí 1 → 256:
    - Vị trí 1: h_1^f (thông tin về pattern đầu)
    - Vị trí 2: h_2^f (tích lũy thông tin)
    - ...
    - Vị trí 256: h_256^f (tổng hợp toàn bộ)

  Backward LSTM đọc từ vị trí 256 → 1:
    - Vị trí 256: h_256^b (thông tin về pattern cuối)
    - Vị trí 255: h_255^b (tích lũy thông tin)
    - ...
    - Vị trí 1: h_1^b (tổng hợp toàn bộ)

  Ghép lại: h_256 = [h_256^f; h_256^b] (128 chiều)
  → Có thông tin đầy đủ về cả pattern cục bộ (từ CNN) và ngữ cảnh (từ BiLSTM)
```

#### 3.2.6. Dense Layers

```python
x = layers.Dense(128, activation='relu', name='dense_1')(x)
outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
```

- **Dense(128, ReLU)**: Kết hợp các đặc trưng từ BiLSTM (từ 256 chiều)
- **Dense(1, Sigmoid)**: Trả về xác suất email là phishing

### 3.3. Tổng số tham số

Mô hình CNN-BiLSTM có khoảng **10.5 triệu tham số**, chủ yếu từ:
- Embedding: 20000 × 256 = 5,120,000
- Conv1D: ~50,000
- BiLSTM: ~800,000
- Dense layers: ~12,000

---

## 4. Trích Xuất Đặc Trưng (Feature Extraction)

### 4.1. Quá trình trích xuất đặc trưng của CNN-BiLSTM

CNN-BiLSTM kết hợp hai phương pháp trích xuất đặc trưng:

1. **Embedding Layer**: Chuyển token IDs thành vector embedding
   - Input: `[id_1, id_2, ..., id_512]` (token IDs)
   - Output: `(batch_size, 512, 128)` (mỗi token → vector 128 chiều)

2. **CNN Block**: Trích xuất pattern cục bộ (n-gram)
   - Conv1D (kernel_size=3) → Phát hiện 3-gram patterns
   - MaxPooling1D → Giảm chiều, giữ pattern quan trọng nhất
   - Output: `(batch_size, 256, 64)` (sau pooling)
   - **Ý nghĩa**: CNN phát hiện các cụm từ đặc trưng cục bộ

3. **BiLSTM Block**: Nắm bắt ngữ cảnh dài hạn
   - Nhận đặc trưng từ CNN
   - Forward LSTM + Backward LSTM → Ngữ cảnh hai chiều
   - Output: `(batch_size, 128)` (64 từ forward + 64 từ backward)
   - **Ý nghĩa**: Hiểu cấu trúc và ngữ nghĩa tổng thể của email

4. **Dense Layers**: Kết hợp đặc trưng để phân loại
   - Dense(64) → Kết hợp đặc trưng từ BiLSTM
   - Dense(32) → Tiếp tục kết hợp
   - Dense(1, sigmoid) → Xác suất phishing

### 4.2. Đặc điểm của đặc trưng CNN-BiLSTM

- **Kết hợp ưu điểm**: Pattern cục bộ (CNN) + Ngữ cảnh dài hạn (BiLSTM)
- **Hiệu quả cao**: Thường đạt kết quả tốt hơn chỉ dùng CNN hoặc chỉ dùng BiLSTM
- **Cân bằng**: Tốc độ và độ chính xác đều tốt

---

## 5. Quy Trình Hoạt Động của CNN-BiLSTM để Nhận Diện Email

### 4.1. Quy trình tổng thể

Khi một email mới được đưa vào mô hình CNN-BiLSTM, quy trình xử lý diễn ra như sau:

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

#### Bước 3: CNN trích xuất đặc trưng cục bộ

**Conv1D quét với kernel_size=3**:

```
Vị trí 1-3: ["Your", "account", "has"]
  → Filter 1: Kích hoạt = 0.7
  → Filter 2: Kích hoạt = 0.3
  → ...

Vị trí 2-4: ["account", "has", "been"]
  → Filter 1: Kích hoạt = 0.8
  → Filter 2: Kích hoạt = 0.4
  → ...

Vị trí 3-5: ["has", "been", "compromised"]
  → Filter 1: Kích hoạt = 0.9 (pattern "been compromised")
  → Filter 2: Kích hoạt = 0.5
  → ...

Vị trí 6-8: ["Please", "verify", "immediately"]
  → Filter 2: Kích hoạt = 0.95 (pattern "verify immediately")
  → Filter 3: Kích hoạt = 0.7
  → ...

Kết quả Conv1D: (512, 64) - mỗi vị trí có 64 giá trị kích hoạt
```

**MaxPooling1D**:

```
(512, 64)
    ↓
MaxPooling1D (pool_size=2)
    ↓
(256, 64) - giảm độ dài, giữ lại đặc trưng quan trọng
```

#### Bước 4: BiLSTM xử lý ngữ cảnh

**Forward LSTM**:
```
Vị trí 1: h_1^f (thông tin về pattern đầu)
Vị trí 2: h_2^f (tích lũy)
...
Vị trí 256: h_256^f (tổng hợp toàn bộ)
```

**Backward LSTM**:
```
Vị trí 256: h_256^b (thông tin về pattern cuối)
Vị trí 255: h_255^b (tích lũy)
...
Vị trí 1: h_1^b (tổng hợp toàn bộ)
```

**Ghép lại**:
```
h_256 = [h_256^f; h_256^b] (128 chiều)
→ Có thông tin về cả pattern cục bộ (CNN) và ngữ cảnh (BiLSTM)
```

#### Bước 5: Dense Layers
```
Vector 128 chiều
    ↓
Dense(64, ReLU) → kết hợp đặc trưng
    ↓
Dense(32, ReLU) → tiếp tục kết hợp
    ↓
Dense(1, Sigmoid) → xác suất phishing
```

#### Bước 6: Quyết định phân loại
```
Xác suất = 0.93 (> 0.5)
    ↓
Kết luận: Phishing (Nhãn 1)
```

### 4.2. Ví dụ cụ thể

**Email 1: Phishing**
```
Input: "Congratulations! You've won a free iPhone. Click here to claim your prize!"

Quá trình xử lý:
1. TextVectorization → Token IDs
2. Embedding → (512, 128)
3. CNN (Conv1D + MaxPooling):
   - Pattern "Congratulations you won": Filter 1 kích hoạt = 0.95
   - Pattern "free iPhone": Filter 2 kích hoạt = 0.9
   - Pattern "Click here": Filter 3 kích hoạt = 0.9
   - → (256, 64) - đặc trưng cục bộ
4. BiLSTM:
   - Forward: Tích lũy thông tin từ đầu → cuối
   - Backward: Tích lũy thông tin từ cuối → đầu
   - → (128,) - ngữ cảnh tổng thể
5. Dense layers → xác suất = 0.94
→ Phishing ✓
```

**Email 2: Benign**
```
Input: "Hi team, just a quick reminder about our meeting tomorrow at 10 AM."

Quá trình xử lý:
1. TextVectorization → Token IDs
2. Embedding → (512, 128)
3. CNN:
   - Pattern "meeting schedule": Filter 10 kích hoạt = 0.6
   - Pattern "team reminder": Filter 11 kích hoạt = 0.5
   - Không có pattern phishing mạnh
   - → (256, 64)
4. BiLSTM:
   - Hiểu ngữ cảnh: Email công việc, thân thiện
   - → (128,)
5. Dense layers → xác suất = 0.15
→ Benign ✓
```

### 4.3. Đặc điểm CNN-BiLSTM học được

1. **Pattern cục bộ (từ CNN)**:
   - "verify your account"
   - "click here to"
   - "urgent action required"
   - "congratulations you won"

2. **Ngữ cảnh và cấu trúc (từ BiLSTM)**:
   - Cấu trúc email phishing: lời chào → lý do → yêu cầu → link
   - Ngữ cảnh phủ định: "Do NOT click" vs "Click here"
   - Tông giọng: gấp gáp, đe dọa vs thân thiện, chuyên nghiệp

3. **Kết hợp**:
   - Nhiều pattern phishing + ngữ cảnh đáng ngờ → xác suất phishing cao
   - Pattern công việc + ngữ cảnh hợp lý → xác suất benign cao

---

## 6. Quy Trình Xây Dựng Mô Hình CNN-BiLSTM

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
# Hyperparameters
MAX_TOKENS = 20000
MAX_SEQUENCE_LENGTH = 512
EMBEDDING_DIM = 128
CNN_FILTERS = 64
CNN_KERNEL_SIZE = 3
BILSTM_UNITS = 64

# Text Vectorization
text_vectorizer = layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH
)
text_vectorizer.adapt(X_train)
```

### 5.3. Xây dựng Embedding Layer

```python
embedding_layer = layers.Embedding(
    input_dim=MAX_TOKENS,
    output_dim=EMBEDDING_DIM,
    input_length=MAX_SEQUENCE_LENGTH,
    name='hybrid_embedding'
)
```

### 5.4. Xây dựng mô hình CNN-BiLSTM

```python
def build_hybrid_cnn_bilstm_model():
    # Input layer
    inputs = layers.Input(shape=(1,), dtype=tf.string, name='text_input')
    
    # Text Vectorization
    x = text_vectorizer(inputs)
    
    # Embedding
    x = embedding_layer(x)
    
    # CNN Block
    x = layers.Conv1D(
        filters=CNN_FILTERS,  # 64
        kernel_size=CNN_KERNEL_SIZE,  # 3
        activation='relu',
        padding='same',
        name='conv1d_block'
    )(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool')(x)
    x = layers.Dropout(0.3, name='dropout_after_cnn')(x)
    
    # BiLSTM Block
    x = layers.Bidirectional(
        layers.LSTM(BILSTM_UNITS, return_sequences=False),
        name='bidirectional_lstm'
    )(x)
    x = layers.Dropout(0.3, name='dropout_after_bilstm')(x)
    
    # Dense layers
    x = layers.Dense(64, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.3, name='dropout_dense')(x)
    x = layers.Dense(32, activation='relu', name='dense_2')(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Hybrid_CNN_BiLSTM')
    
    return model

# Build model
hybrid_model = build_hybrid_cnn_bilstm_model()
```

**Lý do chọn tham số**:
- `CNN_FILTERS=64`: Đủ để học nhiều pattern
- `CNN_KERNEL_SIZE=3`: Phát hiện 3-gram (cụm 3 từ)
- `MaxPooling1D`: Giảm chiều, tăng tính khái quát
- `BILSTM_UNITS=64`: Đủ để nắm bắt ngữ cảnh

### 5.5. Compile mô hình

```python
hybrid_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### 5.6. Chuẩn bị callbacks

```python
callbacks = [
    create_tensorboard_callback(
        file_path='outputs/models/Hybrid_CNN_BiLSTM/logs/tensorboard'
    ),
    create_checkpoint_callback(
        file_path='outputs/models/Hybrid_CNN_BiLSTM/checkpoints/hybrid_model.keras'
    ),
    create_early_stopping_callback(
        monitor='val_loss',      # Theo dõi validation loss
        patience=10,              # Dừng nếu không cải thiện sau 10 epochs
        restore_best_weights=True # Khôi phục weights tốt nhất
    )
]
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
history = hybrid_model.fit(
    X_train_tensor,
    y_train,
    validation_data=(X_val_tensor, y_val),
    batch_size=32,
    epochs=30,
    callbacks=callbacks
)
```

**Quá trình huấn luyện** (thực tế: 4 epochs):
- Epoch 1: Accuracy ~0.99, Loss ~0.03
- Epoch 2: Accuracy ~0.99, Loss ~0.015
- Epoch 3: Accuracy ~0.9938, Loss ~0.01
- Epoch 4: Accuracy ~0.9938, Loss ~0.01 (Early stopping kích hoạt)

### 5.8. Đánh giá mô hình

```python
# Load test data
df_test = pd.read_csv('data/final/final_test.csv')
X_test = df_test['text']
y_test = df_test['label']

# Convert to tensor
X_test_tensor = tf.constant(
    X_test.fillna('').astype(str).values, 
    dtype=tf.string
)

# Predict
y_pred_proba = hybrid_model.predict(X_test_tensor)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()
```

### 5.9. Lưu mô hình và metrics

```python
# Save model
hybrid_model.save('outputs/models/Hybrid_CNN_BiLSTM/hybrid_cnn_bilstm_model.keras')

# Save vocabulary
vocab = text_vectorizer.get_vocabulary()
with open('outputs/models/Hybrid_CNN_BiLSTM/hybrid_embedding_metadata.tsv', 'w', encoding='utf-8') as f:
    for word in vocab:
        f.write(word + '\n')

# Save training history
history_dict = {
    'loss': [float(x) for x in history.history['loss']],
    'accuracy': [float(x) for x in history.history['accuracy']],
    'val_loss': [float(x) for x in history.history['val_loss']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']]
}
with open('outputs/models/Hybrid_CNN_BiLSTM/training_history.json', 'w') as f:
    json.dump(history_dict, f, indent=2)
```

---

## 6. Kết Quả của Mô Hình CNN-BiLSTM

### 6.1. Kết quả trên tập test

Sau khi huấn luyện và đánh giá trên tập test (2,925 email), mô hình CNN-BiLSTM đạt được các chỉ số sau:

| Metric | Giá trị |
|--------|---------|
| **Accuracy** | 0.9863 (98.63%) |
| **Precision** | 0.9863 (98.63%) |
| **Recall** | 0.9863 (98.63%) |
| **F1-Score** | 0.9863 (98.63%) |
| **AUC-ROC** | 0.9989 (99.89%) |

### 6.2. Kết quả trên tập validation

| Metric | Giá trị |
|--------|---------|
| **Accuracy** | 0.986 (98.6%) |
| **Precision** | 0.986 (98.6%) |
| **Recall** | 0.986 (98.6%) |
| **F1-Score** | 0.986 (98.6%) |
| **AUC-ROC** | 0.9988 (99.88%) |

### 6.3. Các chỉ số khác (Test)

| Metric | Giá trị |
|--------|---------|
| **False Discovery Rate (FDR)** | 0.012 (1.2%) |
| **False Negative Rate (FNR)** | 0.0161 (1.61%) |
| **False Omission Rate (FOR)** | 0.0153 (1.53%) |
| **False Positive Rate (FPR)** | 0.0113 (1.13%) |
| **Negative Prediction Value (NPV)** | 0.9847 (98.47%) |

### 6.4. Phân tích kết quả

**Điểm mạnh**:
- **Accuracy rất cao (98.63%)**: Mô hình phân loại cực kỳ chính xác
- **F1-Score rất cao (98.63%)**: Cân bằng hoàn hảo giữa Precision và Recall
- **FNR thấp (1.61%)**: Ít bỏ sót email phishing - quan trọng trong bảo mật
- **FPR thấp (1.13%)**: Ít email benign bị nhận nhầm
- **AUC-ROC gần hoàn hảo (99.89%)**: Khả năng phân biệt giữa phishing và benign rất tốt
- **Kết hợp tốt**: Tận dụng được ưu điểm của cả CNN và BiLSTM
- **Phát hiện pattern đa dạng**: Vừa phát hiện pattern cục bộ, vừa hiểu ngữ cảnh

**So sánh với các mô hình khác**:

| Mô hình | Accuracy (Test) | F1-Score | FNR | AUC-ROC | Đặc điểm |
|---------|----------------|----------|-----|---------|----------|
| CNN-BiLSTM | 98.63% | 98.63% | 1.61% | 99.89% | Kết hợp CNN + BiLSTM |
| BERT | 99.01% | 99.01% | 0.98% | 99.92% | Pre-trained transformer |
| BiLSTM | 98.87% | 98.87% | 1.05% | 99.95% | Chỉ BiLSTM |
| CNN | 98.74% | 98.73% | 2.18% | 99.94% | Chỉ CNN |
| GRU | 98.56% | 98.56% | 1.61% | 99.76% | Chỉ GRU |

**Nhận xét**: CNN-BiLSTM đạt kết quả rất tốt (98.63%), nhờ kết hợp được ưu điểm của cả hai kiến trúc. Mặc dù không tốt nhất, nhưng sự kết hợp này cho phép phát hiện cả pattern cục bộ và ngữ cảnh dài hạn.

### 6.5. Training History

Mô hình được huấn luyện trong 4 epochs (thực tế):

- **Epoch 1**: Training Accuracy: ~0.99, Validation Accuracy: ~0.98
- **Epoch 2**: Training Accuracy: ~0.99, Validation Accuracy: ~0.986
- **Epoch 3**: Training Accuracy: ~0.99, Validation Accuracy: ~0.986
- **Epoch 4**: Training Accuracy: ~0.9938, Validation Accuracy: ~0.986 (Early stopping kích hoạt)

**Nhận xét**:
- Mô hình hội tụ rất nhanh (epoch 1 đã đạt >98%)
- Validation accuracy ổn định ở mức ~98.6%
- Không có dấu hiệu overfitting rõ ràng
- Chỉ cần 4 epochs để đạt kết quả tốt, chứng tỏ kiến trúc hybrid hiệu quả

### 6.6. Confusion Matrix (Test)

Dựa trên kết quả test với 2,925 email:
- **True Positive (TP)**: ~1,440 email phishing được phát hiện đúng
- **True Negative (TN)**: ~1,443 email benign được phân loại đúng
- **False Positive (FP)**: ~17 email benign bị nhận nhầm là phishing (FPR = 1.13%)
- **False Negative (FN)**: ~23 email phishing bị bỏ sót (FNR = 1.61%)

### 6.7. Ví dụ dự đoán

**Email được phát hiện đúng là Phishing**:
```
Input: "Your bank account has been compromised. Please verify your details immediately at this link: http://malicious-site.com/verify"
Prediction: Phishing (xác suất: ~0.96) ✓
→ CNN phát hiện: "account compromised", "verify", "link"
→ BiLSTM hiểu: Cấu trúc email phishing điển hình
```

**Email được phát hiện đúng là Benign**:
```
Input: "Hi team, just a quick reminder about our meeting tomorrow at 10 AM. Please prepare your reports."
Prediction: Benign (xác suất: ~0.12) ✓
→ CNN phát hiện: "meeting", "reminder" (pattern công việc)
→ BiLSTM hiểu: Ngữ cảnh thân thiện, chuyên nghiệp
```

---

## 7. Giải Thích Dự Đoán (Explainable AI - XAI)

CNN-BiLSTM sử dụng **LIME** để giải thích dự đoán, tương tự như các model Keras khác. Xem chi tiết trong phần 7 của TAI_LIEU_CNN.md.

**Đặc điểm của CNN-BiLSTM với LIME**:
- LIME có thể phát hiện cả pattern cục bộ (CNN) và ngữ cảnh dài hạn (BiLSTM)
- Ví dụ: "verify your account" → pattern cục bộ (CNN) + ngữ cảnh "urgent action" (BiLSTM) → trọng số dương cao

---

## 8. Kết Luận

Mô hình CNN-BiLSTM đã được áp dụng thành công trong bài toán phát hiện email phishing với các đặc điểm:

1. **Hiệu quả rất cao**: Đạt accuracy 98.63% trên test set, tốt thứ tư trong các mô hình
2. **AUC-ROC gần hoàn hảo (99.89%)**: Khả năng phân biệt giữa phishing và benign rất tốt
3. **FNR thấp (1.61%)**: Ít bỏ sót email phishing - quan trọng trong bảo mật
4. **Kết hợp tốt**: Tận dụng được ưu điểm của cả CNN (pattern cục bộ) và BiLSTM (ngữ cảnh)
5. **Phát hiện đa dạng**: Có thể phát hiện cả pattern cục bộ và pattern toàn cục
6. **Hội tụ nhanh**: Chỉ cần 4 epochs để đạt kết quả tốt, chứng tỏ kiến trúc hybrid hiệu quả
7. **Cân bằng**: Tốc độ hợp lý (nhanh hơn BERT), độ chính xác cao

CNN-BiLSTM là lựa chọn tốt cho bài toán phát hiện email phishing khi cần độ chính xác cao (98.63%) và muốn tận dụng ưu điểm của cả CNN và BiLSTM. Đặc biệt phù hợp khi cần phát hiện cả pattern cục bộ và ngữ cảnh dài hạn.

