# Tài Liệu Mô Hình CNN (Convolutional Neural Network)

## 1. Khái Niệm Cơ Bản về CNN

### 1.1. CNN là gì?

**CNN (Convolutional Neural Network)** là một kiến trúc mạng nơ-ron sâu được thiết kế đặc biệt để xử lý dữ liệu có cấu trúc lưới, như hình ảnh hoặc chuỗi. Trong xử lý văn bản, CNN được sử dụng với **Conv1D (1-Dimensional Convolution)** để phát hiện các pattern cục bộ trong chuỗi từ, tương tự như cách CNN 2D phát hiện các đặc trưng trong hình ảnh.

### 1.2. Đặc điểm chính của CNN

- **Phát hiện pattern cục bộ**: 
  - CNN sử dụng các bộ lọc (filters/kernels) để quét qua chuỗi và phát hiện các pattern cục bộ
  - Trong văn bản, các pattern này thường là n-gram (cụm từ liên tiếp)

- **Tính bất biến vị trí**: 
  - CNN không quan tâm đến vị trí chính xác của pattern trong chuỗi
  - Chỉ cần pattern xuất hiện ở đâu đó trong email là đủ để kích hoạt

- **Hiệu quả tính toán**: 
  - CNN nhanh hơn RNN vì có thể xử lý song song
  - Ít tham số hơn so với RNN/LSTM

- **Trích xuất đặc trưng tự động**: 
  - CNN tự động học các đặc trưng từ dữ liệu mà không cần feature engineering thủ công

### 1.3. Cấu trúc CNN cho văn bản

CNN cho văn bản thường bao gồm:

1. **Embedding Layer**: Chuyển từ thành vector
2. **Conv1D Layer**: Phát hiện pattern n-gram
3. **Pooling Layer**: Giảm chiều và trích xuất đặc trưng quan trọng
4. **Dense Layers**: Phân loại dựa trên đặc trưng đã trích xuất

**Công thức toán học của Conv1D**:

```
Cho input X có shape (batch, sequence_length, embedding_dim):
- Filter W có shape (kernel_size, embedding_dim, num_filters)
- Convolution: Y[i] = f(Σ(W * X[i:i+kernel_size]) + b)
- Trong đó:
  * i là vị trí trong chuỗi
  * kernel_size là kích thước cửa sổ (ví dụ: 5 từ)
  * f là hàm activation (ví dụ: ReLU)
```

### 1.4. So sánh với RNN

| Đặc điểm | CNN | RNN/LSTM |
|---------|-----|----------|
| Xử lý pattern | Cục bộ (n-gram) | Toàn cục (toàn bộ chuỗi) |
| Thứ tự | Ít quan trọng | Rất quan trọng |
| Tốc độ | Nhanh (song song) | Chậm hơn (tuần tự) |
| Bộ nhớ | Không có | Có (hidden state) |
| Phù hợp | Pattern cục bộ | Phụ thuộc dài hạn |

---

## 2. Ứng Dụng của CNN trong Project Phát Hiện Email Phishing

### 2.1. Bài toán

Trong project này, CNN được sử dụng để giải quyết bài toán **phân loại nhị phân email**:
- **Nhãn 0**: Benign (email bình thường, hợp pháp)
- **Nhãn 1**: Phishing (email lừa đảo, giả mạo)

### 2.2. Tại sao chọn CNN?

1. **Phát hiện pattern n-gram**: 
   - Email phishing thường chứa các cụm từ đặc trưng như "verify your account", "click here", "urgent action required"
   - CNN rất giỏi phát hiện các pattern cục bộ này

2. **Tính bất biến vị trí**: 
   - Pattern "click here" có thể xuất hiện ở đầu, giữa hoặc cuối email
   - CNN không quan tâm vị trí, chỉ cần pattern xuất hiện là đủ

3. **Tốc độ xử lý**: 
   - CNN nhanh hơn RNN vì có thể xử lý song song
   - Phù hợp với ứng dụng cần xử lý nhiều email cùng lúc

4. **Đơn giản và hiệu quả**: 
   - Kiến trúc đơn giản hơn RNN
   - Dễ huấn luyện và điều chỉnh

### 2.3. Dữ liệu sử dụng

- **Tập train**: `final_train.csv` với ~13,648 email (sau khi xử lý mất cân bằng)
- **Tập validation**: `final_val.csv` với ~2,925 email (hoặc chia 30% từ train)
- **Tập test**: `final_test.csv` với ~2,925 email
- **Chia dữ liệu**: 70% train, 15% validation, 15% test (stratified split)
- **Lưu ý**: Trong quá trình training, có thể chia tiếp từ `final_train.csv` thành train/val với `stratify=y` để đảm bảo phân phối lớp đồng đều

---

## 3. Kiến Trúc Mô Hình CNN

### 3.1. Tổng quan kiến trúc

```
Input (Token IDs, shape: batch × 512)
    ↓
Embedding (20000 → 128 dimensions)
    ↓
Conv1D (32 filters, kernel_size=5, ReLU)
    ↓
Dropout (0.2)
    ↓
GlobalMaxPool1D
    ↓
Dropout (0.2)
    ↓
Dense (64 units, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense (1 unit, Sigmoid)
    ↓
Output (Probability: 0 = Benign, 1 = Phishing)
```

### 3.2. Chi tiết từng lớp

#### 3.2.1. Input Layer và Tiền Xử Lý

**Cách Model Nhận Input**:

1. **Input ban đầu**: Email text dạng string (từ extension hoặc dataset)
   ```python
   email_text = "Your account has been suspended. Please verify..."
   ```

2. **TextVectorization** (thực hiện **ngoài mô hình** trong training, nhưng có thể tích hợp trong inference):
   ```python
   text_vectorizer = layers.TextVectorization(
       max_tokens=20000,        # Chỉ giữ 20,000 từ phổ biến nhất
       output_mode='int',       # Trả về token IDs (số nguyên)
       output_sequence_length=512,  # Cắt/pad thành đúng 512 token
   )
   text_vectorizer.adapt(X_train.astype(str))  # Học vocabulary từ train
   ```
   
   **Quá trình**:
   - Email text → Tokenization (tách thành từ)
   - Mỗi từ → Token ID (số nguyên từ 0 đến 19999)
   - Từ không có trong vocabulary → `[UNK]` (ID = 1)
   - Email ngắn → Padding với 0
   - Email dài → Truncate (cắt bớt)
   - Kết quả: Vector `[id_1, id_2, ..., id_512]`

3. **Input Layer của Model**:
   ```python
   inputs = layers.Input(shape=(None,), dtype="float32")
   ```
   - Nhận đầu vào là chuỗi token IDs (đã được TextVectorization xử lý)
   - Shape: `(batch_size, 512)`

**Lưu ý quan trọng**: 
- Trong training, TextVectorization được thực hiện **ngoài mô hình**:
  ```python
  processed_train_text = text_vectorizer(X_train)
  cnn_model.fit(processed_train_text, y_train, ...)
  ```
- Trong inference (qua extension), ModelLoader sẽ tự động xử lý TextVectorization nếu model không có layer này tích hợp sẵn

#### 3.2.2. Embedding Layer
```python
x = layers.Embedding(
    input_dim=len(words_in_vocab),  # ≈ 20000
    output_dim=256,
    input_length=512
)(inputs)
```
- **Chức năng**: Chuyển mỗi token ID thành vector 256 chiều
- **Kết quả**: Tensor `(batch_size, 512, 256)`
- **Vai trò**: Tạo không gian đặc trưng mà CNN sẽ quét để tìm pattern

#### 3.2.3. Conv1D Layers (3 layers)
```python
# Conv Block 1
x = layers.Conv1D(
    filters=128,
    kernel_size=5,
    padding="same",
    activation="relu",
    kernel_regularizer=regularizers.l2(1e-4),
    name="conv1"
)(x)
x = layers.BatchNormalization(name="bn1")(x)
x = layers.Dropout(0.3, name="dropout1")(x)

# Conv Block 2
x = layers.Conv1D(
    filters=256,
    kernel_size=5,
    padding="same",
    activation="relu",
    kernel_regularizer=regularizers.l2(1e-4),
    name="conv2"
)(x)
x = layers.BatchNormalization(name="bn2")(x)
x = layers.Dropout(0.3, name="dropout2")(x)

# Conv Block 3
x = layers.Conv1D(
    filters=256,
    kernel_size=3,
    padding="same",
    activation="relu",
    kernel_regularizer=regularizers.l2(1e-4),
    name="conv3"
)(x)
x = layers.BatchNormalization(name="bn3")(x)
```

**Cơ chế hoạt động**:

1. **Conv Block 1 (128 filters, kernel_size=5)**:
   - Phát hiện pattern 5-gram (cụm 5 từ)
   - 128 filters học 128 pattern khác nhau
   - BatchNormalization + Dropout(0.3) để ổn định và tránh overfitting

2. **Conv Block 2 (256 filters, kernel_size=5)**:
   - Tiếp tục phát hiện pattern 5-gram với nhiều filters hơn
   - Học các pattern phức tạp hơn từ output của block 1

3. **Conv Block 3 (256 filters, kernel_size=3)**:
   - Phát hiện pattern 3-gram (cụm 3 từ)
   - Học các pattern ngắn hơn, chi tiết hơn

**Ví dụ minh họa**:

```
Email: "Your account has been compromised. Please verify immediately."

Sau Embedding: (512, 128) - mỗi từ là vector 128 chiều

Conv1D với kernel_size=5 quét:
  Vị trí 1-5: ["Your", "account", "has", "been", "compromised"]
    → Filter có thể kích hoạt mạnh (pattern "account has been compromised")
  Vị trí 2-6: ["account", "has", "been", "compromised", "Please"]
    → Filter có thể kích hoạt mạnh
  Vị trí 6-10: ["Please", "verify", "immediately", ...]
    → Filter khác kích hoạt mạnh (pattern "Please verify")

Kết quả: (512, 32) - mỗi vị trí có 32 giá trị kích hoạt từ 32 filters
```

#### 3.2.4. Dropout Layer
```python
x = layers.Dropout(0.2)(x)
```
- **Chức năng**: Ngẫu nhiên bỏ 20% neuron trong quá trình train
- **Mục đích**: Giảm overfitting, tăng khả năng khái quát hóa

#### 3.2.4. GlobalMaxPool1D Layer
```python
x = layers.GlobalMaxPool1D()(x)
```

**Cơ chế hoạt động**:
- Nhận đầu vào: `(batch_size, sequence_length=512, filters=256)` (từ Conv Block 3)
- Với mỗi filter, lấy **giá trị lớn nhất** trên toàn bộ chuỗi
- Kết quả: `(batch_size, 256)` - mỗi filter cho một giá trị

**Ý nghĩa**:
- Không quan tâm pattern xuất hiện ở vị trí nào
- Chỉ cần pattern xuất hiện ở đâu đó trong email → filter kích hoạt
- Phù hợp với bài toán: "Email có chứa pattern X hay không?"

**Ví dụ**:
```
Filter 1 (phát hiện "verify your account"):
  Kích hoạt tại vị trí 10: 0.8
  Kích hoạt tại vị trí 50: 0.6
  Kích hoạt tại vị trí 200: 0.9
  → GlobalMaxPool1D lấy: 0.9 (giá trị lớn nhất)

→ Email này có chứa pattern "verify your account" (không quan tâm vị trí)
```

#### 3.2.5. Dense Layers
```python
x = layers.Dense(128, activation="relu", ...)(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation="relu", ...)(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
```

- **Dense(128, ReLU)**: 
  - Kết hợp 256 đặc trưng từ CNN thành biểu diễn phân loại
  - Học các tương tác giữa các pattern:
    - Ví dụ: Email vừa có "urgent" vừa có URL lạ → xác suất phishing cao hơn

- **Dense(64, ReLU)**: 
  - Tiếp tục kết hợp đặc trưng

- **Dense(1, Sigmoid)**: 
  - Trả về xác suất email là phishing (0-1)

### 3.3. Tổng số tham số

Mô hình CNN có khoảng **5.1 triệu tham số**, chủ yếu từ:
- Embedding: 20000 × 256 = 5,120,000
- Conv1D: ~40,000
- Dense layers: ~4,000

---

## 4. Quy Trình Hoạt Động của CNN để Nhận Diện Email

### 4.1. Quy trình tổng thể

Khi một email mới được đưa vào mô hình CNN, quy trình xử lý diễn ra như sau:

#### Bước 1: Tiền xử lý Text (ngoài mô hình)
```
Email gốc: "Your account has been compromised. Please verify immediately."
    ↓
TextVectorization (ngoài mô hình)
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

#### Bước 3: Conv1D phát hiện pattern

**Quét với kernel_size=5**:

```
Vị trí 1-5: ["Your", "account", "has", "been", "compromised"]
  → Filter 1: Kích hoạt = 0.8 (phát hiện "account has been compromised")
  → Filter 2: Kích hoạt = 0.3
  → Filter 3: Kích hoạt = 0.1
  → ...

Vị trí 2-6: ["account", "has", "been", "compromised", "Please"]
  → Filter 1: Kích hoạt = 0.9 (mạnh hơn)
  → Filter 2: Kích hoạt = 0.4
  → ...

Vị trí 6-10: ["Please", "verify", "immediately", ...]
  → Filter 5: Kích hoạt = 0.95 (phát hiện "Please verify")
  → Filter 6: Kích hoạt = 0.7
  → ...

... (quét toàn bộ 512 từ)

Kết quả: (512, 32) - mỗi vị trí có 32 giá trị kích hoạt
```

**Ví dụ cụ thể với các filter**:

```
Filter 1 (học pattern "verify your account"):
  Vị trí 10: 0.8
  Vị trí 50: 0.6
  Vị trí 200: 0.9 ← giá trị lớn nhất

Filter 2 (học pattern "click here"):
  Vị trí 15: 0.7
  Vị trí 100: 0.85 ← giá trị lớn nhất

Filter 3 (học pattern "urgent action"):
  Vị trí 5: 0.6
  Vị trí 80: 0.9 ← giá trị lớn nhất

Filter 4 (học pattern "meeting schedule"):
  Vị trí 20: 0.3 (yếu, không phải pattern chính)
  Vị trí 150: 0.2
```

#### Bước 4: GlobalMaxPool1D
```
(512, 32) - mỗi vị trí có 32 giá trị
    ↓
GlobalMaxPool1D (lấy max cho mỗi filter)
    ↓
(32,) - mỗi filter cho một giá trị lớn nhất
```

**Kết quả**:
```
Filter 1: 0.9 (pattern "verify your account" xuất hiện)
Filter 2: 0.85 (pattern "click here" xuất hiện)
Filter 3: 0.9 (pattern "urgent action" xuất hiện)
Filter 4: 0.3 (pattern "meeting schedule" yếu)
...
→ Vector 32 chiều: [0.9, 0.85, 0.9, 0.3, ...]
```

#### Bước 5: Dense Layers
```
Vector 32 chiều
    ↓
Dense(64, ReLU) → kết hợp các pattern
    ↓
Dense(1, Sigmoid) → xác suất phishing
```

**Ví dụ**:
```
Vector 32 chiều: [0.9, 0.85, 0.9, 0.3, ...]
  → Dense(64) học: "Nếu có nhiều pattern phishing (0.9, 0.85, 0.9) → xác suất cao"
  → Dense(1) → xác suất = 0.92
```

#### Bước 6: Quyết định phân loại
```
Xác suất = 0.92 (> 0.5)
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
3. Conv1D quét:
   - Pattern "Congratulations you won": Filter 1 kích hoạt = 0.95
   - Pattern "free iPhone": Filter 2 kích hoạt = 0.9
   - Pattern "Click here": Filter 3 kích hoạt = 0.9
   - Pattern "claim prize": Filter 4 kích hoạt = 0.85
4. GlobalMaxPool1D → [0.95, 0.9, 0.9, 0.85, ...] (32 chiều)
5. Dense layers → xác suất = 0.94
→ Phishing ✓
```

**Email 2: Benign**
```
Input: "Hi team, just a quick reminder about our meeting tomorrow at 10 AM."

Quá trình xử lý:
1. TextVectorization → Token IDs
2. Embedding → (512, 128)
3. Conv1D quét:
   - Pattern "meeting schedule": Filter 10 kích hoạt = 0.6
   - Pattern "team reminder": Filter 11 kích hoạt = 0.5
   - Không có pattern phishing mạnh
4. GlobalMaxPool1D → [0.1, 0.2, 0.1, ..., 0.6, 0.5, ...] (32 chiều)
5. Dense layers → xác suất = 0.12
→ Benign ✓
```

### 4.3. Đặc điểm CNN học được

CNN tự động học các pattern sau từ dữ liệu:

1. **Pattern từ vựng phishing (n-gram)**:
   - "verify your account"
   - "update payment information"
   - "account has been compromised"
   - "click here to"
   - "congratulations you won"
   - "urgent action required"

2. **Pattern từ vựng benign (n-gram)**:
   - "meeting schedule"
   - "project update"
   - "please find attached"
   - "as discussed"

3. **Sự kết hợp pattern**:
   - Nhiều pattern phishing cùng xuất hiện → xác suất phishing cao
   - Chủ yếu pattern benign → xác suất benign cao

---

## 4. Trích Xuất Đặc Trưng (Feature Extraction)

### 4.1. Quá trình trích xuất đặc trưng của CNN

CNN trích xuất đặc trưng qua các bước sau:

1. **Embedding Layer**: Chuyển token IDs thành vector embedding
   - Input: `[id_1, id_2, ..., id_512]` (token IDs)
   - Output: `(batch_size, 512, 128)` (mỗi token → vector 128 chiều)
   - **Ý nghĩa**: Tạo không gian đặc trưng ngữ nghĩa, từ có nghĩa tương tự → vector gần nhau

2. **Conv1D Layer**: Phát hiện pattern n-gram cục bộ
   - Input: `(batch_size, 512, 128)` (embedding vectors)
   - Filter quét cửa sổ 5 token (kernel_size=5) → phát hiện 5-gram
   - 32 filters → 32 pattern khác nhau được học
   - Output: `(batch_size, 508, 32)` (feature maps)
   - **Ý nghĩa**: Mỗi filter học một pattern cục bộ (ví dụ: "verify your account", "click here now")

3. **GlobalMaxPool1D**: Trích xuất đặc trưng quan trọng nhất
   - Input: `(batch_size, 508, 32)` (feature maps)
   - Lấy giá trị lớn nhất theo chiều sequence → giữ "tín hiệu mạnh nhất" của mỗi filter
   - Output: `(batch_size, 32)` (vector đặc trưng cuối cùng)
   - **Ý nghĩa**: "Email có chứa pattern này hay không?" (không quan tâm vị trí)

4. **Dense Layers**: Kết hợp đặc trưng để phân loại
   - Input: `(batch_size, 32)` (đặc trưng từ CNN)
   - Dense(64) → Kết hợp các pattern
   - Dense(1, sigmoid) → Xác suất phishing

### 4.2. Ví dụ minh họa

**Email**: "Your account has been suspended. Please verify your identity by clicking this link."

**Quá trình**:
1. TextVectorization → `[1234, 567, 890, 123, 456, ...]` (512 token IDs)
2. Embedding → `[[0.2, -0.1, 0.5, ...], [0.3, 0.1, -0.2, ...], ...]` (512 × 128)
3. Conv1D (kernel=5) → Phát hiện pattern "verify your identity by clicking"
   - Filter 1: Activation = 0.9 (pattern rất mạnh)
   - Filter 2: Activation = 0.7 (pattern khác)
   - ...
4. GlobalMaxPool → `[0.9, 0.7, 0.3, ...]` (32 giá trị lớn nhất)
5. Dense → Kết hợp: "Nhiều pattern phishing (0.9, 0.7, 0.8) → xác suất cao"
6. Output → Xác suất phishing = 0.92

### 4.3. Đặc điểm của đặc trưng CNN

- **Pattern cục bộ**: CNN giỏi phát hiện các cụm từ đặc trưng (n-gram)
- **Bất biến vị trí**: Pattern "click here" ở đầu hay cuối email đều được phát hiện
- **Tốc độ nhanh**: Xử lý song song, không cần đọc tuần tự như RNN
- **Hạn chế**: Không nắm bắt được phụ thuộc dài hạn (câu đầu ảnh hưởng đến câu cuối)

---

## 5. Quy Trình Xây Dựng Mô Hình CNN

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
- Dataset đã được chia sẵn thành `final_train.csv`, `final_val.csv`, `final_test.csv` (tỷ lệ 70-15-15)
- Trong training, có thể sử dụng trực tiếp `final_train.csv` và `final_val.csv`
- Hoặc chia tiếp từ `final_train.csv` để có train/val mới (như trong code dưới đây)

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.3,      # 30% validation từ train
    stratify=y,         # Giữ tỷ lệ phishing/benign đồng đều
    random_state=42     # Đảm bảo tái tạo kết quả
)
```

**Giải thích**:
- `stratify=y`: Đảm bảo tỷ lệ phishing/benign trong train và val giống nhau
- Điều này quan trọng để tránh bias trong quá trình training

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

### 5.3. Xây dựng mô hình CNN

```python
# Preprocess text (ngoài mô hình)
processed_train_text = text_vectorizer(X_train)

# Build model
inputs = layers.Input(shape=(None,), dtype="float32")
x = layers.Embedding(
    input_dim=len(words_in_vocab),
    output_dim=128,
    input_length=512
)(inputs)

x = layers.Conv1D(
    filters=32,
    kernel_size=5,
    activation="relu",
    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.L2(1e-4),
    activity_regularizer=regularizers.L2(1e-5)
)(x)

x = layers.Dropout(0.2)(x)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(
    64,
    activation="relu",
    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.L2(1e-4),
    activity_regularizer=regularizers.L2(1e-5)
)(x)

x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

cnn_model = tf.keras.Model(inputs, outputs, name="Conv1D_model")
```

**Lý do chọn tham số**:
- `filters=32`: Đủ để học nhiều pattern khác nhau
- `kernel_size=5`: Phát hiện 5-gram (cụm 5 từ)
- `GlobalMaxPool1D`: Phù hợp với bài toán "có chứa pattern hay không"

### 5.4. Compile mô hình

```python
cnn_model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)
```

### 5.5. Chuẩn bị callbacks

```python
# Model Checkpoint
cnn_model_checkpoint_callback = create_checkpoint_callback(
    file_path='outputs/models/CNN/checkpoints/cnn_checkpoint.keras'
)

# TensorBoard
cnn_model_tensorboard = create_tensorboard_callback(
    file_path='outputs/models/CNN/logs/cnn'
)

# Early Stopping
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',        # Theo dõi validation loss
    patience=5,                 # Dừng nếu val_loss không cải thiện sau 5 epochs
    restore_best_weights=True,  # Khôi phục weights tốt nhất khi dừng
    verbose=1                   # Hiển thị thông báo
)
```

**Giải thích Early Stopping**:
- **Monitor `val_loss`**: Theo dõi loss trên validation set (tổng quát hơn accuracy)
- **Patience=5**: Nếu val_loss không giảm trong 5 epochs liên tiếp → dừng training
- **Restore best weights**: Khôi phục weights của epoch có val_loss thấp nhất
- **Lý do**: Tránh overfitting, dừng khi model không còn học được gì mới
```

### 5.6. Huấn luyện mô hình

```python
cnn_history = cnn_model.fit(
    processed_train_text, y_train,
    validation_split=0.2,
    epochs=30,
    callbacks=[
        cnn_model_tensorboard,
        early_stopping_callback,
        cnn_model_checkpoint_callback
    ]
)
```

**Quá trình huấn luyện** (thực tế: 19 epochs):
- Epoch 1: Accuracy ~0.85, Loss ~0.40
- Epoch 2: Accuracy ~0.98, Loss ~0.05
- Epoch 3: Accuracy ~0.99, Loss ~0.02
- ...
- Epoch 19: Accuracy ~0.9967, Loss ~0.008 (Early stopping kích hoạt)

### 5.7. Đánh giá mô hình

```python
# Load test data
df_test = pd.read_csv('data/final/final_test.csv')
test_text = df_test['text']
test_labels = df_test['label']

# Preprocess
test_text_processed = test_text.fillna('').astype(str)
processed_test_text = text_vectorizer(test_text_processed)

# Predict
pred_labels_cnn = cnn_model.predict(processed_test_text)
avg_pred_labels_cnn = np.round(pred_labels_cnn.flatten())
```

### 5.8. Lưu mô hình và metrics

```python
# Save model
cnn_model.save('outputs/models/CNN/cnn_model.h5')

# Save metrics
cnn_metrics = {
    'model_config': {...},
    'test_evaluation': {...},
    'other_test_metrics': {...}
}
with open('outputs/models/CNN/cnn_metrics.json', 'w') as f:
    json.dump(cnn_metrics, f, indent=4)
```

---

## 6. Kết Quả của Mô Hình CNN

### 6.1. Kết quả trên tập test

Sau khi huấn luyện và đánh giá trên tập test (2,925 email), mô hình CNN đạt được các chỉ số sau:

| Metric | Giá trị |
|--------|---------|
| **Accuracy** | 0.9874 (98.74%) |
| **Precision** | 0.9875 (98.75%) |
| **Recall** | 0.9874 (98.74%) |
| **F1-Score** | 0.9873 (98.73%) |
| **AUC-ROC** | 0.9994 (99.94%) |

### 6.2. Kết quả trên tập validation

| Metric | Giá trị |
|--------|---------|
| **Accuracy** | 0.9867 (98.67%) |
| **Precision** | 0.9867 (98.67%) |
| **Recall** | 0.9867 (98.67%) |
| **F1-Score** | 0.9867 (98.67%) |
| **AUC-ROC** | 0.9991 (99.91%) |

### 6.3. Các chỉ số khác (Test)

| Metric | Giá trị |
|--------|---------|
| **False Discovery Rate (FDR)** | 0.0043 (0.43%) |
| **False Negative Rate (FNR)** | 0.0218 (2.18%) |
| **False Omission Rate (FOR)** | 0.0203 (2.03%) |
| **False Positive Rate (FPR)** | 0.004 (0.4%) |
| **Negative Prediction Value (NPV)** | 0.9797 (97.97%) |

### 6.4. Phân tích kết quả

**Điểm mạnh**:
- **Accuracy rất cao (98.74%)**: Mô hình phân loại cực kỳ chính xác
- **F1-Score rất cao (98.73%)**: Cân bằng tốt giữa Precision và Recall
- **FPR cực thấp (0.4%)**: Rất ít email benign bị nhận nhầm là phishing
- **FDR cực thấp (0.43%)**: Rất ít dự đoán phishing là sai
- **AUC-ROC gần hoàn hảo (99.94%)**: Khả năng phân biệt giữa phishing và benign rất tốt
- **Tốc độ nhanh**: CNN xử lý nhanh hơn RNN

**Điểm cần cải thiện**:
- **FNR cao hơn (2.18%)**: So với BERT và BiLSTM, CNN bỏ sót nhiều email phishing hơn một chút

### 6.5. Confusion Matrix (Test)

Dựa trên kết quả test với 2,925 email:
- **True Positive (TP)**: ~1,432 email phishing được phát hiện đúng
- **True Negative (TN)**: ~1,456 email benign được phân loại đúng
- **False Positive (FP)**: ~6 email benign bị nhận nhầm là phishing (FPR = 0.4%, rất thấp!)
- **False Negative (FN)**: ~31 email phishing bị bỏ sót (FNR = 2.18%)

### 6.6. So sánh với các mô hình khác

| Mô hình | Accuracy (Test) | F1-Score | FNR | AUC-ROC | Tốc độ |
|---------|----------------|----------|-----|---------|--------|
| CNN | 98.74% | 98.73% | 2.18% | 99.94% | Nhanh nhất |
| BERT | 99.01% | 99.01% | 0.98% | 99.92% | Chậm nhất |
| BiLSTM | 98.87% | 98.87% | 1.05% | 99.95% | Trung bình |
| CNN-BiLSTM | 98.63% | 98.63% | 1.61% | 99.89% | Trung bình |
| GRU | 98.56% | 98.56% | 1.61% | 99.76% | Trung bình |

**Nhận xét**: CNN đạt kết quả rất tốt (98.74%) và nhanh nhất, phù hợp cho ứng dụng cần xử lý nhiều email cùng lúc. FPR cực thấp (0.4%) là điểm mạnh của CNN.

### 6.6. Ví dụ dự đoán

**Email được phát hiện đúng là Phishing**:
```
Input: "Congratulations! You've won a free iPhone. Click here to claim your prize!"
Prediction: Phishing (xác suất: 0.94) ✓
→ CNN phát hiện các pattern: "congratulations won", "free iPhone", "click here"
```

**Email được phát hiện đúng là Benign**:
```
Input: "Hi team, just a quick reminder about our meeting tomorrow at 10 AM."
Prediction: Benign (xác suất: 0.12) ✓
→ CNN phát hiện pattern: "meeting reminder", không có pattern phishing
```

**Email bị nhận nhầm (False Positive)**:
```
Input: "Limited time offer: Get 50% off all products today!"
Prediction: Phishing (xác suất: 0.65) ✗
→ CNN phát hiện pattern "limited time offer" giống phishing, nhưng thực tế là marketing hợp pháp
```

---

## 7. Giải Thích Dự Đoán (Explainable AI - XAI)

### 7.1. Cách Model "Biết" Từ Nào Quan Trọng

**Quan trọng**: Model không tự động "gắn nhãn từng từ" là phishing hay benign. Model chỉ output xác suất tổng thể. Để biết "từ nào quan trọng", project sử dụng **LIME (Local Interpretable Model-Agnostic Explanations)**.

### 7.2. Cơ Chế LIME

**LIME hoạt động như sau**:

1. **Tạo nhiều biến thể của email**:
   - LIME loại bỏ/ẩn một số từ trong email gốc
   - Tạo ra hàng nghìn phiên bản email khác nhau
   - Ví dụ: "Your account has been suspended. Please verify..." 
     → "Your account [MASK] suspended. [MASK] verify..."
     → "[MASK] account has been suspended. Please [MASK]..."

2. **Gọi model dự đoán** cho từng biến thể:
   - Mỗi email biến thể → xác suất phishing
   - So sánh với xác suất của email gốc

3. **Fit mô hình tuyến tính cục bộ**:
   - LIME fit một mô hình tuyến tính đơn giản để giải thích dự đoán
   - Mô hình này học: "Từ nào khi bị loại bỏ làm xác suất thay đổi nhiều nhất?"
   - Từ làm xác suất phishing giảm → từ đó quan trọng cho phishing
   - Từ làm xác suất phishing tăng → từ đó quan trọng cho benign

4. **Trả về trọng số từng từ**:
   - Mỗi từ được gán một trọng số (weight)
   - Trọng số dương → từ làm tăng xác suất phishing
   - Trọng số âm → từ làm giảm xác suất phishing (tăng benign)

### 7.3. Ví Dụ Giải Thích LIME

**Email**: "Your account has been suspended. Please verify your identity immediately."

**Kết quả LIME** (top 5 từ quan trọng):
```
1. "verify"      → weight: +0.123  (tăng xác suất phishing)
2. "suspended"   → weight: +0.098  (tăng xác suất phishing)
3. "immediately" → weight: +0.087  (tăng xác suất phishing)
4. "account"    → weight: +0.065  (tăng xác suất phishing)
5. "please"     → weight: -0.032  (giảm xác suất phishing, tăng benign)
```

**Giải thích**: 
- Các từ "verify", "suspended", "immediately" là dấu hiệu mạnh của phishing
- Từ "please" có thể xuất hiện trong email hợp pháp → trọng số âm

### 7.4. Hiển Thị Trong Extension

Extension sử dụng kết quả LIME để:
- **Highlight các từ quan trọng** trong email preview
- Màu đỏ → từ làm tăng xác suất phishing
- Màu xanh → từ làm tăng xác suất benign
- Độ đậm màu phụ thuộc vào độ lớn của trọng số

**Code tham khảo** (trong `notebooks/XAI/lime_explainer.py`):
```python
explanation = self.explainer.explain_instance(
    email_text,
    predict_fn,
    num_features=15,    # Top 15 từ quan trọng
    num_samples=5000    # Tạo 5000 biến thể email
)
```

---

## 8. Kết Luận

Mô hình CNN đã được áp dụng thành công trong bài toán phát hiện email phishing với các đặc điểm:

1. **Hiệu quả rất cao**: Đạt accuracy 98.74% trên test set, cải thiện đáng kể so với phiên bản trước
2. **FPR cực thấp (0.4%)**: Rất ít email benign bị nhận nhầm - điểm mạnh của CNN
3. **AUC-ROC gần hoàn hảo (99.94%)**: Khả năng phân biệt giữa phishing và benign rất tốt
4. **Tốc độ**: Nhanh nhất trong các mô hình, phù hợp với ứng dụng thời gian thực
5. **Đơn giản**: Kiến trúc đơn giản, dễ hiểu và triển khai
6. **Phát hiện pattern**: Tự động học các pattern n-gram đặc trưng của phishing

CNN phù hợp cho các ứng dụng cần cân bằng giữa độ chính xác cao (98.74%) và tốc độ xử lý nhanh, đặc biệt khi cần xử lý số lượng lớn email trong thời gian ngắn. FPR cực thấp làm cho CNN trở thành lựa chọn tốt khi cần tránh cảnh báo sai.

