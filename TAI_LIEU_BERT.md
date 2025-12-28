# Tài Liệu Mô Hình BERT (Bidirectional Encoder Representations from Transformers)

## 1. Khái Niệm Cơ Bản về BERT

### 1.1. BERT là gì?

**BERT (Bidirectional Encoder Representations from Transformers)** là một mô hình ngôn ngữ lớn (large language model) được phát triển bởi Google vào năm 2018. BERT sử dụng kiến trúc **Transformer** với cơ chế **Attention** để hiểu ngữ nghĩa của văn bản theo cả hai chiều (bidirectional).

### 1.2. Đặc điểm chính của BERT

- **Bidirectional Context**: 
  - BERT đọc và hiểu văn bản theo cả hai chiều (trái → phải và phải → trái) đồng thời
  - Khác với các mô hình RNN chỉ đọc một chiều

- **Pre-trained trên dữ liệu lớn**: 
  - BERT được huấn luyện trước trên hàng tỷ từ từ Wikipedia và sách
  - Đã học được kiến thức ngôn ngữ phong phú

- **Transfer Learning**: 
  - Có thể fine-tune cho các tác vụ cụ thể (như phân loại email)
  - Chỉ cần huấn luyện thêm một vài lớp trên đầu mô hình pre-trained

- **Self-Attention Mechanism**: 
  - Tự động học mối quan hệ giữa các từ trong câu
  - Hiểu được ngữ cảnh đầy đủ của mỗi từ

### 1.3. Kiến trúc BERT

BERT sử dụng kiến trúc **Transformer Encoder**:

1. **Input Embedding**: 
   - Token Embedding: Mã hóa từ thành vector
   - Segment Embedding: Phân biệt câu đầu và câu thứ hai (nếu có)
   - Position Embedding: Mã hóa vị trí của từ trong câu

2. **Transformer Encoder Layers** (12 layers cho BERT-base):
   - **Multi-Head Self-Attention**: Học mối quan hệ giữa các từ
   - **Feed-Forward Network**: Xử lý thông tin
   - **Layer Normalization**: Ổn định quá trình huấn luyện
   - **Residual Connection**: Giúp gradient lan truyền tốt hơn

3. **Output**: 
   - [CLS] token: Đại diện cho toàn bộ câu (dùng cho phân loại)
   - Token embeddings: Đại diện cho từng từ

**Công thức Self-Attention**:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Trong đó:
- Q (Query): Câu hỏi "từ này liên quan đến từ nào?"
- K (Key): "Từ này có thể trả lời câu hỏi gì?"
- V (Value): "Thông tin thực tế của từ này"
- d_k: Chiều của key vector
```

### 1.4. So sánh với các mô hình khác

| Đặc điểm | BERT | RNN/LSTM | CNN |
|---------|------|-----------|-----|
| Hướng xử lý | Bidirectional (đồng thời) | Sequential (tuần tự) | Local (cục bộ) |
| Pre-trained | Có (kiến thức phong phú) | Không | Không |
| Hiểu ngữ cảnh | Rất tốt (self-attention) | Tốt (hidden state) | Yếu (chỉ pattern cục bộ) |
| Tốc độ | Chậm (nhiều tham số) | Trung bình | Nhanh |
| Số tham số | ~110M (BERT-base) | ~5M | ~2.5M |

---

## 2. Ứng Dụng của BERT trong Project Phát Hiện Email Phishing

### 2.1. Bài toán

Trong project này, BERT được sử dụng để giải quyết bài toán **phân loại nhị phân email**:
- **Nhãn 0**: Benign (email bình thường, hợp pháp)
- **Nhãn 1**: Phishing (email lừa đảo, giả mạo)

### 2.2. Tại sao chọn BERT?

1. **Hiểu ngữ nghĩa sâu sắc**: 
   - BERT đã được pre-trained trên dữ liệu lớn, hiểu rõ ngữ nghĩa tiếng Anh
   - Có thể phân biệt các sắc thái ngữ nghĩa tinh tế

2. **Bidirectional Context**: 
   - Hiểu ngữ cảnh đầy đủ của mỗi từ
   - Phân biệt được "Do NOT click" vs "Click here"

3. **Self-Attention**: 
   - Tự động học mối quan hệ giữa các từ xa nhau trong email
   - Hiểu được cấu trúc và logic của email

4. **Transfer Learning**: 
   - Tận dụng kiến thức đã học từ dữ liệu lớn
   - Chỉ cần fine-tune cho tác vụ cụ thể

5. **Hiệu quả cao**: 
   - Thường đạt kết quả tốt nhất trong các bài toán NLP
   - Phù hợp với bài toán phát hiện email phishing

### 2.3. Dữ liệu sử dụng

- **Tập train**: `final_train.csv` với ~13,648 email (sau khi xử lý mất cân bằng)
- **Tập validation**: `final_val.csv` với ~2,925 email (hoặc chia 30% từ train)
- **Tập test**: `final_test.csv` với ~2,925 email
- **Chia dữ liệu**: 70% train, 15% validation, 15% test (stratified split)

### 2.4. BERT Model sử dụng

- **Model**: `bert-base-uncased`
- **Số layers**: 12
- **Hidden size**: 768
- **Số attention heads**: 12
- **Tổng số tham số**: ~110 triệu

---

## 3. Kiến Trúc Mô Hình BERT

### 3.1. Tổng quan kiến trúc

```
Input Text
    ↓
BERT Tokenizer (WordPiece)
    ↓
Token IDs + Attention Mask
    ↓
BERT Embeddings
    (Token Embedding + Position Embedding + Segment Embedding)
    ↓
12 × Transformer Encoder Layers
    (Multi-Head Self-Attention + Feed-Forward)
    ↓
[CLS] Token Representation (768 dimensions)
    ↓
Classification Head (Linear Layer)
    ↓
Output (2 classes: Benign, Phishing)
```

### 3.2. Chi tiết từng phần

#### 3.2.1. BERT Tokenizer

```python
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
```

**Cơ chế hoạt động**:

1. **WordPiece Tokenization**:
   - Chia từ thành các subword tokens
   - Ví dụ: "unhappiness" → ["un", "##happiness"]
   - Giúp xử lý từ mới (out-of-vocabulary)

2. **Special Tokens**:
   - `[CLS]`: Token đặc biệt ở đầu câu, dùng cho phân loại
   - `[SEP]`: Token phân cách giữa các câu
   - `[PAD]`: Token padding để đảm bảo độ dài cố định
   - `[UNK]`: Token cho từ không biết

3. **Token IDs và Attention Mask**:
   - Token IDs: Mã hóa từ thành số
   - Attention Mask: Đánh dấu vị trí thực (1) và padding (0)

**Ví dụ**:
```
Input: "Your account has been compromised."

Tokenizer:
  Tokens: ["[CLS]", "your", "account", "has", "been", "compromised", ".", "[SEP]"]
  Token IDs: [101, 2115, 2175, 2038, 2042, 5458, 1012, 102]
  Attention Mask: [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ..., 0] (pad đến 512)
```

#### 3.2.2. BERT Embeddings

BERT sử dụng 3 loại embedding:

1. **Token Embedding**: 
   - Mã hóa từ thành vector 768 chiều
   - Đã được pre-trained

2. **Position Embedding**: 
   - Mã hóa vị trí của từ trong câu (0-511)
   - Giúp BERT hiểu thứ tự

3. **Segment Embedding**: 
   - Phân biệt câu đầu (0) và câu thứ hai (1)
   - Trong bài toán này, chỉ có một câu nên tất cả là 0

**Kết quả**: Vector 768 chiều cho mỗi token

#### 3.2.3. Transformer Encoder Layers (12 layers)

Mỗi layer bao gồm:

1. **Multi-Head Self-Attention**:
   - 12 attention heads, mỗi head có 64 chiều
   - Học mối quan hệ giữa các từ
   - Ví dụ: Từ "compromised" chú ý đến "account", "verify", "link"

2. **Feed-Forward Network**:
   - 2 lớp Dense: 768 → 3072 → 768
   - Xử lý thông tin từ attention

3. **Layer Normalization + Residual Connection**:
   - Ổn định quá trình huấn luyện
   - Giúp gradient lan truyền tốt

**Ví dụ minh họa Self-Attention**:

```
Email: "Your account has been compromised. Please verify immediately."

Layer 1 (Attention):
  "compromised" chú ý đến: "account" (0.4), "verify" (0.3), "immediately" (0.2), ...
  
Layer 2 (Attention):
  "compromised" chú ý đến: "account" (0.5), "verify" (0.4), "immediately" (0.1), ...
  (Tích lũy thêm thông tin)

...

Layer 12 (Attention):
  "compromised" có đầy đủ ngữ cảnh về toàn bộ email
  → Representation phong phú, hiểu rõ ý nghĩa trong ngữ cảnh
```

#### 3.2.4. Input Processing

**Cách Model Nhận Input**:

1. **Input ban đầu**: Email text dạng string
   ```python
   email_text = "Your account has been compromised. Please verify..."
   ```

2. **BERT Tokenizer** (WordPiece):
   ```python
   tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
   inputs = tokenizer(
       email_text,
       padding=True,
       truncation=True,
       max_length=512,
       return_tensors="pt"
   )
   ```
   - Email text → WordPiece tokens → Token IDs
   - Tạo `input_ids` và `attention_mask`
   - Kết quả: `input_ids` shape `(1, 512)`, `attention_mask` shape `(1, 512)`

3. **BERT Embeddings**:
   - Token Embedding + Position Embedding + Segment Embedding
   - Output: `(batch_size, 512, 768)` (mỗi token → vector 768 chiều)

#### 3.2.5. [CLS] Token Representation

```python
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
```

- **[CLS] token**: Token đặc biệt ở đầu câu
- Sau 12 layers, [CLS] token chứa thông tin tổng hợp về toàn bộ email
- Kích thước: 768 chiều

#### 3.2.5. Classification Head

```python
# Trong BertForSequenceClassification
classifier = nn.Linear(768, 2)  # 2 classes: Benign, Phishing
```

- **Linear Layer**: 768 → 2
- **Output**: Logits cho 2 lớp (Benign, Phishing)
- **Softmax**: Chuyển logits thành xác suất

### 3.3. Tổng số tham số

Mô hình BERT-base có khoảng **110 triệu tham số**, chủ yếu từ:
- Embeddings: ~24M
- 12 Transformer layers: ~85M
- Classification head: ~1.5K

---

## 4. Trích Xuất Đặc Trưng (Feature Extraction)

### 4.1. Quá trình trích xuất đặc trưng của BERT

BERT trích xuất đặc trưng qua các bước sau:

1. **BERT Tokenizer (WordPiece)**: Chia text thành subword tokens
   - Input: Email text dạng string
   - Output: Token IDs `[id_1, id_2, ..., id_512]` + Attention Mask
   - **Đặc điểm**: WordPiece giúp xử lý từ mới (out-of-vocabulary)

2. **BERT Embeddings**: Tạo embedding đa chiều
   - Token Embedding: Mã hóa từ thành vector 768 chiều
   - Position Embedding: Mã hóa vị trí của từ
   - Segment Embedding: Phân biệt câu (trong bài toán này tất cả là 0)
   - Output: `(batch_size, 512, 768)` (mỗi token → vector 768 chiều)

3. **12 Transformer Encoder Layers**: Xử lý qua self-attention
   - **Multi-Head Self-Attention**: Mỗi token "nhìn" vào tất cả các token khác
     - 12 attention heads, mỗi head học mối quan hệ khác nhau
     - Ví dụ: "compromised" chú ý đến "account", "verify", "immediately"
   - **Feed-Forward Network**: Xử lý thông tin từ attention
   - **Layer Normalization + Residual**: Ổn định training
   - Sau mỗi layer, representation của mỗi token được cập nhật với thông tin từ các token khác
   - Output sau 12 layers: `(batch_size, 512, 768)` (mỗi token có representation phong phú)

4. **[CLS] Token Representation**: Tóm tắt toàn bộ email
   - [CLS] token ở đầu câu, sau 12 layers chứa thông tin tổng hợp
   - Output: `(batch_size, 768)` (vector 768 chiều)

5. **Classification Head**: Phân loại
   - Linear(768 → 2) → Logits cho 2 lớp
   - Softmax → Xác suất Benign/Phishing

### 4.2. Đặc điểm của đặc trưng BERT

- **Contextual Embeddings**: Mỗi token có representation phụ thuộc vào ngữ cảnh
  - Ví dụ: "bank" trong "river bank" vs "bank account" có representation khác nhau
- **Bidirectional**: Hiểu ngữ cảnh cả hai chiều đồng thời (không như RNN)
- **Self-Attention**: Tự động học mối quan hệ giữa các từ xa nhau
- **Pre-trained**: Đã học kiến thức ngôn ngữ phong phú từ dữ liệu lớn

---

## 5. Quy Trình Hoạt Động của BERT để Nhận Diện Email

### 4.1. Quy trình tổng thể

Khi một email mới được đưa vào mô hình BERT, quy trình xử lý diễn ra như sau:

#### Bước 1: Tokenization
```
Email gốc: "Your account has been compromised. Please verify immediately."
    ↓
BERT Tokenizer (WordPiece)
    ↓
Tokens: ["[CLS]", "your", "account", "has", "been", "compromised", ".", "please", "verify", "immediately", ".", "[SEP]"]
    ↓
Token IDs: [101, 2115, 2175, 2038, 2042, 5458, 1012, 3533, 4567, 3456, 1012, 102, 0, 0, ..., 0] (pad đến 512)
Attention Mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ..., 0]
```

#### Bước 2: Embeddings
```
Token IDs
    ↓
BERT Embeddings (Token + Position + Segment)
    ↓
Embedding Vectors: (512, 768) - mỗi token là vector 768 chiều
```

#### Bước 3: Transformer Encoder (12 layers)

**Layer 1 - Self-Attention**:
```
Mỗi token "nhìn" vào tất cả các token khác:
  "compromised" chú ý đến:
    - "account" (attention weight: 0.4)
    - "verify" (attention weight: 0.3)
    - "immediately" (attention weight: 0.2)
    - ...
  → Tích lũy thông tin từ các từ liên quan
```

**Layer 2-12**:
```
Tiếp tục tích lũy thông tin qua các layers:
  Layer 2: "compromised" hiểu rõ hơn về ngữ cảnh
  Layer 3: Tích lũy thêm thông tin về cấu trúc email
  ...
  Layer 12: "compromised" có đầy đủ ngữ cảnh về toàn bộ email
```

**Kết quả sau 12 layers**:
```
Mỗi token có representation 768 chiều, chứa thông tin về:
  - Bản thân token đó
  - Các token liên quan (qua attention)
  - Ngữ cảnh tổng thể của email
```

#### Bước 4: [CLS] Token Representation
```
[CLS] token sau 12 layers: (768,)
  → Chứa thông tin tổng hợp về toàn bộ email
  → Đại diện cho ngữ nghĩa và cấu trúc của email
```

#### Bước 5: Classification Head
```
[CLS] token (768,)
    ↓
Linear Layer (768 → 2)
    ↓
Logits: [2.5, -1.3] (logits cho Benign và Phishing)
    ↓
Softmax
    ↓
Probabilities: [0.97, 0.03] (97% Benign, 3% Phishing)
```

#### Bước 6: Quyết định phân loại
```
Probabilities: [0.03, 0.97] (3% Benign, 97% Phishing)
    ↓
Argmax → Class 1 (Phishing)
    ↓
Kết luận: Phishing (Nhãn 1)
```

### 4.2. Ví dụ cụ thể

**Email 1: Phishing**
```
Input: "Your bank account has been compromised. Please verify your details immediately at this link: http://malicious-site.com/verify"

Quá trình xử lý:
1. Tokenization:
   ["[CLS]", "your", "bank", "account", "has", "been", "compromised", ...]
   
2. Embeddings: (512, 768)

3. Self-Attention (12 layers):
   - "compromised" chú ý đến: "account" (0.5), "verify" (0.4), "link" (0.3)
   - "verify" chú ý đến: "compromised" (0.4), "link" (0.5), "immediately" (0.3)
   - "link" chú ý đến: "verify" (0.6), "compromised" (0.3), "http" (0.4)
   - ... (tích lũy qua 12 layers)

4. [CLS] token: (768,) - tổng hợp toàn bộ thông tin
   → Hiểu: Email có cấu trúc phishing điển hình
   → Pattern: Cảnh báo + Yêu cầu hành động + Link

5. Classification:
   → Logits: [-1.2, 3.5]
   → Probabilities: [0.05, 0.95]
   → Phishing ✓
```

**Email 2: Benign**
```
Input: "Hi team, just a quick reminder about our meeting tomorrow at 10 AM. Please prepare your reports."

Quá trình xử lý:
1. Tokenization:
   ["[CLS]", "hi", "team", "just", "a", "quick", "reminder", ...]

2. Embeddings: (512, 768)

3. Self-Attention (12 layers):
   - "meeting" chú ý đến: "team" (0.4), "reminder" (0.3), "tomorrow" (0.3)
   - "reports" chú ý đến: "meeting" (0.4), "prepare" (0.3)
   - ... (tích lũy qua 12 layers)

4. [CLS] token: (768,)
   → Hiểu: Email công việc, thân thiện
   → Pattern: Thông tin, nhắc nhở hợp lý

5. Classification:
   → Logits: [2.8, -1.5]
   → Probabilities: [0.98, 0.02]
   → Benign ✓
```

### 4.3. Đặc điểm BERT học được

1. **Ngữ nghĩa sâu sắc**:
   - Hiểu rõ nghĩa của từ trong ngữ cảnh
   - Phân biệt "verify" trong email cảnh báo vs email phishing

2. **Mối quan hệ giữa các từ**:
   - Tự động học mối quan hệ xa nhau
   - Ví dụ: "account" và "compromised" có thể cách xa nhưng BERT vẫn hiểu mối liên hệ

3. **Cấu trúc email**:
   - Hiểu cấu trúc tổng thể của email
   - Phân biệt cấu trúc phishing vs benign

4. **Pattern phức tạp**:
   - Phát hiện pattern không chỉ dựa trên từ vựng mà còn dựa trên ngữ nghĩa và cấu trúc

---

## 6. Quy Trình Xây Dựng Mô Hình BERT

### 5.1. Chuẩn bị dữ liệu

#### Bước 1: Load dữ liệu
```python
df_train = pd.read_csv('data/final/final_train.csv')
X = df_train['text'].fillna("").astype(str)
y = df_train['label'].astype(int)
```

#### Bước 2: Chia train/validation

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

### 5.2. Xây dựng Dataset Class

```python
class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item
```

### 5.3. Load BERT Model và Tokenizer

```python
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)

model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  # Benign, Phishing
)
```

### 5.4. Tạo Dataset

```python
train_dataset = EmailDataset(X_train, y_train, tokenizer, max_length=512)
val_dataset = EmailDataset(X_val, y_val, tokenizer, max_length=512)
```

### 5.5. Cấu hình Training Arguments

```python
training_args = TrainingArguments(
    output_dir='outputs/models/BERT/checkpoints/bert_base',
    
    # Evaluation
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    
    # Training configuration
    learning_rate=2e-5,  # Learning rate nhỏ cho fine-tuning
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=15,
    weight_decay=0.01,
    
    # Model selection
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="tensorboard"
)
```

**Lý do chọn tham số**:
- `learning_rate=2e-5`: Learning rate nhỏ cho fine-tuning (không làm hỏng pre-trained weights)
- `batch_size=8`: Nhỏ do BERT có nhiều tham số
- `num_train_epochs=15`: Đủ để fine-tune mà không overfit

### 5.6. Tạo Trainer

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    metrics = calculate_results(labels, preds)
    return metrics

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,  # Dừng nếu không cải thiện sau 3 epochs
            early_stopping_threshold=0.001  # Ngưỡng cải thiện tối thiểu
        )
    ],
)
```

### 5.7. Huấn luyện mô hình

```python
train_result = trainer.train()
```

**Quá trình huấn luyện** (thực tế: 6 epochs):
- Epoch 1: Training Loss ~0.3, Validation F1 ~0.95
- Epoch 2: Training Loss ~0.1, Validation F1 ~0.98
- Epoch 3: Training Loss ~0.05, Validation F1 ~0.99
- ...
- Epoch 6: Training Loss ~0.01, Validation F1 ~0.9935 (Early stopping kích hoạt)

### 5.8. Đánh giá mô hình

```python
# Load test data
df_test = pd.read_csv('data/final/final_test.csv')
test_text = df_test['text'].fillna("").astype(str)
test_labels = df_test['label'].astype(int).values

# Create test dataset
test_dataset = EmailDataset(test_text, test_labels, tokenizer, max_length=512)

# Predict
pred_output = trainer.predict(test_dataset)
test_logits = pred_output.predictions
test_pred_labels = np.argmax(test_logits, axis=-1)
```

### 5.9. Lưu mô hình và metrics

```python
# Save model
model.save_pretrained('outputs/models/BERT/bert_base_email_model')
tokenizer.save_pretrained('outputs/models/BERT/bert_base_email_model')

# Save metrics
bert_metrics = {
    'model_config': {...},
    'test_evaluation': {...},
    'other_test_metrics': {...}
}
with open('outputs/models/BERT/bert_base_metrics.json', 'w') as f:
    json.dump(bert_metrics, f, indent=4)
```

---

## 7. Giải Thích Dự Đoán (Explainable AI - XAI)

### 7.1. Cách Model "Biết" Từ Nào Quan Trọng

BERT sử dụng **SHAP (SHapley Additive exPlanations)** với cơ chế **masking token** để giải thích dự đoán. Khác với LIME (dùng cho Keras models), SHAP cho BERT hoạt động bằng cách:

1. **Tokenize email** thành WordPiece tokens
2. **Mask từng token** bằng `[MASK]` token
3. **Gọi model dự đoán** cho mỗi phiên bản đã mask
4. **Đo thay đổi logit/probability**:
   - Baseline: Xác suất phishing của email gốc
   - Masked: Xác suất phishing khi mask token đó
   - SHAP value = Baseline - Masked (delta)
5. **Trả về trọng số từng token**:
   - Token có SHAP value dương cao → quan trọng cho phishing
   - Token có SHAP value âm → quan trọng cho benign

### 7.2. Ví Dụ Giải Thích SHAP cho BERT

**Email**: "Your account has been compromised. Please verify your identity immediately."

**Kết quả SHAP** (top 5 token quan trọng):
```
1. "compromised"  → weight: +0.234  (tăng xác suất phishing mạnh)
2. "verify"       → weight: +0.187  (tăng xác suất phishing)
3. "immediately"  → weight: +0.156  (tăng xác suất phishing)
4. "account"      → weight: +0.098  (tăng xác suất phishing)
5. "your"         → weight: -0.032  (giảm xác suất phishing nhẹ)
```

**Giải thích**: 
- Các token "compromised", "verify", "immediately" là dấu hiệu mạnh của phishing
- Token "your" có thể xuất hiện trong email hợp pháp → trọng số âm nhẹ

### 7.3. So Sánh SHAP vs LIME

| Đặc điểm | SHAP (BERT) | LIME (Keras) |
|---------|-------------|--------------|
| Cơ chế | Mask token và đo delta | Tạo biến thể email và fit mô hình tuyến tính |
| Tốc độ | Nhanh hơn (batch processing) | Chậm hơn (nhiều samples) |
| Độ chính xác | Cao (trực tiếp từ model) | Tốt (xấp xỉ cục bộ) |
| Phù hợp | Transformer models | Mọi model (model-agnostic) |

**Code tham khảo** (trong `notebooks/XAI/shap_explainer.py`):
```python
# Mask từng token và đo thay đổi
for token in bert_tokens:
    masked_input_ids = original_input_ids.clone()
    masked_input_ids[0][pos] = mask_token_id
    ...
    delta_logit = baseline_logit_phishing - masked_logit_phishing
    shap_value = delta_logit if abs(delta_logit) > 0.01 else delta_prob * 1000
```

---

## 8. Kết Quả của Mô Hình BERT

### 8.1. Kết quả trên tập test

Sau khi huấn luyện và đánh giá trên tập test (2,925 email), mô hình BERT đạt được các chỉ số sau:

| Metric | Giá trị |
|--------|---------|
| **Accuracy** | 0.9901 (99.01%) |
| **Precision** | 0.9901 (99.01%) |
| **Recall** | 0.9901 (99.01%) |
| **F1-Score** | 0.9901 (99.01%) |
| **AUC-ROC** | 0.9992 (99.92%) |

### 8.2. Kết quả trên tập validation

| Metric | Giá trị |
|--------|---------|
| **Accuracy** | 0.9935 (99.35%) |
| **Precision** | 0.9935 (99.35%) |
| **Recall** | 0.9935 (99.35%) |
| **F1-Score** | 0.9935 (99.35%) |
| **AUC-ROC** | 0.9997 (99.97%) |

### 8.3. Các chỉ số khác (Test)

| Metric | Giá trị |
|--------|---------|
| **False Discovery Rate (FDR)** | 0.0105 (1.05%) |
| **False Negative Rate (FNR)** | 0.0098 (0.98%) |
| **False Omission Rate (FOR)** | 0.0093 (0.93%) |
| **False Positive Rate (FPR)** | 0.01 (1.0%) |
| **Negative Prediction Value (NPV)** | 0.9907 (99.07%) |

### 8.4. Phân tích kết quả

**Điểm mạnh**:
- **Accuracy rất cao (99.01%)**: Mô hình phân loại cực kỳ chính xác
- **F1-Score rất cao (99.01%)**: Cân bằng hoàn hảo giữa Precision và Recall
- **FNR cực thấp (0.98%)**: Rất ít email phishing bị bỏ sót (quan trọng trong bảo mật)
- **FPR rất thấp (1.0%)**: Rất ít email benign bị nhận nhầm
- **NPV rất cao (99.07%)**: Khi mô hình dự đoán Benign, khả năng đúng rất cao
- **AUC-ROC gần hoàn hảo (99.92%)**: Khả năng phân biệt giữa phishing và benign rất tốt

**So sánh với các mô hình khác**:

| Mô hình | Accuracy (Test) | F1-Score | FNR | AUC-ROC | Tốc độ |
|---------|------------------|----------|-----|---------|--------|
| BERT | 99.01% | 99.01% | 0.98% | 99.92% | Chậm nhất |
| BiLSTM | 98.87% | 98.87% | 1.05% | 99.95% | Trung bình |
| CNN-BiLSTM | 98.63% | 98.63% | 1.61% | 99.89% | Trung bình |
| CNN | 98.74% | 98.73% | 2.18% | 99.94% | Nhanh nhất |
| GRU | 98.56% | 98.56% | 1.61% | 99.76% | Trung bình |

**Nhận xét**: 
- BERT đạt kết quả tốt nhất (99.01%), vượt trội so với các mô hình khác
- FNR cực thấp (0.98%) - rất quan trọng trong bảo mật
- AUC-ROC gần hoàn hảo, chứng tỏ khả năng phân loại xuất sắc
- Tốc độ chậm nhất do số tham số lớn (~110M)

### 8.5. Confusion Matrix (Test)

Dựa trên kết quả test với 2,925 email:
- **True Positive (TP)**: ~1,448 email phishing được phát hiện đúng
- **True Negative (TN)**: ~1,448 email benign được phân loại đúng
- **False Positive (FP)**: ~15 email benign bị nhận nhầm là phishing (FPR = 1.0%)
- **False Negative (FN)**: ~14 email phishing bị bỏ sót (FNR = 0.98%, rất ít!)

### 6.5. Ví dụ dự đoán

**Email được phát hiện đúng là Phishing**:
```
Input: "Your bank account has been compromised. Please verify your details immediately at this link: http://malicious-site.com/verify"
Prediction: Phishing (xác suất: 0.97) ✓
→ BERT hiểu: Cấu trúc phishing điển hình, ngữ cảnh đáng ngờ
```

**Email được phát hiện đúng là Benign**:
```
Input: "Hi team, just a quick reminder about our meeting tomorrow at 10 AM. Please prepare your reports."
Prediction: Benign (xác suất: 0.98) ✓
→ BERT hiểu: Ngữ cảnh công việc, thân thiện, hợp lý
```

**Email có ngữ cảnh phủ định (phát hiện đúng)**:
```
Input: "This is an example of a phishing attempt. Do NOT click any suspicious links."
Prediction: Benign (xác suất: 0.95) ✓
→ BERT hiểu: "Do NOT" phủ định, "example" chỉ là ví dụ, không phải phishing thật
```

---

## 7. Kết Luận

Mô hình BERT đã được áp dụng thành công trong bài toán phát hiện email phishing với các đặc điểm:

1. **Hiệu quả xuất sắc**: Đạt accuracy 99.01% trên test set, tốt nhất trong tất cả các mô hình
2. **FNR cực thấp (0.98%)**: Rất ít email phishing bị bỏ sót - rất quan trọng trong bảo mật
3. **AUC-ROC gần hoàn hảo (99.92%)**: Khả năng phân biệt giữa phishing và benign rất xuất sắc
4. **Hiểu ngữ nghĩa sâu sắc**: Nhờ pre-training trên dữ liệu lớn, BERT hiểu rõ ngữ nghĩa và ngữ cảnh
5. **Self-Attention**: Tự động học mối quan hệ giữa các từ, hiểu cấu trúc email
6. **Transfer Learning**: Tận dụng kiến thức đã học, chỉ cần fine-tune 6 epochs cho tác vụ cụ thể

BERT phù hợp cho các ứng dụng cần độ chính xác cao nhất và có đủ tài nguyên tính toán. Đặc biệt, FNR cực thấp (0.98%) và accuracy cao nhất (99.01%) làm cho BERT trở thành lựa chọn tốt nhất cho các hệ thống bảo mật cần phát hiện được hầu hết email phishing.

