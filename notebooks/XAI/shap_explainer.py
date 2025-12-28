"""
Module SHAP (SHapley Additive exPlanations) cho BERT.

Mô-đun này triển khai một cách giải thích dạng "SHAP-like" bằng kỹ thuật
mask từng token và đo độ thay đổi xác suất/logit, được tối ưu để dùng
trong extension trình duyệt (khoảng vài giây cho mỗi email).
"""

from __future__ import annotations

import threading
import time
import warnings
import re
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")


class SHAPExplainer:
    """
    Tạo giải thích dạng SHAP cho các model BERT-based (HuggingFace).

    Ý tưởng chính:
    - Cache model/tokenizer, chỉ load 1 lần (thread‑safe).
    - Dùng batching khi chạy các phiên bản đã mask token.
    - Gom và lọc các token để lấy danh sách từ khóa dễ đọc cho người dùng.
    """

    def __init__(self, model_path: str, device: Optional[str] = None, preload_model: bool = True) -> None:
        """
        Args:
            model_path: Thư mục chứa model HuggingFace (config.json, tokenizer, weights...).
            device: `'cuda'`, `'cpu'` hoặc None để tự động chọn.
            preload_model: Nếu True, load model ngay khi khởi tạo (trong background thread).
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Cache / trạng thái load
        self._model: Optional[torch.nn.Module] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._explainer: Any = None  # Giữ lại thuộc tính này để không phá vỡ API cũ nếu có dùng reflection
        self._model_loading: bool = False
        self._model_lock = threading.Lock()

        # Giới hạn độ dài chuỗi đầu vào theo config model
        self._max_length = self._get_max_length_from_config()

        print(f"🔧 SHAPExplainer initialized (device: {self.device}, max_length: {self._max_length})")

        # Preload model trong background để lần gọi đầu không bị trễ
        if preload_model:
            print(f"📥 Pre-loading model từ {model_path} trong background...")
            self._load_thread = threading.Thread(target=self._load_model, daemon=True)
            self._load_thread.start()
            # Cho background thread một chút thời gian để bắt đầu
            time.sleep(0.1)

    # --------------------------------------------------------------------- #
    # Model loading helpers
    # --------------------------------------------------------------------- #
    def _get_max_length_from_config(self) -> int:
        """
        Lấy `max_length` hợp lý từ file `config.json` của model.

        Returns:
            Độ dài tối đa cho tokenizer (đã trừ 2 token đặc biệt CLS/SEP).
            - Nếu không đọc được config: 512.
        """
        import json
        import os

        config_path = os.path.join(self.model_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)

                max_position_embeddings = config.get("max_position_embeddings", 512)
                # Trừ 2 cho special tokens (CLS và SEP)
                return max_position_embeddings - 2
            except Exception as exc:  # giữ hành vi cũ: log và fallback
                print(f"⚠ Không thể đọc config, sử dụng max_length mặc định 512: {exc}")
                return 512

        # Không có config, dùng mặc định cho BERT
        return 512

    def _load_model(self) -> None:
        """
        Load model & tokenizer một lần duy nhất (thread‑safe).
        """
        # Chặn các thread khác nếu đang load
        with self._model_lock:
            if self._model is not None and self._tokenizer is not None:
                return
            if self._model_loading:
                return
            self._model_loading = True

        try:
            print(f"📥 Loading BERT model from {self.model_path}...")
            start_time = time.time()

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True,
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
            )
            model.to(self.device)
            model.eval()

            # Warm‑up model với dummy input (giúp lần chạy thật nhanh ổn định hơn)
            try:
                dummy_inputs = tokenizer(
                    "test",
                    padding=True,
                    truncation=True,
                    max_length=min(128, self._max_length),
                    return_tensors="pt",
                )
                dummy_inputs = {k: v.to(self.device) for k, v in dummy_inputs.items()}
                with torch.no_grad():
                    _ = model(**dummy_inputs)
            except Exception as warmup_error:
                print(f"⚠ Warm-up model thất bại (không ảnh hưởng): {warmup_error}")

            # Ghi lại vào instance sau khi load xong hoàn toàn
            with self._model_lock:
                self._tokenizer = tokenizer
                self._model = model
                self._model_loading = False

            print(f"✓ Model loaded in {time.time() - start_time:.2f}s")
        except Exception as exc:
            with self._model_lock:
                self._model_loading = False
            print(f"❌ Lỗi khi load model: {exc}")
            raise

    def _wait_until_model_ready(self) -> None:
        """
        Đảm bảo model/tokenizer đã sẵn sàng, chờ background thread nếu cần.
        """
        self._load_model()

        if hasattr(self, "_load_thread") and self._load_thread.is_alive():
            print("⏳ Đang đợi model load xong...")
            self._load_thread.join(timeout=60)
            if self._model is None:
                raise RuntimeError("Model không thể load được trong thời gian cho phép")

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model chưa được load. Vui lòng đợi thêm hoặc kiểm tra lại.")

    # --------------------------------------------------------------------- #
    # Prediction helpers
    # --------------------------------------------------------------------- #
    def _create_predict_fn(self):
        """
        Trả về hàm nhận list string và trả về xác suất \\(shape = (n_samples, 2)\\).
        """
        self._wait_until_model_ready()

        def predict_fn(texts: Iterable[str]) -> np.ndarray:
            # Đảm bảo đầu vào là list[str]
            if isinstance(texts, str):
                texts = [texts]
            elif not isinstance(texts, (list, tuple)):
                texts = [str(texts)]

            assert self._tokenizer is not None  # để mypy/IDE hài lòng
            inputs = self._tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            assert self._model is not None
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

            # BERT output: [prob_class_0, prob_class_1] (benign, phishing)
            return probabilities

        return predict_fn

    # --------------------------------------------------------------------- #
    # Core explanation logic
    # --------------------------------------------------------------------- #
    def explain_with_shap(
        self,
        email_text: str,
        num_samples: int = 50,  # Giữ lại tham số cho backward‑compat, không dùng trực tiếp
        max_features: int = 15,
        token_limit: Optional[int] = None,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Sinh giải thích dạng SHAP cho một email.

        Args:
            email_text: Nội dung email cần giải thích.
            num_samples: Tham số giữ lại để tương thích (không dùng trực tiếp nữa).
            max_features: Số lượng từ khóa quan trọng trả về.
            token_limit: Giới hạn số token đầu vào để tăng tốc (None = dùng hết).
            batch_size: Kích thước batch khi chạy các bản đã mask.
        """
        start_time = time.time()
        self._wait_until_model_ready()

        predict_fn = self._create_predict_fn()
        initial_pred = predict_fn([email_text])[0]
        prob_benign = float(initial_pred[0])
        prob_phishing = float(initial_pred[1])
        label = "phishing" if prob_phishing > 0.5 else "benign"
        probability = prob_phishing if prob_phishing > 0.5 else prob_benign

        print("🔍 Generating SHAP explanation (analyzing up to 512 tokens)...")

        # Tokenize để lấy subwords
        assert self._tokenizer is not None
        bert_tokens = self._tokenizer.tokenize(email_text)
        if not bert_tokens:
            bert_tokens = email_text.split()

        # Giới hạn số token
        effective_limit = self._max_length
        if token_limit is not None:
            effective_limit = min(effective_limit, token_limit)
        bert_tokens = bert_tokens[: min(effective_limit, len(bert_tokens))]

        # Baseline: dự đoán trên email gốc
        baseline_pred = predict_fn([email_text])[0]
        baseline_prob_phishing = float(baseline_pred[1])

        encoded = self._tokenizer.encode_plus(
            email_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        original_input_ids = encoded["input_ids"].clone()
        attention_mask = encoded["attention_mask"]
        mask_token_id = self._tokenizer.mask_token_id

        shap_values_simple: List[Dict[str, float]] = []
        masked_inputs_list: List[torch.Tensor] = []
        token_meta: List[str] = []

        # Tạo bản masked cho từng token
        for token in bert_tokens:
            token_clean = token.replace("##", "").strip()
            if not token_clean:
                continue

            try:
                token_id = self._tokenizer.convert_tokens_to_ids([token])[0]
                if token_id == self._tokenizer.unk_token_id:
                    token_id = self._tokenizer.convert_tokens_to_ids([token_clean])[0]
            except Exception:
                # Không map được token → bỏ qua
                continue

            masked_input_ids = original_input_ids.clone()
            mask_positions = (masked_input_ids[0] == token_id).nonzero(as_tuple=True)[0]
            if not len(mask_positions):
                continue

            # Chỉ mask token đầu tiên không phải CLS/SEP
            masked = False
            for pos in mask_positions:
                if 0 < pos < len(masked_input_ids[0]) - 1:
                    masked_input_ids[0][pos] = mask_token_id
                    masked = True
                    break

            if masked:
                masked_inputs_list.append(masked_input_ids)
                token_meta.append(token_clean or token)

        if masked_inputs_list:
            baseline_logit_phishing = float(
                torch.logit(torch.tensor(baseline_prob_phishing), eps=1e-6)
            )

            idx = 0
            while idx < len(masked_inputs_list):
                batch_inputs_ids = masked_inputs_list[idx : idx + batch_size]
                batch_tokens = token_meta[idx : idx + batch_size]

                batch_input_ids_tensor = torch.cat(batch_inputs_ids, dim=0).to(self.device)
                batch_attention_mask = attention_mask.repeat(
                    batch_input_ids_tensor.size(0), 1
                ).to(self.device)

                inputs = {
                    "input_ids": batch_input_ids_tensor,
                    "attention_mask": batch_attention_mask,
                }

                try:
                    assert self._model is not None
                    with torch.no_grad():
                        outputs = self._model(**inputs)
                        logits = outputs.logits
                        probs = torch.softmax(logits, dim=-1)

                    for j in range(batch_input_ids_tensor.size(0)):
                        masked_prob_phishing = float(probs[j][1])
                        masked_logit_phishing = float(logits[j][1])

                        delta_logit = baseline_logit_phishing - masked_logit_phishing
                        delta_prob = baseline_prob_phishing - masked_prob_phishing

                        shap_value = delta_logit if abs(delta_logit) > 0.01 else delta_prob * 1000

                        shap_values_simple.append(
                            {
                                "token": batch_tokens[j],
                                "weight": float(shap_value),
                            }
                        )
                except Exception:
                    # Giữ nguyên hành vi cũ: nếu batch lỗi thì bỏ qua batch đó
                    pass

                idx += batch_size

        # ------------------------------------------------------------------ #
        # Gom và lọc token
        # ------------------------------------------------------------------ #
        token_dict: Dict[str, float] = {}
        for item in shap_values_simple:
            token = item["token"]
            weight = item["weight"]
            token_dict[token] = token_dict.get(token, 0.0) + weight

        def _is_valid_word(token: str) -> bool:
            """
            Kiểm tra token có phải là “từ thật sự” để hiển thị cho người dùng.
            """
            if not token or not token.strip():
                return False

            token_clean = token.strip()
            token_lower = token_clean.lower()

            if len(token_clean) < 2:
                return False

            english_stopwords = {
                "the",
                "and",
                "or",
                "but",
                "if",
                "while",
                "for",
                "on",
                "in",
                "at",
                "to",
                "from",
                "by",
                "with",
                "of",
                "as",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "a",
                "an",
                "this",
                "that",
                "these",
                "those",
                "it",
                "its",
                "into",
                "about",
                "over",
                "under",
                "up",
                "down",
                "your",
                "you",
                "we",
                "our",
                "us",
                "i",
                "me",
                "my",
            }
            if token_lower in english_stopwords:
                return False

            special_chars = [
                "/",
                ".",
                ":",
                "-",
                "!",
                ",",
                "'",
                '"',
                ";",
                "?",
                "(",
                ")",
                "[",
                "]",
                "{",
                "}",
                "=",
                "+",
                "*",
                "&",
                "%",
                "$",
                "#",
                "@",
                "^",
                "~",
                "`",
                "|",
                "\\",
            ]
            if token_clean in special_chars:
                return False

            # Cho phép các từ có chứa chữ cái, có thể kèm số hoặc dấu '-'
            if re.match(r"^[a-zA-Z]+([-][a-zA-Z0-9]+)*$", token_clean):
                return True
            if re.match(r"^[a-zA-Z]+[0-9]*$", token_clean) or re.match(
                r"^[0-9]*[a-zA-Z]+$", token_clean
            ):
                return True
            if re.match(r"^[a-zA-Z]+$", token_clean):
                return True

            return False

        filtered_tokens = [
            {"token": token, "weight": weight}
            for token, weight in token_dict.items()
            if _is_valid_word(token)
        ]
        filtered_tokens.sort(key=lambda x: abs(x["weight"]), reverse=True)

        class SimpleSHAPValues:
            """
            Đối tượng “giả” mô phỏng cấu trúc shap_values để tương thích code cũ.
            """

            def __init__(self, tokens_data: List[Dict[str, float]]) -> None:
                if tokens_data:
                    self.values = np.array([[t["weight"] for t in tokens_data]])
                    self.data = [t["token"] for t in tokens_data]
                else:
                    self.values = np.array([[]])
                    self.data: List[str] = []

        shap_values = SimpleSHAPValues(filtered_tokens)
        important_tokens = filtered_tokens[:max_features]

        elapsed_time = time.time() - start_time
        print(f"✓ SHAP explanation completed in {elapsed_time:.2f}s")

        return {
            "email": email_text,
            "prediction_label": label,
            "prediction_probability": probability,
            "important_tokens": important_tokens[:max_features],
            "shap_values": shap_values,
            "elapsed_time": elapsed_time,
        }

    def explain_with_shap_fast(self, email_text: str, max_features: int = 15) -> Dict[str, Any]:
        """
        Phiên bản nhanh cho extension: giới hạn token để chạy trong vài giây.
        """
        return self.explain_with_shap(
            email_text=email_text,
            num_samples=30,  # giữ tham số cho backward‑compat
            max_features=max_features,
            token_limit=80,
            batch_size=32,
        )

    def warmup(self, text: str = "test email for warmup", max_features: int = 5) -> None:
        """
        Chạy nhanh một lần explain để warm‑up model/tokenizer & kernel.
        """
        try:
            print("🔥 Đang warm-up SHAPExplainer...")
            _ = self.explain_with_shap_fast(text, max_features=max_features)
            print("✓ Warm-up SHAPExplainer hoàn tất")
        except Exception as exc:
            print(
                f"⚠ Warm-up SHAPExplainer thất bại (không ảnh hưởng đến hoạt động chính): {exc}"
            )

