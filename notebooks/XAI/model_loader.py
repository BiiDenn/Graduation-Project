"""
Module để load và quản lý các mô hình Deep Learning cho XAI.

Hỗ trợ:
- BiLSTM.h5 / .keras
- CNN.h5
- GRU.h5 / .keras
- Hybrid_CNN_BiLSTM.h5 / .keras
- BERT (PyTorch, HuggingFace format)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf

# Import PyTorch và Transformers cho BERT (optional)
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]


class ModelLoader:
    """
    Chịu trách nhiệm:
    - Tìm, load và lưu trữ toàn bộ models dùng cho XAI.
    - Cung cấp hàm `predict()` thống nhất cho mọi loại model.
    """

    def __init__(self, base_path: str = "output/models") -> None:
        """
        Args:
            base_path: Đường dẫn đến thư mục chứa các mô hình.
        """
        self.base_path = base_path
        self.models: Dict[str, Dict[str, Any]] = {}
        self.tokenizers: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Keras helpers
    # ------------------------------------------------------------------ #
    def _reinitialize_text_vectorization(
        self, model: tf.keras.Model, model_name: str, model_path: str
    ) -> None:
        """
        Re-initialize TextVectorization layer vocabulary nếu cần.
        """
        try:
            text_vec_layer = None
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.TextVectorization):
                    text_vec_layer = layer
                    break

            if text_vec_layer is None:
                return

            vocab = text_vec_layer.get_vocabulary()
            if vocab and len(vocab) > 2:
                print(f"  TextVectorization vocabulary đã có sẵn ({len(vocab)} tokens)")
                return

            # Tìm vocabulary file: nếu model trong checkpoints, tìm ở thư mục cha
            model_dir = os.path.dirname(model_path)
            if "checkpoints" in model_dir:
                # Lên 1 cấp để ra thư mục model gốc
                model_base_dir = os.path.dirname(model_dir)
            else:
                model_base_dir = model_dir
            
            vocab_file = None
            possible_vocab_files = [
                os.path.join(model_base_dir, f"{model_name.lower()}_embedding_metadata.tsv"),
                os.path.join(model_base_dir, "hybrid_cnn_bilstm_embedding_metadata.tsv"),
                os.path.join(model_base_dir, "hybrid_embedding_metadata.tsv"),
            ]

            for vocab_path in possible_vocab_files:
                if os.path.exists(vocab_path):
                    vocab_file = vocab_path
                    break

            if vocab_file:
                print(f"  Đang load vocabulary từ {vocab_file}...")

                layer_config = text_vec_layer.get_config()
                max_tokens = layer_config.get("max_tokens", 20000)

                with open(vocab_file, "r", encoding="utf-8") as f:
                    words_from_file = [line.strip() for line in f if line.strip()]

                num_words_to_take = max_tokens - 2
                vocabulary = ["", "[UNK]"] + words_from_file[:num_words_to_take]

                print(
                    f"  Loaded {len(words_from_file)} words from file, "
                    f"taking first {num_words_to_take}"
                )
                text_vec_layer.set_vocabulary(vocabulary)
                print(f"  ✓ Đã re-initialize vocabulary ({len(vocabulary)} tokens)")
            else:
                print("  ⚠ Không tìm thấy vocabulary file.")
                print("  Model có thể không hoạt động đúng!")

        except Exception as exc:
            print(f"  ⚠ Lỗi khi re-initialize TextVectorization: {exc}")
    
    def load_keras_model(self, model_name: str, model_path: str) -> tf.keras.Model:
        """
        Load mô hình Keras (.h5 hoặc .keras) với nhiều phương án fallback.

        Nếu gặp lỗi (đặc biệt là encoding trên Windows), hàm sẽ raise
        exception để pipeline dừng hoàn toàn (giữ nguyên hành vi cũ).
        """
        print(f"Đang load mô hình {model_name} từ {model_path}...")

        original_encoding = os.environ.get("PYTHONIOENCODING", None)
        os.environ["PYTHONIOENCODING"] = "utf-8"

        load_methods = [
            lambda: tf.keras.models.load_model(model_path),
            lambda: tf.keras.models.load_model(model_path, compile=False),
            lambda: tf.keras.models.load_model(model_path, safe_mode=False)
            if hasattr(tf.keras.models, "load_model")
            else None,
        ]

        last_error: Exception | None = None
        for i, load_method in enumerate(load_methods, 1):
            try:
                if load_method is None:
                    continue

                model = load_method()
                print(f"✓ Đã load thành công mô hình {model_name} (method {i})")

                if model_path.endswith(".h5"):
                    self._reinitialize_text_vectorization(model, model_name, model_path)

                if original_encoding:
                    os.environ["PYTHONIOENCODING"] = original_encoding
                elif "PYTHONIOENCODING" in os.environ:
                    del os.environ["PYTHONIOENCODING"]

                return model

            except (UnicodeDecodeError, ValueError) as exc:
                last_error = exc
                error_msg = str(exc)
                if (
                    ("codec can't decode" in error_msg or "charmap" in error_msg.lower())
                    and i < len(load_methods)
                ):
                    print(f"  Thử method {i} thất bại, đang thử method tiếp theo...")
                    continue
                break
            except Exception as exc:
                last_error = exc
                break

        if original_encoding:
            os.environ["PYTHONIOENCODING"] = original_encoding
        elif "PYTHONIOENCODING" in os.environ:
            del os.environ["PYTHONIOENCODING"]

        if last_error:
            error_msg = str(last_error)
            if "codec can't decode" in error_msg or "charmap" in error_msg.lower():
                print(
                    f"✗ Lỗi encoding khi load mô hình {model_name} "
                    f"(đã thử {len(load_methods)} methods):"
                )
                print(
                    "  Nguyên nhân: File mô hình chứa ký tự đặc biệt "
                    "không thể decode bằng encoding mặc định."
                )
                print("  Hành động: Pipeline sẽ DỪNG HOÀN TOÀN để đảm bảo tính nhất quán.")
                print(
                    "  Giải pháp: Cần re-save mô hình với encoding đúng "
                    "hoặc convert sang format .h5"
                )
            else:
                print(f"✗ Lỗi khi load mô hình {model_name}: {error_msg}")
            raise last_error

        raise RuntimeError(f"Không thể load mô hình {model_name} với bất kỳ method nào.")
    
    def load_pytorch_model(self, model_name: str, model_path: str) -> Tuple[Any, Any]:
        """
        Load mô hình PyTorch (BERT từ HuggingFace).
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError(
                "PyTorch và Transformers không có sẵn. "
                "Cài đặt: pip install torch transformers"
            )

        print(f"Đang load mô hình {model_name} từ {model_path}...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.to(device)
            model.eval()  # Set to evaluation mode

            print(f"✓ Đã load thành công mô hình {model_name} (device: {device})")
            return model, tokenizer

        except Exception as exc:
            error_msg = (
                f"\n{'='*80}\n"
                f"LỖI: Không thể load mô hình {model_name}\n"
                f"{'='*80}\n"
                f"Chi tiết lỗi:\n{str(exc)}\n"
                f"{'='*80}\n"
            )
            print(error_msg)
            raise RuntimeError(f"Không thể load mô hình {model_name}") from exc
    
    def load_all_models(self) -> Dict[str, Any]:
        """
        Load tất cả các mô hình có sẵn.

        QUAN TRỌNG:
        - Hàm này yêu cầu TẤT CẢ mô hình phải load thành công.
        - Nếu BẤT KỲ mô hình nào lỗi, pipeline sẽ DỪNG HOÀN TOÀN.
        """
        print(f"\n🔍 Base path: {os.path.abspath(self.base_path)}")
        print(f"🔍 Base path exists: {os.path.exists(self.base_path)}\n")

        models_dict: Dict[str, Dict[str, Any]] = {}
        required_models: list[tuple[str, str, str]] = []

        # BiLSTM
        bilstm_path_h5 = os.path.join(self.base_path, "BiLSTM", "bilstm_model.h5")
        bilstm_path_keras = os.path.join(
            self.base_path, "BiLSTM", "checkpoints", "bilstm_model.keras"
        )

        bilstm_path = None
        bilstm_path_h5_abs = os.path.abspath(bilstm_path_h5)
        bilstm_path_keras_abs = os.path.abspath(bilstm_path_keras)

        if os.path.exists(bilstm_path_h5):
            bilstm_path = bilstm_path_h5
            print(f"  ✓ Tìm thấy BiLSTM.h5 tại: {bilstm_path_h5_abs}")
        elif os.path.exists(bilstm_path_keras):
            bilstm_path = bilstm_path_keras
            print(f"  ✓ Tìm thấy BiLSTM checkpoint tại: {bilstm_path_keras_abs}")
        else:
            print("  ✗ Không tìm thấy BiLSTM tại:")
            print(f"    - {bilstm_path_h5_abs} (exists: {os.path.exists(bilstm_path_h5)})")
            print(
                f"    - {bilstm_path_keras_abs} "
                f"(exists: {os.path.exists(bilstm_path_keras)})"
            )

        if bilstm_path:
            required_models.append(("BiLSTM", bilstm_path, "keras"))
        else:
            raise FileNotFoundError(
                "Không tìm thấy mô hình BiLSTM tại các vị trí:\n"
                f"  - {bilstm_path_h5_abs}\n"
                f"  - {bilstm_path_keras_abs}\n"
                "Tất cả mô hình phải có sẵn để pipeline hoạt động."
            )

        # CNN
        cnn_path_h5 = os.path.join(self.base_path, "CNN", "cnn_model.h5")
        cnn_path_keras = os.path.join(
            self.base_path, "CNN", "checkpoints", "cnn_model.keras"
        )
        
        cnn_path = None
        if os.path.exists(cnn_path_h5):
            cnn_path = cnn_path_h5
            print(f"  ✓ Tìm thấy CNN.h5 tại: {os.path.abspath(cnn_path_h5)}")
        elif os.path.exists(cnn_path_keras):
            cnn_path = cnn_path_keras
            print(f"  ✓ Tìm thấy CNN checkpoint tại: {os.path.abspath(cnn_path_keras)}")
        else:
            print("  ✗ Không tìm thấy CNN tại:")
            print(f"    - {os.path.abspath(cnn_path_h5)} (exists: {os.path.exists(cnn_path_h5)})")
            print(f"    - {os.path.abspath(cnn_path_keras)} (exists: {os.path.exists(cnn_path_keras)})")
        
        if cnn_path:
            required_models.append(("CNN", cnn_path, "keras"))
        else:
            raise FileNotFoundError(
                "Không tìm thấy mô hình CNN tại các vị trí:\n"
                f"  - {os.path.abspath(cnn_path_h5)}\n"
                f"  - {os.path.abspath(cnn_path_keras)}\n"
                "Tất cả mô hình phải có sẵn để pipeline hoạt động."
            )

        # GRU
        gru_path_h5 = os.path.join(self.base_path, "GRU", "gru_model.h5")
        gru_path_fixed = os.path.join(self.base_path, "GRU", "gru_model_fixed.h5")
        gru_path_keras = os.path.join(
            self.base_path, "GRU", "checkpoints", "gru_model.keras"
        )

        gru_path = None
        if os.path.exists(gru_path_h5):
            gru_path = gru_path_h5
            print("  Lưu ý: Sử dụng file .h5 (ưu tiên để tránh lỗi encoding)")
        elif os.path.exists(gru_path_fixed):
            gru_path = gru_path_fixed
            print(f"  Lưu ý: Sử dụng file fixed: {gru_path_fixed}")
        elif os.path.exists(gru_path_keras):
            gru_path = gru_path_keras
            print(
                "  Cảnh báo: Đang load file .keras - có thể gặp lỗi encoding trên Windows"
            )

        if gru_path:
            required_models.append(("GRU", gru_path, "keras"))
        else:
            raise FileNotFoundError(
                "Không tìm thấy mô hình GRU tại các vị trí:\n"
                f"  - {gru_path_keras}\n"
                f"  - {gru_path_h5}\n"
                f"  - {gru_path_fixed}\n"
                "Tất cả mô hình phải có sẵn để pipeline hoạt động."
            )

        # Hybrid_CNN_BiLSTM
        hybrid_path_h5 = os.path.join(
            self.base_path, "Hybrid_CNN_BiLSTM", "hybrid_cnn_bilstm_model.h5"
        )
        hybrid_path_keras = os.path.join(
            self.base_path, "Hybrid_CNN_BiLSTM", "checkpoints", "hybrid_cnn_bilstm_model.keras"
        )

        hybrid_path = None
        if os.path.exists(hybrid_path_h5):
            hybrid_path = hybrid_path_h5
            print("  Lưu ý: Sử dụng file .h5 (ưu tiên để tránh lỗi encoding)")
        elif os.path.exists(hybrid_path_keras):
            hybrid_path = hybrid_path_keras
            print(
                "  Cảnh báo: Đang load file .keras - có thể gặp lỗi encoding trên Windows"
            )

        if hybrid_path:
            required_models.append(("Hybrid_CNN_BiLSTM", hybrid_path, "keras"))
        else:
            raise FileNotFoundError(
                "Không tìm thấy mô hình Hybrid_CNN_BiLSTM tại các vị trí:\n"
                f"  - {hybrid_path_h5}\n"
                f"  - {hybrid_path_keras}\n"
                "Tất cả mô hình phải có sẵn để pipeline hoạt động."
            )

        # BERT
        if PYTORCH_AVAILABLE:
            bert_path = os.path.join(self.base_path, "BERT", "bert_base_email_model")
            if os.path.exists(bert_path) and os.path.exists(
                os.path.join(bert_path, "config.json")
            ):
                required_models.append(("BERT", bert_path, "pytorch"))
            else:
                print(f"  ⚠ Không tìm thấy mô hình BERT tại {bert_path}")
                print("  BERT sẽ không được load. Cài đặt PyTorch và Transformers để sử dụng BERT.")
        else:
            print("  ⚠ PyTorch không có sẵn. BERT sẽ không được load.")

        # Load từng mô hình
        for model_name, model_path, model_type in required_models:
            try:
                if model_type == "keras":
                    model = self.load_keras_model(model_name, model_path)
                    models_dict[model_name] = {"model": model, "type": "keras"}
                elif model_type == "pytorch":
                    model, tokenizer = self.load_pytorch_model(model_name, model_path)
                    models_dict[model_name] = {
                        "model": model,
                        "tokenizer": tokenizer,
                        "type": "pytorch",
                    }
            except Exception as exc:
                error_msg = (
                    f"\n{'='*80}\n"
                    f"LỖI: Không thể load mô hình {model_name}\n"
                    f"{'='*80}\n"
                    f"Pipeline sẽ DỪNG HOÀN TOÀN để đảm bảo tính nhất quán.\n"
                    f"Tất cả mô hình phải load thành công hoặc không chạy gì cả.\n"
                    f"\nChi tiết lỗi:\n{str(exc)}\n"
                    f"{'='*80}\n"
                )
                print(error_msg)
                raise RuntimeError(
                    "Pipeline dừng: Mô hình "
                    f"{model_name} không thể load được. Vui lòng fix lỗi trước khi tiếp tục."
                ) from exc

        print(f"\n{'='*80}")
        print(f"✓ ĐÃ LOAD THÀNH CÔNG TẤT CẢ {len(models_dict)} MÔ HÌNH")
        print(f"  Các mô hình: {', '.join(models_dict.keys())}")
        print(f"{'='*80}\n")

        self.models = models_dict
        return models_dict
    
    def predict_keras(self, model: tf.keras.Model, email_text: str) -> Tuple[str, float]:
        """
        Dự đoán với mô hình Keras (BiLSTM, CNN, GRU, Hybrid_CNN_BiLSTM).
        """
        has_text_vectorization = any(
            isinstance(layer, tf.keras.layers.TextVectorization) for layer in model.layers
        )

        if has_text_vectorization:
            email_tensor = tf.convert_to_tensor([email_text], dtype=tf.string)
        else:
            if not hasattr(self, "_shared_text_vectorizer"):
                self._shared_text_vectorizer = tf.keras.layers.TextVectorization(
                    max_tokens=20000,
                    output_mode="int",
                    output_sequence_length=200,
                )

                bilstm_vocab_file = os.path.join(
                    self.base_path, "BiLSTM", "bilstm_embedding_metadata.tsv"
                )
                if os.path.exists(bilstm_vocab_file):
                    import io

                    with io.open(bilstm_vocab_file, "r", encoding="utf-8") as f:
                        words = [line.strip() for line in f if line.strip()]
                    vocabulary = ["", "[UNK]"] + words[:19998]
                    self._shared_text_vectorizer.set_vocabulary(vocabulary)

            email_tensor = self._shared_text_vectorizer([email_text])

        prediction = model.predict(email_tensor, verbose=0)

        if prediction.ndim > 1:
            prob = (
                float(prediction[0][0])
                if prediction.shape[1] == 1
                else float(np.max(prediction[0]))
            )
        else:
            prob = float(prediction[0])

        label = "phishing" if prob > 0.5 else "benign"
        probability = prob if prob > 0.5 else 1 - prob

        return label, probability
    
    def predict_pytorch(
        self, model: Any, tokenizer: Any, email_text: str, model_name: str | None = None
    ) -> Tuple[str, float]:
        """
        Dự đoán với mô hình PyTorch (BERT).
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch không có sẵn")

        max_length = 512

        inputs = tokenizer(
            email_text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():  # type: ignore[union-attr]
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            probabilities = probabilities.cpu().numpy()[0]

        prob_benign = float(probabilities[0])
        prob_phishing = float(probabilities[1])

        label = "phishing" if prob_phishing > 0.5 else "benign"
        probability = prob_phishing if prob_phishing > 0.5 else prob_benign

        return label, probability

    def predict(self, model_name: str, email_text: str) -> Tuple[str, float]:
        """
        Dự đoán với mô hình bất kỳ.
        """
        if model_name not in self.models:
            raise ValueError(
                f"Mô hình {model_name} chưa được load. Hãy gọi load_all_models() trước."
            )

        model_info = self.models[model_name]

        if model_info["type"] == "keras":
            return self.predict_keras(model_info["model"], email_text)
        if model_info["type"] == "pytorch":
            return self.predict_pytorch(
                model_info["model"],
                model_info["tokenizer"],
                email_text,
                model_name=model_name,
            )

        raise ValueError(f"Loại mô hình không được hỗ trợ: {model_info['type']}")

