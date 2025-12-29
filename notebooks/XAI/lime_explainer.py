"""
Module LIME (Local Interpretable Model-Agnostic Explanations)
để giải thích dự đoán của các mô hình Deep Learning (Keras).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import tensorflow as tf
from lime.lime_text import LimeTextExplainer


class LIMEExplainer:
    """
    Sinh giải thích LIME cho các mô hình Keras trong `ModelLoader`.
    """

    def __init__(self, model_loader: Any) -> None:
        """
        Args:
            model_loader: Instance `ModelLoader` đã load các mô hình.
        """
        self.model_loader = model_loader
        self.explainer = LimeTextExplainer(class_names=["benign", "phishing"])

    def _create_predict_fn_keras(
        self, model: tf.keras.Model, model_name: Optional[str] = None
    ) -> Callable[[List[str]], np.ndarray]:
        """
        Tạo hàm dự đoán cho LIME từ một mô hình Keras.

        - Nếu model có `TextVectorization` thì nhận trực tiếp raw text.
        - Nếu không (như CNN), tái sử dụng `TextVectorization` từ BiLSTM.
        """
        has_text_vectorization = any(
            isinstance(layer, tf.keras.layers.TextVectorization) for layer in model.layers
        )

        text_vectorizer: Optional[tf.keras.layers.TextVectorization] = None
        if not has_text_vectorization:
            # Lấy TextVectorization từ BiLSTM (đã được train cùng vocabulary)
            bilstm_info = self.model_loader.models.get("BiLSTM")
            if bilstm_info and bilstm_info.get("model") is not None:
                bilstm_model = bilstm_info["model"]
                for layer in bilstm_model.layers:
                    if isinstance(layer, tf.keras.layers.TextVectorization):
                        text_vectorizer = layer
                        break

        def predict_fn(texts: List[str]) -> np.ndarray:
            """
            Hàm dự đoán cho LIME.

            Returns:
                Mảng shape `(n_samples, 2)` với `[prob_benign, prob_phishing]`.
            """
            texts_tensor = tf.convert_to_tensor(texts, dtype=tf.string)

            if text_vectorizer is not None:
                texts_tensor = text_vectorizer(texts_tensor)

            predictions = model.predict(texts_tensor, verbose=0)

            results: List[List[float]] = []
            for pred in predictions:
                if pred.ndim == 0:
                    prob = float(pred)
                elif len(pred) == 1:
                    prob = float(pred[0])
                else:
                    prob = float(np.max(pred))

                prob_benign = 1 - prob if prob > 0.5 else prob
                prob_phishing = prob if prob > 0.5 else 1 - prob
                results.append([prob_benign, prob_phishing])

            return np.array(results)

        return predict_fn

    def explain_with_lime(
        self,
        model_name: str,
        email_text: str,
        num_features: int = 10,
        num_samples: int = 5000,
    ) -> Dict[str, Any]:
        """
        Tạo LIME explanation cho một email và một mô hình Keras.
        """
        if model_name not in self.model_loader.models:
            raise ValueError(f"Mô hình {model_name} chưa được load.")

        model_info = self.model_loader.models[model_name]

        if model_info["type"] != "keras":
            raise ValueError(f"Loại mô hình không được hỗ trợ: {model_info['type']}")

        predict_fn = self._create_predict_fn_keras(model_info["model"], model_name)

        # Tăng num_features để đảm bảo sau khi lọc stopwords vẫn còn đủ từ
        # Cần ít nhất 10 từ cho nhóm chính + 5 từ cho nhóm phụ = 15 từ
        # Nhưng sau khi lọc stopwords có thể mất 30-50%, nên yêu cầu nhiều hơn
        expanded_num_features = max(num_features * 2, 25)  # Ít nhất 25 features
        
        explanation = self.explainer.explain_instance(
            email_text,
            predict_fn,
            num_features=expanded_num_features,
            num_samples=num_samples,
        )

        # Lấy dự đoán ban đầu từ `ModelLoader` (giữ đúng logic phân loại)
        label, probability = self.model_loader.predict(model_name, email_text)

        important_tokens: List[Dict[str, float]] = []
        exp_list = explanation.as_list()

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

        for token, weight in exp_list:
            token_clean = token.strip()
            token_lower = token_clean.lower()

            if not token_clean or len(token_clean) < 2:
                continue
            if token_lower in english_stopwords:
                continue

            important_tokens.append(
                {
                    "token": token_clean,
                    "weight": float(weight),
                }
            )

        # Tách tokens thành 2 nhóm: positive (phishing) và negative (benign)
        positive_tokens = [t for t in important_tokens if t["weight"] > 0]
        negative_tokens = [t for t in important_tokens if t["weight"] < 0]
        
        # Sắp xếp mỗi nhóm theo absolute weight (giảm dần)
        positive_tokens.sort(key=lambda x: abs(x["weight"]), reverse=True)
        negative_tokens.sort(key=lambda x: abs(x["weight"]), reverse=True)
        
        # Dựa vào prediction label để quyết định số lượng từ hiển thị
        # Nếu dự đoán là phishing: ưu tiên 10 từ phishing, chỉ 5 từ benign
        # Nếu dự đoán là benign: ưu tiên 10 từ benign, chỉ 5 từ phishing
        if label.lower() == "phishing":
            # Lấy 10 từ positive (phishing) và 5 từ negative (benign)
            selected_positive = positive_tokens[:10]
            selected_negative = negative_tokens[:5]
        else:  # benign
            # Lấy 10 từ negative (benign) và 5 từ positive (phishing)
            selected_positive = positive_tokens[:5]
            selected_negative = negative_tokens[:10]
        
        # Kết hợp lại và sắp xếp để hiển thị (ưu tiên hiển thị nhóm chính trước)
        # Nhưng vẫn giữ thứ tự theo absolute weight trong mỗi nhóm
        final_tokens = []
        if label.lower() == "phishing":
            # Hiển thị phishing tokens trước, sau đó benign tokens
            final_tokens = selected_positive + selected_negative
        else:  # benign
            # Hiển thị benign tokens trước, sau đó phishing tokens
            final_tokens = selected_negative + selected_positive

        return {
            "email": email_text,
            "prediction_label": label,
            "prediction_probability": probability,
            "important_tokens": final_tokens,
            "explanation_object": explanation,
        }

    def visualize_explanation(
        self,
        explanation_result: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> None:
        """
        In kết quả LIME ra console, và (tuỳ chọn) lưu ra file HTML.
        """
        explanation = explanation_result.get("explanation_object")
        if explanation is None:
            print("Không có explanation object để visualize.")
            return

        print("\n" + "=" * 80)
        print("LIME EXPLANATION")
        print("=" * 80)
        print(f"Email: {explanation_result['email'][:100]}...")
        print(
            f"Prediction: {explanation_result['prediction_label'].upper()} "
            f"(Probability: {explanation_result['prediction_probability']:.4f})"
        )
        print("\nImportant Tokens:")
        print("-" * 80)

        for i, token_info in enumerate(explanation_result["important_tokens"], 1):
            weight = token_info["weight"]
            token = token_info["token"]
            sign = "+" if weight > 0 else ""
            print(f"{i:2d}. {token:30s} {sign}{weight:8.4f}")

        print("=" * 80 + "\n")

        if save_path:
            try:
                explanation.save_to_file(save_path)
                print(f"Đã lưu visualization vào {save_path}")
            except Exception as exc:
                print(f"Không thể lưu visualization: {str(exc)}")

