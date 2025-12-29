"""
Pipeline chính để chạy XAI cho tất cả các mô hình (Keras + BERT).

Kết hợp:
- `ModelLoader` để load/predict.
- `LIMEExplainer` cho các model Keras.
- `SHAPExplainer` cho BERT (nếu có).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .model_loader import ModelLoader
from .lime_explainer import LIMEExplainer
from .shap_explainer import SHAPExplainer
from .output_handler import OutputHandler

PROJECT_ROOT = Path(__file__).parent.parent.parent


class XAIPipeline:
    """
    Pipeline cao cấp để:
    - Load tất cả models.
    - Chạy dự đoán và sinh giải thích (LIME/SHAP) cho email.
    """

    def __init__(
        self,
        models_base_path: str = "output/models",
        output_dir: str = "output/explanations",
    ) -> None:
        """
        Args:
            models_base_path: Thư mục chứa models (tương đối theo project root hoặc tuyệt đối).
            output_dir: Thư mục để lưu kết quả XAI (JSON/HTML).
        """
        if not os.path.isabs(models_base_path):
            models_base_path = str(PROJECT_ROOT / models_base_path)
        if not os.path.isabs(output_dir):
            output_dir = str(PROJECT_ROOT / output_dir)

        self.models_base_path = models_base_path
        self.output_dir = output_dir

        self.model_loader = ModelLoader(base_path=models_base_path)
        self.output_handler = OutputHandler(output_dir=output_dir)

        self.lime_explainer: Optional[LIMEExplainer] = None
        self.shap_explainer: Optional[SHAPExplainer] = None

    # ------------------------------------------------------------------ #
    # Khởi tạo pipeline
    # ------------------------------------------------------------------ #
    def initialize(self) -> None:
        """
        Load tất cả các mô hình và khởi tạo explainers.
        """
        print("=" * 80)
        print("KHỞI TẠO XAI PIPELINE")
        print("=" * 80)

        models = self.model_loader.load_all_models()
        if not models:
            raise ValueError(
                "Không tìm thấy mô hình nào. Hãy kiểm tra đường dẫn models_base_path."
            )

        print(f"\nĐã load {len(models)} mô hình: {', '.join(models.keys())}")

        self.lime_explainer = LIMEExplainer(self.model_loader)

        if "BERT" in models:
            bert_path = os.path.join(self.models_base_path, "BERT", "bert_base_email_model")
            if os.path.exists(bert_path):
                try:
                    self.shap_explainer = SHAPExplainer(bert_path)
                    print("✓ SHAP explainer đã được khởi tạo cho BERT")
                except Exception as exc:
                    print(f"⚠ Không thể khởi tạo SHAP explainer cho BERT: {exc}")
                    self.shap_explainer = None
            else:
                print(f"⚠ Không tìm thấy BERT model tại {bert_path}")
                self.shap_explainer = None

        print("\n✓ Pipeline đã sẵn sàng!\n")

    # ------------------------------------------------------------------ #
    # API chính
    # ------------------------------------------------------------------ #
    def explain_email(
        self,
        email_text: str,
        model_names: Optional[List[str]] = None,
        run_lime: bool = True,
        save_outputs: bool = True,
    ) -> Dict[str, Any]:
        """
        Giải thích một email với một hoặc nhiều mô hình.
        """
        if self.lime_explainer is None:
            raise RuntimeError("Pipeline chưa được khởi tạo. Hãy gọi initialize() trước.")

        results: Dict[str, Any] = {
            "email": email_text,
            "predictions": {},
            "lime_explanations": {},
            "shap_explanations": {},
        }

        if model_names is None:
            model_names = list(self.model_loader.models.keys())

        print("=" * 80)
        print("GIẢI THÍCH EMAIL")
        print("=" * 80)
        print(f"Email: {email_text[:100]}...")
        print(f"Mô hình sẽ chạy: {', '.join(model_names)}")
        print("=" * 80 + "\n")

        for model_name in model_names:
            print(f"\n[{model_name}] Đang xử lý...")

            label, probability = self.model_loader.predict(model_name, email_text)
            results["predictions"][model_name] = {
                "label": label,
                "probability": probability,
            }
            print(f"  Prediction: {label.upper()} ({probability:.4f})")

            if model_name == "BERT" and run_lime:
                if self.shap_explainer is not None:
                    try:
                        shap_result = self.shap_explainer.explain_with_shap_fast(
                            email_text,
                            max_features=15,
                        )
                        results["shap_explanations"][model_name] = shap_result

                        if save_outputs:
                            self.output_handler.save_shap_explanation(
                                model_name,
                                shap_result,
                            )

                        print("  Top 5 từ quan trọng (SHAP):")
                        for i, token_info in enumerate(
                            shap_result["important_tokens"][:5], 1
                        ):
                            print(
                                f"    {i}. {token_info['token']:20s} "
                                f"{token_info['weight']:+.4f}"
                            )
                        print(
                            f"  ⚡ SHAP hoàn thành trong "
                            f"{shap_result.get('elapsed_time', 0):.2f}s"
                        )
                    except Exception as exc:
                        print(f"  ✗ Lỗi khi chạy SHAP: {exc}")
                        results["shap_explanations"][model_name] = None
                else:
                    print("  ⚠ SHAP explainer không có sẵn cho BERT")
            elif run_lime:
                try:
                    assert self.lime_explainer is not None
                    # Tăng num_features để đảm bảo có đủ từ sau khi lọc stopwords
                    lime_result = self.lime_explainer.explain_with_lime(
                        model_name,
                        email_text,
                        num_features=25,  # Tăng từ 15 lên 25 để đảm bảo đủ từ
                    )
                    results["lime_explanations"][model_name] = lime_result

                    if save_outputs:
                        self.output_handler.save_lime_explanation(
                            model_name,
                            lime_result,
                        )

                    # Hiển thị tokens theo logic mới: ưu tiên nhóm chính
                    tokens = lime_result["important_tokens"]
                    positive_tokens = [t for t in tokens if t["weight"] > 0]
                    negative_tokens = [t for t in tokens if t["weight"] < 0]
                    
                    prediction_label = lime_result.get("prediction_label", "").lower()
                    
                    if prediction_label == "phishing":
                        print(f"  Top {len(positive_tokens)} từ làm tăng khả năng PHISHING:")
                        for i, token_info in enumerate(positive_tokens, 1):
                            print(
                                f"    {i}. {token_info['token']:20s} "
                                f"{token_info['weight']:+.4f}"
                            )
                        if negative_tokens:
                            print(f"  Top {len(negative_tokens)} từ làm tăng khả năng BENIGN:")
                            for i, token_info in enumerate(negative_tokens, 1):
                                print(
                                    f"    {i}. {token_info['token']:20s} "
                                    f"{token_info['weight']:+.4f}"
                                )
                    else:  # benign
                        print(f"  Top {len(negative_tokens)} từ làm tăng khả năng BENIGN:")
                        for i, token_info in enumerate(negative_tokens, 1):
                            print(
                                f"    {i}. {token_info['token']:20s} "
                                f"{token_info['weight']:+.4f}"
                            )
                        if positive_tokens:
                            print(f"  Top {len(positive_tokens)} từ làm tăng khả năng PHISHING:")
                            for i, token_info in enumerate(positive_tokens, 1):
                                print(
                                    f"    {i}. {token_info['token']:20s} "
                                    f"{token_info['weight']:+.4f}"
                                )

                except Exception as exc:
                    print(f"  ✗ Lỗi khi chạy LIME: {exc}")
                    results["lime_explanations"][model_name] = None

        print("\n" + "=" * 80)
        print("HOÀN TẤT")
        print("=" * 80 + "\n")

        return results

    def explain_batch(
        self,
        emails: List[str],
        model_names: Optional[List[str]] = None,
        run_lime: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Giải thích nhiều emails tuần tự.
        """
        results: List[Dict[str, Any]] = []
        for i, email in enumerate(emails, 1):
            print(f"\n{'='*80}")
            print(f"EMAIL {i}/{len(emails)}")
            print(f"{'='*80}\n")

            result = self.explain_email(
                email,
                model_names=model_names,
                run_lime=run_lime,
                save_outputs=True,
            )
            results.append(result)

        return results


def main() -> None:
    """
    Hàm main demo cách sử dụng `XAIPipeline`.
    """
    pipeline = XAIPipeline()
    pipeline.initialize()

    test_emails = [
        "Your account has been suspended. Please verify your identity immediately by clicking this link: http://verify-account.com",
        "Hi team, just a quick reminder about our meeting tomorrow at 10 AM. Please prepare your reports.",
        "Congratulations! You've won a free iPhone. Click here to claim your prize now!",
    ]

    print("\n" + "=" * 80)
    print("CHẠY XAI PIPELINE - VÍ DỤ")
    print("=" * 80 + "\n")

    result = pipeline.explain_email(
        test_emails[0],
        model_names=None,
        run_lime=True,
        save_outputs=True,
    )

    print("\n" + "=" * 80)
    print("TÓM TẮT KẾT QUẢ")
    print("=" * 80)
    print(f"Email: {result['email'][:80]}...")
    print("\nPredictions:")
    for model_name, pred in result["predictions"].items():
        print(f"  {model_name:10s}: {pred['label']:8s} ({pred['probability']:.4f})")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
