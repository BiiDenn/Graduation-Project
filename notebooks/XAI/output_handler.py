"""
Module để xử lý và lưu kết quả XAI ra file JSON.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict


class OutputHandler:
    """
    Xử lý việc lưu / đọc / in kết quả XAI (LIME, SHAP).
    """

    def __init__(self, output_dir: str = "output/explanations") -> None:
        """
        Args:
            output_dir: Thư mục để lưu các file JSON (tự tạo nếu chưa tồn tại).
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # LIME
    # ------------------------------------------------------------------ #
    def save_lime_explanation(
        self,
        model_name: str,
        explanation_result: Dict[str, Any],
        filename: str | None = None,
    ) -> str:
        """
        Lưu kết quả LIME explanation ra file JSON.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name.lower()}_lime_{timestamp}.json"

        if not filename.endswith(".json"):
            filename += ".json"

        filepath = os.path.join(self.output_dir, filename)

        output_data = {
            "email": explanation_result["email"],
            "prediction_label": explanation_result["prediction_label"],
            "prediction_probability": explanation_result["prediction_probability"],
            "important_tokens": explanation_result.get("important_tokens", []),
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(f"✓ Đã lưu LIME explanation vào {filepath}")
        return filepath

    # ------------------------------------------------------------------ #
    # Common helpers
    # ------------------------------------------------------------------ #
    def load_explanation(self, filepath: str) -> Dict[str, Any]:
        """
        Load explanation từ file JSON.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def print_explanation_summary(self, explanation_data: Dict[str, Any]) -> None:
        """
        In ra summary của explanation một cách dễ đọc (dùng cho CLI/debug).
        """
        print("\n" + "=" * 80)
        print("EXPLANATION SUMMARY")
        print("=" * 80)

        print(f"Model: {explanation_data.get('model_name', 'Unknown')}")
        print(f"Email: {explanation_data.get('email', '')[:100]}...")
        print(f"Prediction: {explanation_data.get('prediction_label', 'Unknown').upper()}")
        print(f"Probability: {explanation_data.get('prediction_probability', 0):.4f}")

        if "important_tokens" in explanation_data:
            print("\nImportant Tokens (LIME/SHAP):")
            print("-" * 80)
            for i, token_info in enumerate(explanation_data["important_tokens"][:10], 1):
                token = token_info.get("token", "")
                weight = token_info.get("weight", 0)
                sign = "+" if weight > 0 else ""
                print(f"{i:2d}. {token:30s} {sign}{weight:8.4f}")

        heatmap_path = explanation_data.get("heatmap_path")
        if heatmap_path:
            print(f"\nHeatmap saved at: {heatmap_path}")

        print("=" * 80 + "\n")

    # ------------------------------------------------------------------ #
    # SHAP
    # ------------------------------------------------------------------ #
    def save_shap_explanation(
        self,
        model_name: str,
        explanation_result: Dict[str, Any],
        filename: str | None = None,
    ) -> str:
        """
        Lưu kết quả SHAP explanation ra file JSON.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name.lower()}_shap_{timestamp}.json"

        if not filename.endswith(".json"):
            filename += ".json"

        filepath = os.path.join(self.output_dir, filename)

        output_data = {
            "email": explanation_result["email"],
            "prediction_label": explanation_result["prediction_label"],
            "prediction_probability": explanation_result["prediction_probability"],
            "important_tokens": explanation_result["important_tokens"],
            "elapsed_time": explanation_result.get("elapsed_time", 0),
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "method": "SHAP",
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(f"✓ Đã lưu SHAP explanation vào {filepath}")
        return filepath

