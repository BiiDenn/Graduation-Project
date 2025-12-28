"""
XAI Pipeline - Example Usage
=============================
Các ví dụ minh họa cách dùng `XAIPipeline` để giải thích dự đoán.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from notebooks.XAI.xai_pipeline import XAIPipeline
except ImportError:
    from xai_pipeline import XAIPipeline


def example_1_single_email() -> None:
    """
    Ví dụ 1: Giải thích một email với tất cả models Keras (+ BERT nếu có).
    """
    print("=" * 80)
    print("VÍ DỤ 1: Giải thích một email với nhiều models")
    print("=" * 80)

    pipeline = XAIPipeline()
    pipeline.initialize()

    email = """
    URGENT: Your PayPal account has been limited!

    We detected suspicious activity on your account. To avoid permanent 
    suspension, please verify your identity immediately:

    http://secure-paypal-verify.com/account-restore

    You have 24 hours to respond before your account is closed.
    """

    print(f"\n📧 Email:\n{email.strip()}\n")

    results = pipeline.explain_email(
        email_text=email,
        model_names=None,
        save_outputs=True,
    )

    print("\n" + "=" * 80)
    print("📊 PREDICTIONS")
    print("=" * 80)
    for model, pred in results["predictions"].items():
        label = pred["label"].upper()
        prob = pred["probability"]
        emoji = "🚨" if label == "PHISHING" else "✅"
        print(f"{emoji} {model:10s}: {label:10s} ({prob:.4f})")

    print("\n✅ Results saved to: output/explanations/")


def example_2_multiple_emails() -> None:
    """
    Ví dụ 2: Phân tích nhiều emails cùng lúc.
    """
    print("\n" + "=" * 80)
    print("VÍ DỤ 2: Phân tích nhiều emails")
    print("=" * 80)

    pipeline = XAIPipeline()
    pipeline.initialize()

    emails = [
        "🚨 Congratulations! You won $1,000,000! Click here to claim your prize now!",
        "✅ Hi team, here's the meeting notes from today's standup. Please review.",
        "🚨 Your Netflix subscription expires today! Update payment immediately.",
        "✅ Project deployment completed successfully. All tests passed.",
    ]

    for i, email in enumerate(emails, 1):
        print(f"\n{'='*80}")
        print(f"Email {i}/4: {email[:60]}...")
        print("=" * 80)

        result = pipeline.explain_email(
            email,
            model_names=None,
            save_outputs=False,
        )

        votes = {"phishing": 0, "benign": 0}
        for _, pred in result["predictions"].items():
            votes[pred["label"]] += 1

        consensus = "PHISHING 🚨" if votes["phishing"] > votes["benign"] else "BENIGN ✅"
        print(f"Consensus: {consensus} ({votes['phishing']} phishing, {votes['benign']} benign)")


def example_3_specific_model() -> None:
    """
    Ví dụ 3: Chỉ chạy một model cụ thể (nhanh hơn).
    """
    print("\n" + "=" * 80)
    print("VÍ DỤ 3: Chỉ chạy GRU model")
    print("=" * 80)

    pipeline = XAIPipeline()
    pipeline.initialize()

    email = "Verify your account now or it will be closed permanently!"

    result = pipeline.explain_email(
        email_text=email,
        model_names=["GRU"],
        save_outputs=True,
    )

    print(f"\n📊 GRU Prediction: {result['predictions']['GRU']['label'].upper()}")
    print(f"   Confidence: {result['predictions']['GRU']['probability']:.4f}")

    print("\n🔍 Top Keywords (LIME):")
    for i, token in enumerate(
        result["lime_explanations"]["GRU"]["important_tokens"][:5], 1
    ):
        weight = token["weight"]
        sign = "+" if weight > 0 else ""
        print(f"   {i}. {token['token']:20s} {sign}{weight:.4f}")


def example_4_fast_mode() -> None:
    """
    Ví dụ 4: Chế độ nhanh (chỉ prediction, không LIME).
    """
    print("\n" + "=" * 80)
    print("VÍ DỤ 4: Chế độ nhanh (chỉ prediction)")
    print("=" * 80)

    pipeline = XAIPipeline()
    pipeline.initialize()

    email = "Click here to reset your password immediately!"

    result = pipeline.explain_email(
        email_text=email,
        model_names=None,
        run_lime=False,
        save_outputs=False,
    )

    print("\n⚡ Fast mode: Chỉ mất ~5-10 giây")
    print("\n📊 Quick Predictions:")
    for model, pred in result["predictions"].items():
        print(f"  {model}: {pred['label']}")


if __name__ == "__main__":
    print("\n" + "🔍" * 40)
    print("XAI PIPELINE - EXAMPLE USAGE")
    print("🔍" * 40 + "\n")

    try:
        # example_1_single_email()
        example_2_multiple_emails()
        # example_3_specific_model()
        # example_4_fast_mode()

        print("\n" + "=" * 80)
        print("✅ HOÀN TẤT!")
        print("=" * 80)
        print("\nĐể chạy ví dụ khác, uncomment trong example_usage.py")
        print("Hoặc viết code riêng của bạn dựa trên các ví dụ trên.")

    except Exception as exc:
        print(f"\n❌ LỖI: {exc}")
        import traceback

        traceback.print_exc()
