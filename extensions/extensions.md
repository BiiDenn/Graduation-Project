Hãy thiết kế một Chrome Extension (hoặc Edge Extension) có khả năng phân loại email trong Gmail.com hoặc Outlook thành hai nhãn:
    - benign (hợp lệ)
    - phishing
Extension phải tích hợp 5 mô hình Deep Learning gồm: GRU.h5, CNN.h5, BiLSTM.h5, CNN-BiLSTM.h5, và BERT (lưu ở dạng folder HuggingFace, không phải .h5)

Tất cả các models nằm trong folder output/.

Tôi cần bạn đề xuất một ý tưởng hoàn chỉnh – chi tiết – logic – dễ sử dụng – tối ưu trải nghiệm người dùng – và khả thi trong thực tế.

1. Mục tiêu chính của Extension
Khi người dùng mở bất kỳ email nào trong Gmail hoặc Outlook:

    * Content Script tự động đọc nội dung email, gồm: subject và body text
    * Gửi nội dung email sang backend Python để chạy dự đoán bằng 5 model.
    * Extension hiển thị kết quả của từng mô hình, bao gồm:
        - Nhãn dự đoán: benign hoặc phishing
        - Confidence (%)
        - Progress bar hiển thị trực quan:
            + Xanh → benign
            + Đỏ → phishing
        - Nút Explain → mở modal hiển thị giải thích XAI:
            + LIME cho GRU / CNN / BiLSTM / CNN-BiLSTM
            + SHAP token-level cho BERT

🔥 Yêu cầu quan trọng:
    * Prediction phải hiển thị nhanh, không để user chờ model load.
    * XAI phải hiển thị ngay lập tức, không gây delay. 

2. Kiến trúc hệ thống
Extension và backend được chia thành 2 phần:

A. Browser Extension
Gồm 3 thành phần chính:

2.1 Content Script
    - Tự động phát hiện khi user mở email trong Gmail/Outlook.
    - Trích xuất DOM: Subject và Body text
    - Lưu nội dung email vào localStorage hoặc chrome.storage.

2.2 Popup UI
    - Lấy nội dung email đã lưu.
    - Gửi request đến backend Python để chạy model.
    - Nhận kết quả và hiển thị UI.
    - Khi user bấm “Explain”, gọi API lấy XAI HTML.
2.3. Background Worker
    - Điều phối giao tiếp giữa popup và content script.
    - Xử lý event lâu dài.

B. Backend Python (Flask hoặc FastAPI)
Backend cần thực hiện:
2.4 Load 4 mô hình .h5 khi khởi động server: GRU, CNN, BiLSTM, CNN-BiLSTM và load BERT HuggingFace từ folder.

2.5 Khởi tạo SHAP Explainer cho BERT khi server start (giữ global).

2.6 API /predict:
    - Nhận nội dung email.
    - Trả về prediction + confidence của 5 model.

2.7 API /explain:
    - Với model LSTM/GRU/CNN → trả HTML LIME.
    - Với BERT → trả HTML SHAP (token-level heatmap hoặc force plot). 

Flow tổng thể
User mở email 
→ Content Script đọc nội dung 
→ Popup gửi API predict 
→ Backend trả kết quả 5 model 
→ Popup hiển thị UI 
→ User bấm Explain 
→ Popup mở modal XAI (HTML từ backend)

3. Thiết kế giao diện người dùng (UI/UX)
➡️ Popup UI
Header: Email Phishing Detection
Email Summary: 
    - Subject (rút gọn)
    - Preview 1–2 dòng body text

Block kết quả cho từng model. Mỗi model có:
    - Tên model: GRU / CNN / BiLSTM / CNN-BiLSTM / BERT
    - Nhãn dự đoán:
        + Benign (màu xanh)
        + Phishing (màu đỏ)
    - Thanh progress bar thể hiện confidence
    - Nút Explain Prediction

➡️ Modal XAI
Hiển thị HTML do backend trả về:
    - Với LIME:
        + Highlight màu theo mức đóng góp từ
        + Green → giảm xác suất phishing
        + Red → tăng xác suất phishing

    - Với SHAP (BERT):
        + Heatmap token-level
        + Force plot
        + HTML từ shap.plots
    
    - Modal phải:
        + Rõ ràng
        + Load nhanh
        + Có thể scoll để xem toàn bộ

4. Yêu cầu riêng cho SHAP – BERT
Khi backend start:
    - Load BERT model
    - Khởi tạo SHAP Explainer (DeepExplainer hoặc GradientExplainer)
    - Lưu explainer global để không khởi tạo lại

Khi user yêu cầu XAI:
    - Tính SHAP values cho từng token
    - Xuất HTML highlight token theo mức đóng góp
    - Trả HTML cho frontend

5. Tính năng thông minh cần có
    - Không thay đổi giao diện Gmail/Outlook.
    - DOM extraction → không cần Gmail API.
    - Dự đoán 5 model song song → phản hồi < 1–2 giây.
    - Majority vote: Nếu ≥ 3/5 models dự đoán phishing → cảnh báo mạnh.
    - Cho phép:
        + Mở XAI trong tab mới
        + Xuất PDF XAI cho báo cáo đồ án\

6. Tính ổn định & khả năng mở rộng
    - Backend tách biệt → dễ nâng cấp/đổi mô hình.
    - Extension chỉ đọc DOM → không xâm phạm quyền riêng tư.
    - Giao diện đơn giản → dễ maintain.
    - Mở rộng thêm mô hình Ensemble hoặc GPT-based detector trong tương lai.

7. Lợi thế của ý tưởng này
    - Tích hợp đa mô hình → kết quả tin cậy hơn.
    - XAI đầy đủ → giải thích rõ ràng, phù hợp đồ án tốt nghiệp và ngành cybersecurity.
    - Không phụ thuộc Gmail API → chạy mượt, không giới hạn.
    - Dễ trình bày trong hội đồng vì UI đẹp – trực quan – hiện đại.
    - Khả thi trong doanh nghiệp: Có thể dùng làm hệ thống cảnh báo phishing nội bộ.