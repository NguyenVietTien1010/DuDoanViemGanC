
# 🧠 Ứng dụng Học Máy – Dự đoán Hành Vi & Viêm Gan C

Dự án bao gồm **hai ứng dụng chính** sử dụng Streamlit kết hợp học máy và biểu diễn logic để:
- 🤖 Dự đoán hành vi con người từ thời tiết và cảm xúc
- 🩺 Dự đoán nguy cơ mắc Viêm Gan C dựa vào các chỉ số sinh hóa

---

## 📁 Nội dung dự án

| Tên file       | Mô tả ngắn |
|----------------|------------|
| `main.py` / `main2.py` | Ứng dụng Trợ lý Logic hành vi (MLP + Logic biểu diễn) |
| `demo.py`      | Ứng dụng dự đoán bệnh Viêm Gan C |
| `data.csv`, `data2.csv` | Dữ liệu mô tả tình huống (main.py) |
| `HepatitisCdata.csv` | Dữ liệu bệnh học (demo.py) |
| `mlp_model.pth`, `losses.pkl` | Mô hình và log huấn luyện |
| `requirements.txt` | Danh sách thư viện cần cài |

---

## 🚀 Cách chạy ứng dụng

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng Trợ lý hành vi logic

```bash
streamlit run main.py
# Hoặc bản có biểu đồ loss:
streamlit run main2.py
```

### 3. Chạy ứng dụng Dự đoán Viêm Gan C

```bash
streamlit run demo.py
```

Truy cập tại `http://localhost:8501` trong trình duyệt.

---

## 🤖 Ứng dụng 1 – Trợ Lý Logic Hành Vi

- Huấn luyện MLP từ dữ liệu mô tả thời tiết & cảm xúc
- Trích xuất đặc trưng logic: AND, OR, XOR
- Dự đoán hành động (đi chơi, nghỉ ngơi) và giải thích theo biểu thức logic
- Giao diện trực quan, có biểu đồ thống kê và Graphviz mô phỏng mạng nơ-ron

---

## 🩺 Ứng dụng 2 – Dự đoán Viêm Gan C

- Dữ liệu: `HepatitisCdata.csv` với >10 chỉ số sinh hóa
- Tiền xử lý bằng StandardScaler, PCA, KMeans
- Mô hình: KNN và Random Forest
- Khai phá luật Apriori + giải thích y học
- Phân tích chỉ số sinh hóa bằng biểu đồ Plotly
- So sánh độ chính xác, precision, recall, F1-score

---

## 📊 Hiệu suất mô hình

| Thuật toán       | Accuracy | Precision | Recall | F1-score |
|------------------|----------|-----------|--------|----------|
| Random Forest    | ~97%     | ~95%      | ~95%   | ~95%     |
| KNN              | ~94%     | ~91%      | ~91%   | ~91%     |

---

## 👨‍⚕️ Ý nghĩa chỉ số y khoa

Ứng dụng cung cấp giải thích ngắn gọn từng chỉ số như ALT, AST, GGT, BIL, ALB… để người dùng hiểu rõ hơn về tình trạng gan của mình.

---

## 📌 Mục tiêu

- Minh họa kết hợp giữa AI + logic + y học
- Dễ triển khai, giao diện đẹp, tiếng Việt đầy đủ
- Phù hợp cho sinh viên học các môn ML, NLP, AI y tế

---

## 👤 Tác giả

Nguyễn Viết Tiến – Đồ án học phần Khoa học Dữ liệu  
Hướng dẫn sử dụng, tài liệu chi tiết có trong mỗi ứng dụng.

