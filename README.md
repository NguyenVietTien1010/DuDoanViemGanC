
# 🩺 Ứng dụng Dự đoán Viêm Gan C bằng Học Máy

Ứng dụng web giúp dự đoán nguy cơ mắc **Viêm Gan C** dựa trên dữ liệu xét nghiệm sinh hóa. Giao diện xây dựng bằng **Streamlit**, sử dụng các mô hình học máy như **Random Forest** và **K-Nearest Neighbors (KNN)**.

---

## 📂 Nội dung dự án

| Tên file              | Mô tả                                                |
|-----------------------|------------------------------------------------------|
| `demo.py`             | Mã nguồn chính của ứng dụng Streamlit               |
| `HepatitisCdata.csv`  | Dữ liệu y khoa gồm các chỉ số sinh hóa              |
| `requirements.txt`    | Danh sách thư viện cần cài đặt                      |

---

## 🚀 Cách chạy ứng dụng

### 1. Cài đặt thư viện cần thiết

```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng

```bash
streamlit run demo.py
```

Ứng dụng sẽ hiển thị tại: `http://localhost:8501`

---

## 🧪 Dữ liệu và tiền xử lý

- Dữ liệu từ file `HepatitisCdata.csv`
- Gồm hơn 10 chỉ số sinh hóa: ALT, AST, GGT, BIL, ALB, ALP, CHE, CHOL, CREA, PROT...
- Nhãn `Target`: 0 (không bệnh), 1 (có bệnh)
- Tiền xử lý gồm: lọc dữ liệu, chuẩn hóa (StandardScaler), mã hóa giới tính, tạo biến `Flag_High_Risk`

---

## 🧠 Mô hình học máy

- **Random Forest** (100 cây, trọng số cân bằng)
- **KNN** (k=5, khoảng cách có trọng số)
- Đánh giá hiệu suất bằng: Accuracy, Precision, Recall, F1-score

---

## 🔍 Khai phá luật Apriori

- Trích xuất các tập phổ biến và luật kết hợp từ dữ liệu
- Hiển thị bảng luật và giải thích y học cho từng chỉ số liên quan

---

## 📊 Trực quan hóa

- Biểu đồ **PCA + KMeans** để phân cụm bệnh nhân
- Biểu đồ **Histogram** và **Boxplot** so sánh các chỉ số sinh hóa giữa hai nhóm
- Bảng **so sánh mô hình** dựa trên các chỉ số hiệu suất

---

## 🧬 Dự đoán cá nhân hóa

- Nhập dữ liệu xét nghiệm theo form
- Dự đoán kết quả có/không mắc bệnh
- Phân tích xác suất, lý do theo luật Apriori
- So sánh chỉ số người dùng với ngưỡng bình thường

---

## 📈 Hiệu suất mô hình

| Mô hình         | Accuracy | Precision | Recall | F1-score |
|-----------------|----------|-----------|--------|----------|
| Random Forest   | ~97%     | ~95%      | ~95%   | ~95%     |
| KNN             | ~94%     | ~91%      | ~91%   | ~91%     |

---

## 🧑‍⚕️ Ứng dụng dành cho

- Sinh viên học ngành Y – Dược – Khoa học dữ liệu
- Các bài toán AI trong y tế
- Người dân muốn hiểu về nguy cơ gan và theo dõi sức khỏe

---

## 👤 Tác giả

Nguyễn Viết Tiến  
Dự án học phần: Ứng dụng Học Máy & Y học Dự phòng  
