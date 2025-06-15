import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
from sklearn.decomposition import PCA
import random

st.set_page_config(page_title="Dự đoán Viêm Gan C", layout="wide")
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f9fbfc;
    }
    .block-container {
        padding: 2rem 3rem 2rem 3rem;
    }
    .stButton button {
        border-radius: 0.5rem;
        background-color: #e3f2fd;
        color: #0d47a1;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #e3f2fd;
        padding: 0.25rem;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #0d47a1;
        font-weight: bold;
    }
    [data-baseweb="radio"] > div {
        gap: 0.5rem;
        padding-top: 0.3rem;
    }
    [data-baseweb="radio"] label {
        background-color: #e3f2fd;
        padding: 6px 14px;
        border-radius: 10px;
        font-weight: 500;
        color: #0d47a1;
        border: 1px solid #90caf9;
        transition: all 0.2s ease;
    }
    [data-baseweb="radio"] input:checked + div {
        font-weight: bold !important;
        color: #0d47a1;
    }
    </style>
""", unsafe_allow_html=True)
st.title("🩺 Ứng dụng Dự đoán Viêm Gan C")

# Load dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv("HepatitisCdata.csv")
    df = df.dropna()
    df = df[df['Category'] != '']
    df = df[df['Sex'].isin(['m', 'f'])]
    df['Target'] = df['Category'].apply(lambda x: 0 if 'Blood Donor' in x else 1)
    # Cập nhật nhãn nếu chỉ số nguy cơ cao
    high_risk = (df['ALT'] > 100) | (df['AST'] > 100) | (df['BIL'] > 3) | (df['GGT'] > 100)
    df.loc[high_risk & (df['Target'] == 0), 'Target'] = 1
    return df

df = load_data()

# Tiền xử lý
label_enc = LabelEncoder()
df['Sex'] = label_enc.fit_transform(df['Sex'])

# Tạo cờ nguy cơ cao dựa trên ngưỡng y khoa
threshold_flags = ((df['ALT'] > 100) | (df['AST'] > 100) | (df['BIL'] > 3) | (df['GGT'] > 100)).astype(int)
df['Flag_High_Risk'] = threshold_flags

numerical_cols = ['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

key_features = ['ALT', 'AST', 'BIL', 'GGT', 'ALB', 'ALP']
X = df[key_features + ['Sex', 'Flag_High_Risk']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Gán trọng số cho mẫu nguy cơ cao
sample_weights = (X_train['Flag_High_Risk'] * 2 + 1)

# Huấn luyện model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model_knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
model_rf.fit(X_train, y_train, sample_weight=sample_weights)
model_knn.fit(X_train, y_train)

# Trực quan độ quan trọng đặc trưng với Random Forest
importances = model_rf.feature_importances_
feature_names = X.columns  # sau khi chọn feature chính
feature_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)

def evaluate_model(model):
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    }

metrics_knn = evaluate_model(model_knn)
metrics_rf = evaluate_model(model_rf)

# KMeans để phân cụm nguy cơ
kmeans_features = df.select_dtypes(include=[np.number]).drop(columns=['Target'])
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['Risk_Cluster'] = kmeans.fit_predict(kmeans_features)
# ==== Tabs chính ====
tabs = st.tabs(["📄 Dữ liệu", "🔍 Luật Apriori", "🧠 Dự đoán", "📊 Trực quan","📈 So sánh Mô Hình"])


with tabs[0]:
    st.subheader("📁 Dữ liệu gốc")
    st.dataframe(df)

    st.subheader("📖 Bảng ý nghĩa các chỉ số")
    explanation = {
        "Chỉ số": ["ALT", "AST", "GGT", "Bilirubin", "Albumin", "ALP", "CHE", "CHOL", "CREA", "Protein tổng"],
        "Ý nghĩa": [
            "ALT là enzyme chủ yếu trong gan, tăng cao chỉ ra tổn thương gan do viêm hoặc xơ hóa.",
            "AST là enzyme giúp đánh giá tổn thương gan và các mô khác, thường tăng cùng ALT.",
            "GGT cao thường liên quan đến tổn thương gan do rượu hoặc nhiễm độc.",
            "Bilirubin cao có thể gây vàng da, là dấu hiệu của suy giảm chức năng gan.",
            "Albumin do gan sản xuất, mức thấp có thể báo hiệu suy gan.",
            "ALP tăng cao có thể liên quan đến tắc nghẽn đường mật.",
            "CHE giảm liên quan đến các bất thường về chức năng gan.",
            "Cholesterol có thể bất thường khi gan không hoạt động tốt.",
            "Creatinine thường liên quan đến chức năng thận, theo dõi cùng các chỉ số gan.",
            "Protein tổng quan thể hiện tình trạng dinh dưỡng và chức năng gan."
        ]
    }
    st.table(pd.DataFrame(explanation))


    st.subheader("📊 Phân cụm KMeans (toàn bộ chỉ số) - Trực quan bằng PCA")

    features = kmeans_features.values
    pca = PCA(n_components=2)
    components = pca.fit_transform(features)

    df_pca = pd.DataFrame({
        'PC1': components[:, 0],
        'PC2': components[:, 1],
        'Cluster': df['Risk_Cluster'].astype(int).astype(str), 
        'Target': df['Target'].map({0: 'Không bệnh', 1: 'Có bệnh'})
    })

    # Lọc nếu có nhãn khác ngoài '0' và '1'
    df_pca = df_pca[df_pca['Cluster'].isin(['0', '1'])]

    fig = px.scatter(
        df_pca,
        x='PC1',
        y='PC2',
        color='Cluster',
        symbol='Target',
        title='Phân cụm KMeans dựa trên toàn bộ chỉ số (PCA 2D)',
        labels={'PC1': 'Thành phần chính 1', 'PC2': 'Thành phần chính 2', 'Cluster': 'Nhóm nguy cơ'}
    )
    st.plotly_chart(fig, use_container_width=True)



with tabs[1]:
    st.subheader("🔍 Khai phá luật Apriori")

    # === Tính toán tập phổ biến và luật kết hợp ===
    apriori_data = df[numerical_cols + ['Flag_High_Risk']] > 0
    min_support = st.slider("📊 Ngưỡng support", 0.01, 0.5, 0.1, 0.01)
    min_conf = st.slider("📈 Ngưỡng confidence", 0.1, 1.0, 0.6, 0.05)

    frequent_itemsets = apriori(apriori_data, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)

    if not rules.empty:
        # ===== Hiển thị bảng luật gốc =====
        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ', '.join(map(str, x)))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: ', '.join(map(str, x)))
        st.dataframe(
            rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']]
            .rename(columns={'antecedents_str': 'antecedents', 'consequents_str': 'consequents'})
        )

        # ===== Dropdown lọc phần giải thích luật =====
        all_items = sorted(set(
            item for rule in rules.itertuples(index=False)
            for item in rule.antecedents.union(rule.consequents)
        ))
        selected_filter = st.selectbox("🧪 Lọc phần giải thích luật theo chỉ số:", ["Tất cả"] + all_items)

        st.markdown("## 📌 Giải thích từng luật Apriori")

        for idx, row in rules.iterrows():
            antecedents = ', '.join(map(str, row['antecedents']))
            consequents = ', '.join(map(str, row['consequents']))

            # Chỉ hiện nếu lọc đúng
            if selected_filter == "Tất cả" or selected_filter in row['antecedents'] or selected_filter in row['consequents']:
                with st.expander(f"📎 Luật #{idx+1}: {antecedents} → {consequents}"):
                    st.markdown(f"- ✅ **Confidence**: `{row['confidence']:.2f}`")
                    st.markdown(f"- 🚀 **Lift**: `{row['lift']:.2f}`")
                    st.markdown("### 🧠 Giải thích y khoa:")

                    for biomarker in row['antecedents']:
                        if biomarker == 'ALT':
                            st.markdown("- **ALT** tăng cao → tổn thương tế bào gan (viêm gan, xơ gan, viêm gan virus).")
                        elif biomarker == 'AST':
                            st.markdown("- **AST** tăng → cùng ALT là viêm gan, nếu AST > ALT → có thể do rượu.")
                        elif biomarker == 'BIL':
                            st.markdown("- **Bilirubin** cao → vàng da, gan không xử lý tốt sắc tố mật.")
                        elif biomarker == 'GGT':
                            st.markdown("- **GGT** tăng → gan nhiễm độc (rượu, thuốc), hoặc tắc mật.")
                        elif biomarker == 'ALB':
                            st.markdown("- **Albumin** thấp → gan suy giảm khả năng tổng hợp protein.")
                        elif biomarker == 'CHE':
                            st.markdown("- **Cholinesterase** thấp → xơ gan, ung thư gan.")
                        elif biomarker == 'CHOL':
                            st.markdown("- **Cholesterol** cao → liên quan đến gan nhiễm mỡ.")
                        elif biomarker == 'CREA':
                            st.markdown("- **Creatinine** cao → chức năng thận yếu, thường kèm xơ gan mất bù.")
                        elif biomarker == 'ALP':
                            st.markdown("- **ALP** tăng → tắc nghẽn mật hoặc u gan.")
                        elif biomarker == 'PROT':
                            st.markdown("- **Protein tổng** thấp → dinh dưỡng kém hoặc gan yếu.")
    else:
        st.warning("⚠️ Không tìm thấy luật phù hợp với các ngưỡng đã chọn.")


with tabs[2]:
    st.subheader("🧠 Dự đoán nguy cơ Viêm Gan C")

    model_choice = st.selectbox("🔧 Chọn mô hình dự đoán", ["Random Forest", "KNN"])
    preset = st.radio("📋 Chọn dữ liệu mẫu", ["Nguy cơ thấp", "Nguy cơ cao", "Ngẫu nhiên"], index=0)

    if preset == "Nguy cơ thấp":
        values = dict(age=35, alb=45, alp=78, alt=25, ast=22, bil=0.8, che=8.0, chol=5.2, crea=1.0, ggt=30, prot=70, sex="Male")
    elif preset == "Nguy cơ cao":
        values = dict(age=65, alb=28, alp=220, alt=120, ast=105, bil=3.5, che=3.8, chol=6.5, crea=1.5, ggt=180, prot=83, sex="Male")
    else:

        values = dict(
            age=random.randint(20, 80),
            alb=random.uniform(25, 50),
            alp=random.uniform(50, 300),
            alt=random.uniform(10, 150),
            ast=random.uniform(10, 150),
            bil=random.uniform(0.3, 4),
            che=random.uniform(3.5, 10),
            chol=random.uniform(3, 8),
            crea=random.uniform(0.6, 1.8),
            ggt=random.uniform(10, 180),
            prot=random.uniform(55, 85),
            sex=random.choice(["Male", "Female"])
        )

    st.info("🌀 Lưu ý: Dữ liệu ngẫu nhiên chỉ tạo khi bạn nhấn nút 'Tạo lại ngẫu nhiên'.")

    rerun_trigger = st.button("🔄 Tạo lại ngẫu nhiên") if preset == "Ngẫu nhiên" else False

    if rerun_trigger:
        st.session_state.random_trigger = not st.session_state.get("random_trigger", False)

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Tuổi", 0, 100, int(values["age"]))
            alt = st.number_input("ALT", 0.0, 200.0, float(values["alt"]))
            ast = st.number_input("AST", 0.0, 200.0, float(values["ast"]))
            bil = st.number_input("Bilirubin", 0.0, 10.0, float(values["bil"]))
            ggt = st.number_input("GGT", 0.0, 200.0, float(values["ggt"]))
            alb = st.number_input("Albumin", 10.0, 60.0, float(values["alb"]))
        with col2:
            alp = st.number_input("ALP", 20.0, 400.0, float(values["alp"]))
            che = st.number_input("CHE", 1.0, 13.0, float(values["che"]))
            chol = st.number_input("Cholesterol", 2.0, 10.0, float(values["chol"]))
            crea = st.number_input("Creatinine", 0.5, 2.0, float(values["crea"]))
            prot = st.number_input("Protein", 40.0, 100.0, float(values["prot"]))
            sex = st.selectbox("Giới tính", ["Male", "Female"], index=0 if values['sex'] == "Male" else 1)
        submitted = st.form_submit_button("📊 Dự đoán")

    if submitted:
        encoded_sex = label_enc.transform([sex.lower()[0]])[0]
        input_raw = pd.DataFrame([{ "Age": age, "ALB": alb, "ALP": alp, "ALT": alt, "AST": ast, "BIL": bil, "CHE": che, "CHOL": chol, "CREA": crea, "GGT": ggt, "PROT": prot, "Sex": encoded_sex }])

        original_input = input_raw.copy()
        input_scaled = scaler.transform(input_raw[numerical_cols])
        input_data = pd.DataFrame(input_scaled, columns=numerical_cols)

        input_data['Sex'] = encoded_sex
        input_data['Flag_High_Risk'] = int((alt > 100) or (ast > 100) or (bil > 3) or (ggt > 100))
        input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

        model = model_rf if model_choice == "Random Forest" else model_knn
        prediction_proba = model.predict_proba(input_data)[0] if hasattr(model, "predict_proba") else [1 - model.predict(input_data)[0], model.predict(input_data)[0]]
        prediction = model.predict(input_data)[0]

        if prediction == 0:
            st.success("✅ Kết quả: Không có dấu hiệu viêm gan C")
        else:
            st.warning("⚠️ Kết quả: Có khả năng viêm gan C. Hãy kiểm tra chuyên sâu.")

        st.subheader("📉 Tỉ lệ dự đoán")
        st.write(f"- Xác suất không bệnh: {prediction_proba[0]*100:.2f}%")
        st.write(f"- Xác suất có bệnh: {prediction_proba[1]*100:.2f}%")

        # ✅ Phân tích Apriori nếu có
        from mlxtend.frequent_patterns import apriori, association_rules
        binarized = (input_raw[numerical_cols] > 0).astype(int)
        fake_df = pd.concat([binarized, pd.DataFrame({'Flag_High_Risk': [input_data['Flag_High_Risk'].iloc[0]]})], axis=1)
        min_support, min_conf = 0.01, 0.6
        frequent_itemsets = apriori(df[numerical_cols + ['Flag_High_Risk']] > 0, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
        matched_rules = rules[rules['antecedents'].apply(lambda x: x.issubset(set(fake_df.columns[fake_df.iloc[0]==1])))]
        matched_rules = matched_rules.sort_values(by="lift", ascending=False).head(5)
        if not matched_rules.empty:
            st.markdown("### 🧠 Luật Apriori liên quan đến hồ sơ của bạn:")
            for idx, row in matched_rules.iterrows():
                antecedents = ', '.join(map(str, row['antecedents']))
                consequents = ', '.join(map(str, row['consequents']))
                st.markdown(f"📌 **{antecedents} → {consequents}**")
                st.markdown(f"- Confidence: `{row['confidence']:.2f}` | Lift: `{row['lift']:.2f}`")

        st.markdown("### 🌱 Tài liệu tham khảo chăm sóc gan:")
        health_links = [
            ("📜 Dinh dưỡng bảo vệ gan", "https://www.vinmec.com/vie/bai-viet/cac-thuc-pham-tot-cho-gan-cua-ban-vi"),
            ("🚫 Các thực phẩm nên tránh", "https://medlatec.vn/tin-tuc/benh-viem-gan-c-kieng-an-gi-va-nen-an-gi-s153-n21189"),
            ("💪 Tập luyện tăng cường sức khỏe gan", "https://thanhnien.vn/5-bai-tap-don-gian-giup-tang-cuong-suc-khoe-gan-18524092514545204.htm"),
            ("🧬 Viêm gan C là gì?", "https://www.vinmec.com/vie/bai-viet/benh-viem-gan-c-la-gi-vi-sao-toi-co-mac-benh-nay-vi")
        ]
        for label, url in health_links:
            st.markdown(f"- [{label}]({url})")

        st.subheader("📋 So sánh chỉ số của bạn với ngưỡng chuẩn")
        ref_ranges = {
            "ALT": (7, 56), "AST": (5, 40), "BIL": (0.3, 1.2), "GGT": (9, 48),
            "ALB": (35, 50), "ALP": (30, 120), "CHE": (5, 12), "CHOL": (3, 5.2),
            "CREA": (60, 110), "PROT": (60, 80)
        }
        compare_rows = []
        for key, (low, high) in ref_ranges.items():
            if key in original_input.columns:
                val = float(original_input[key].iloc[0])
                val_converted = val * 88.4 if key == "CREA" else val
                compare_rows.append({
                    "Chỉ số": key,
                    "Giá trị của bạn": round(val_converted, 2),
                    "Mức bình thường": f"{low} – {high}",
                    "Trạng thái": "⚠️ Ngoài ngưỡng" if val_converted < low or val_converted > high else "✅ Bình thường"
                })
        df_compare = pd.DataFrame(compare_rows)
        st.dataframe(df_compare, use_container_width=True)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=['Không bệnh', 'Có bệnh'], output_dict=True)

        st.subheader("📈 Hiệu suất mô hình")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{report['accuracy']*100:.1f}%")
        col2.metric("Precision", f"{report['Có bệnh']['precision']*100:.1f}%")
        col3.metric("Recall", f"{report['Có bệnh']['recall']*100:.1f}%")
        col4.metric("F1", f"{report['Có bệnh']['f1-score']*100:.1f}%")

with tabs[3]:
    st.subheader("📊 Trực quan chỉ số sinh hóa")

    st.markdown("### 🧪 Biểu đồ Histogram: Phân bố chỉ số giữa hai nhóm")
    st.markdown("""
Biểu đồ Histogram giúp hình dung sự **phân bố giá trị** của từng chỉ số sinh hóa trong hai nhóm:
- 🟢 **Không bệnh**: người bình thường
- 🔴 **Có bệnh**: người có vấn đề gan (xơ gan, viêm gan, v.v.)

**Cách đọc:**
- Trục ngang: giá trị của chỉ số (VD: ALT từ 0 đến cao)
- Trục dọc: số người có giá trị đó
- Nhóm 'có bệnh' thường có cột lệch sang phải (giá trị cao hơn)

**Ý nghĩa:**
→ Nếu nhóm 'có bệnh' có giá trị cao hơn → chỉ số đó có khả năng **dự báo bệnh gan**
    """)

    for col in ['ALT', 'AST', 'BIL', 'GGT']:
        fig = px.histogram(df, x=col, color=df['Target'].map({0: 'Không bệnh', 1: 'Có bệnh'}),
                           barmode="overlay", nbins=30, title=f"📊 Phân bố chỉ số {col}")
        st.plotly_chart(fig)

        if col == 'ALT':
            st.markdown("🔍 **ALT** là enzyme đặc trưng trong gan. Nếu nhóm 'có bệnh' lệch phải → ALT cao là dấu hiệu tổn thương gan.")
        elif col == 'AST':
            st.markdown("🔍 **AST** cũng là enzyme gan. Nếu AST cao ở nhóm bệnh → có thể là viêm gan hoặc tổn thương do rượu.")
        elif col == 'BIL':
            st.markdown("🔍 **Bilirubin** cao gây vàng da. Histogram cho thấy nhóm bệnh có nhiều người có BIL cao → suy giảm chức năng gan.")
        elif col == 'GGT':
            st.markdown("🔍 **GGT** tăng rõ ở nhóm bệnh → dấu hiệu gan nhiễm độc hoặc rối loạn mật.")

    st.markdown("### 📦 Biểu đồ Boxplot: So sánh chỉ số giữa hai nhóm")
    st.markdown("""
Boxplot minh họa sự khác biệt về:
- **Trung vị (median)** của chỉ số
- **Khoảng biến thiên** (từ thấp đến cao)
- Và **giá trị bất thường** (chấm tròn)

**Cách đọc:**
- Hộp càng cao → giá trị càng lớn
- Đường giữa hộp → trung vị
- Nhiều chấm ngoài hộp → chỉ số dao động mạnh, bất thường

**Ý nghĩa:**
→ Nếu hộp nhóm 'có bệnh' cao hơn → chỉ số đó có thể dùng để **chẩn đoán bệnh gan**
    """)

    for col in ['ALT', 'AST', 'BIL', 'GGT']:
        fig = px.box(df, x='Target', y=col, points="all",
                     labels={'Target': 'Tình trạng bệnh', col: col},
                     title=f"📦 So sánh chỉ số {col}")
        fig.update_layout(xaxis=dict(tickvals=[0, 1], ticktext=['Không bệnh', 'Có bệnh']))
        st.plotly_chart(fig)

        # Giải thích từng chỉ số
        if col == 'ALT':
            st.markdown("🔍 **ALT** cao hơn ở nhóm bệnh → dấu hiệu viêm gan phổ biến, dễ nhận diện.")
        elif col == 'AST':
            st.markdown("🔍 **AST** có giá trị cao hơn và nhiều outlier → cảnh báo nguy cơ tổn thương gan mạnh.")
        elif col == 'BIL':
            st.markdown("🔍 **Bilirubin** nhóm bệnh vừa cao vừa nhiều bất thường → cảnh báo gan suy yếu.")
        elif col == 'GGT':
            st.markdown("🔍 **GGT** vượt trội ở nhóm bệnh → liên quan đến gan nhiễm độc và tắc mật.")
# ==== TAB 4: So sánh mô hình ====
with tabs[4]:


    compare_df = pd.DataFrame({
        "Chỉ số": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "KNN": [
            f"{metrics_knn['Accuracy']*100:.2f}%",
            f"{metrics_knn['Precision']*100:.2f}%",
            f"{metrics_knn['Recall']*100:.2f}%",
            f"{metrics_knn['F1']*100:.2f}%"
        ],
        "Random Forest": [
            f"{metrics_rf['Accuracy']*100:.2f}%",
            f"{metrics_rf['Precision']*100:.2f}%",
            f"{metrics_rf['Recall']*100:.2f}%",
            f"{metrics_rf['F1']*100:.2f}%"
        ]
    })

    st.markdown("### 📊 Bảng so sánh chỉ số mô hình")
    st.dataframe(compare_df, use_container_width=True)

    st.markdown("### 📈 Biểu đồ cột trực quan")
    chart_df = compare_df.melt(id_vars="Chỉ số", var_name="Mô hình", value_name="Giá trị")
    chart_df["Giá trị"] = chart_df["Giá trị"].str.replace("%", "").astype(float)

    import plotly.express as px
    fig = px.bar(chart_df, x="Chỉ số", y="Giá trị", color="Mô hình", barmode="group",
                 text="Giá trị", height=400, title="So sánh hiệu suất dự đoán")
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(yaxis=dict(title="%"), xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧠 K-Nearest Neighbors (KNN)")
        st.markdown("**✅ Ưu điểm:**")
        st.markdown("""
        - Dễ hiểu, dễ triển khai
        - Phù hợp với dữ liệu ít, không cần huấn luyện nhiều
        """)
        st.markdown("**⚠️ Nhược điểm:**")
        st.markdown("""
        - Chậm khi dữ liệu lớn
        - Nhạy với dữ liệu nhiễu (outlier)
        - Không giải thích được quá trình dự đoán
        """)

    with col2:
        st.markdown("### 🌲 Random Forest")
        st.markdown("**✅ Ưu điểm:**")
        st.markdown("""
        - Hiệu suất cao, kháng nhiễu tốt
        - Có thể đánh giá tầm quan trọng của thuộc tính
        - Tổng quát tốt trên dữ liệu lớn
        """)
        st.markdown("**⚠️ Nhược điểm:**")
        st.markdown("""
        - Phức tạp hơn, khó giải thích chi tiết từng cây
        - Thời gian huấn luyện lâu hơn một chút
        """)

    st.info("👉 **KNN** phù hợp với bài toán nhỏ, minh bạch, dữ liệu sạch.👉 **Random Forest** phù hợp cho sản phẩm thật, dữ liệu phức tạp.")

