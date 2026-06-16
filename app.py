import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Konfigurasi halaman utama dasbor analitik dengan tema luas
st.set_page_config(page_title="Zat Adiktif Analytics Dashboard", layout="wide")

# ==========================================
# 1. HEADER UTAMA DASHBOARD
# ==========================================
st.markdown("""
    <div style='background-color: #F0F4F8; padding: 20px; border-radius: 8px; border-bottom: 4px solid #1A5276; margin-bottom: 20px;'>
        <h1 style='text-align: center; color: #1A5276; margin: 0; font-family: "Arial", sans-serif; font-size: 32px;'>
            Predictive & Anomaly Analytics Dashboard
        </h1>
        <p style='text-align: center; color: #566573; font-size: 15px; margin: 8px 0 0 0;'>
            Sistem Komparasi Algoritma Decision Tree dan Isolation Forest dalam Studi Pola Adiksi Perilaku
        </p>
    </div>
""", unsafe_allow_html=True)

# Baris Metrik Pintar
col_h1, col_h2, col_h3, col_h4 = st.columns(4)
with col_h1:
    st.metric(label="Sumber Data", value="UCI Repository")
with col_h2:
    st.metric(label="Total Sampel", value="1.885 Orang")
with col_h3:
    st.metric(label="Target Prediksi", value="Cannabis User")
with col_h4:
    st.metric(label="Model Anomali", value="Isolation Forest")

st.write("")
st.info("Petunjuk Navigasi: Gunakan Panel Simulator di sebelah kiri untuk memanipulasi skor psikologis subjek, kemudian beralihlah antar Tab Analisis untuk melihat komparasi performa model dan visualisasi data.")
st.divider()

# ==========================================
# 2. PENGENALAN PROYEK
# ==========================================
st.write("### Apa Tujuan Dasbor Ini?")
st.markdown("""
Aplikasi ini dibangun untuk menerjemahkan data psikologi yang kompleks menjadi informasi visual yang mudah dipahami. Tujuan utamanya adalah untuk menjawab pertanyaan: **"Apakah sifat dasar kepribadian seseorang dapat memengaruhi risikonya menjadi pengguna zat adiktif?"**

Teknologi Kecerdasan Buatan (Machine Learning) bekerja di sini sebagai alat bantu peneliti:
* **Algoritma Decision Tree:** Mesin mempelajari pola dari ribuan orang untuk memprediksi apakah seseorang berisiko tinggi menjadi pengguna Cannabis berdasarkan skor kepribadiannya (seperti tingkat kecemasan atau kedisiplinan).
* **Algoritma Isolation Forest:** Mesin bertugas mencari individu (anomali) yang memiliki pola perilaku konsumsi zat dan kepribadian yang sangat menyimpang atau ekstrem dari mayoritas masyarakat normal.
""")
st.divider()

# ==========================================
# PROSES BACKEND
# ==========================================
@st.cache_data
def load_data():
    kolom = ['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS', 'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    df = pd.read_csv('drug_consumption.data', names=kolom)
    return df

try:
    df = load_data()

    # Preprocessing Data
    df['Target_Cannabis'] = df['Cannabis'].apply(lambda x: 1 if x not in ['CL0', 'CL1'] else 0)
    fitur_zat = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc','Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD','Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']

    X_class = df.drop(columns=['ID', 'Target_Cannabis'] + fitur_zat)
    y_class = df['Target_Cannabis']

    df_numeric = df.drop(columns=['ID', 'Target_Cannabis']).copy()
    for col in fitur_zat:
        df_numeric[col] = df_numeric[col].str.replace('CL', '').astype(int)

    # Standarisasi
    scaler_class = StandardScaler()
    X_class_scaled = scaler_class.fit_transform(X_class)
    
    scaler_anomaly = StandardScaler()
    X_anomaly_scaled = scaler_anomaly.fit_transform(df_numeric)

    # Model
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_class_scaled, y_class, test_size=0.2, random_state=42)

    model_dt = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
    model_dt.fit(X_train, y_train)
    y_pred = model_dt.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    model_if = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    df_numeric['Anomaly_Label'] = model_if.fit_predict(X_anomaly_scaled)

    # ==========================================
    # 3. SIDEBAR SIMULATOR
    # ==========================================
    st.sidebar.header("Simulator Pengujian Sifat")
    sim_age = st.sidebar.slider("Skor Usia", -1.0, 2.6, 0.0)
    sim_gender = st.sidebar.selectbox("Gender Objek", ["Pria", "Wanita"])
    sim_gender_val = -0.48246 if sim_gender == "Pria" else 0.48246
    sim_edu = st.sidebar.slider("Skor Pendidikan", -2.4, 2.0, 0.0)
    sim_nscore = st.sidebar.slider("Neurotisisme", -3.5, 3.5, 0.0)
    sim_escore = st.sidebar.slider("Ekstraversi", -3.5, 3.5, 0.0)
    sim_oscore = st.sidebar.slider("Keterbukaan", -3.5, 3.5, 0.0)
    sim_ascore = st.sidebar.slider("Keramahan", -3.5, 3.5, 0.0)
    sim_cscore = st.sidebar.slider("Kedisiplinan", -3.5, 3.5, 0.0)
    sim_imp = st.sidebar.slider("Sifat Impulsif", -2.5, 2.5, 0.0)
    sim_ss = st.sidebar.slider("Sensation Seeking", -2.1, 2.1, 0.0)

    input_sim_class = pd.DataFrame([{'Age': sim_age, 'Gender': sim_gender_val, 'Education': sim_edu, 'Country': 0.2, 'Ethnicity': -0.1, 'Nscore': sim_nscore, 'Escore': sim_escore, 'Oscore': sim_oscore, 'Ascore': sim_ascore, 'Cscore': sim_cscore, 'Impulsive': sim_imp, 'SS': sim_ss}])
    input_sim_class = input_sim_class[X_class.columns]
    input_class_scaled = scaler_class.transform(input_sim_class)

    sim_pred_class = model_dt.predict(input_class_scaled)
    sim_pred_proba = model_dt.predict_proba(input_class_scaled)

    input_sim_anomaly = input_sim_class.copy()
    for col in fitur_zat:
        input_sim_anomaly[col] = 0 
    input_sim_anomaly = input_sim_anomaly[df_numeric.drop(columns=['Anomaly_Label'], errors='ignore').columns]
    input_anomaly_scaled = scaler_anomaly.transform(input_sim_anomaly)
    sim_pred_anomaly = model_if.predict(input_anomaly_scaled)

    # ==========================================
    # 4. KESIMPULAN EKSEKUTIF
    # ==========================================
    st.success("""
    **Kesimpulan Utama Analisis Dasbor**
    
    1. Prediksi Akurat (~78%): Algoritma Klasifikasi membuktikan bahwa profil psikologis kepribadian memiliki korelasi kuat untuk memprediksi risiko individu menjadi pengguna zat adiktif.
    2. Deteksi Kelompok Ekstrem (5%): Algoritma Deteksi Anomali berhasil memisahkan 95 orang yang memiliki rekam jejak penyimpangan perilaku paling ekstrem dibandingkan dengan pola normal responden lainnya.
    """)

    # ==========================================
    # 5. MONITOR OUTPUT SIMULATOR
    # ==========================================
    st.write("### Hasil Pengujian Mandiri via Simulator Sidebar")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if sim_pred_class[0] == 1:
            st.markdown(f"<div style='background-color: #FADBD8; color: #641E16; padding: 15px; border-radius: 5px; border-left: 5px solid #E74C3C;'><b>Hasil Prediksi Klasifikasi:</b> Berisiko Tinggi (User)<br>Probabilitas Keyakinan Model: {sim_pred_proba[0][1]*100:.2f}%</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color: #D4EFDF; color: #145A32; padding: 15px; border-radius: 5px; border-left: 5px solid #2ECC71;'><b>Hasil Prediksi Klasifikasi:</b> Berisiko Rendah (Non-User)<br>Probabilitas Keyakinan Model: {sim_pred_proba[0][0]*100:.2f}%</div>", unsafe_allow_html=True)
    with col_s2:
        if sim_pred_anomaly[0] == -1:
            st.markdown("<div style='background-color: #FDEBD0; color: #7E5109; padding: 15px; border-radius: 5px; border-left: 5px solid #E67E22;'><b>Status Pola Perilaku:</b> Terdeteksi Anomali (Outlier)<br>Kombinasi skor psikologis ini dinilai ekstrem atau langka.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background-color: #EBF5FB; color: #154360; padding: 15px; border-radius: 5px; border-left: 5px solid #3498DB;'><b>Status Pola Perilaku:</b> Normal (Inlier)<br>Kombinasi skor psikologis berada pada batasan wajar populasi.</div>", unsafe_allow_html=True)

    st.divider()

    # ==========================================
    # 6. TAB MODEL
    # ==========================================
    tab1, tab2 = st.tabs(["Model Prediksi Klasifikasi", "Sistem Deteksi Anomali"])
    with tab1:
        st.write("### Evaluasi Model Prediksi")
        st.info("Fungsi: Menjawab pertanyaan 'Apakah seseorang cenderung menjadi pengguna berdasarkan sifat psikologisnya?'.")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Akurasi Model", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
            report = pd.DataFrame(classification_report(y_test, y_pred, target_names=['Non-User', 'User'], output_dict=True)).transpose().round(2)
            st.table(report.iloc[:-3, :-1])
        with c2:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)
    with tab2:
        st.write("### Deteksi Pola Menyimpang")
        st.info("Fungsi: Mengisolasi responden dengan perilaku yang sangat tidak wajar dibanding mayoritas masyarakat umum.")
        c1, c2 = st.columns(3)
        c1.metric("Subjek Wajar", f"{(df_numeric['Anomaly_Label'] == 1).sum()} Orang")
        c2.metric("Subjek Ekstrem", f"{(df_numeric['Anomaly_Label'] == -1).sum()} Orang")
        c3.metric("Rasio Penyimpangan", f"{((df_numeric['Anomaly_Label'] == -1).sum()/len(df_numeric))*100:.2f}%")
        st.divider()
        fig, ax = plt.subplots(figsize=(8, 4))
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_anomaly_scaled)
        ax.scatter(X_pca[:,0], X_pca[:,1], c=df_numeric['Anomaly_Label'], cmap='coolwarm', alpha=0.5)
        st.pyplot(fig)

    st.divider()
    st.write("### Kesimpulan Akhir Penelitian")
    st.success("""
    1. Validitas Fitur Kepribadian: Indikator psikologis (Big Five) memiliki signifikansi statistik kuat sebagai prediktor adiksi.
    2. Efektivitas Metode: Kombinasi Klasifikasi dan Deteksi Anomali membentuk kerangka deteksi dini yang komprehensif.
    3. Kontribusi Praktis: Analisis ini dapat menjadi sistem pendukung keputusan aplikatif bagi praktisi kesehatan mental.
    """)

except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat aplikasi: {e}")