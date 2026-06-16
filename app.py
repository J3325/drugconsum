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
import graphviz

# Pengaturan halaman tema luas/dashboard
st.set_page_config(page_title="Zat Adiktif Analytics Dashboard", layout="wide")

# ==========================================
# HEADER UTAMA DASHBOARD
# ==========================================
st.markdown("<h1 style='text-align: center; color: #1A5276; margin-bottom: 0;'>Predictive & Anomaly Analytics Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #566573; font-size: 16px; margin-top: 5px;'>Studi Perilaku Konsumsi Zat Berdasarkan Metrik Demografi dan Profil Psikologis (Big Five Personality)</p>", unsafe_allow_html=True)
st.divider()

# ==========================================
# PROSES BACKEND (Latar Belakang)
# ==========================================
@st.cache_data
def load_data():
    kolom = ['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS', 'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    df = pd.read_csv('drug_consumption.data', names=kolom)
    return df

try:
    df = load_data()

    # Preprocessing
    df['Target_Cannabis'] = df['Cannabis'].apply(lambda x: 1 if x not in ['CL0', 'CL1'] else 0)
    fitur_zat = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc','Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD','Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']

    X_class = df.drop(columns=['ID', 'Target_Cannabis'] + fitur_zat)
    y_class = df['Target_Cannabis']

    df_numeric = df.drop(columns=['ID']).copy()
    for col in fitur_zat:
        df_numeric[col] = df_numeric[col].str.replace('CL', '').astype(int)

    # Standarisasi
    scaler = StandardScaler()
    X_class_scaled = scaler.fit_transform(X_class)
    X_anomaly_scaled = scaler.fit_transform(df_numeric)

    # Split Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_class_scaled, y_class, test_size=0.2, random_state=42)

    # Model 1: Decision Tree
    model_dt = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
    model_dt.fit(X_train, y_train)
    y_pred = model_dt.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Model 2: Isolation Forest
    model_if = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    df_numeric['Anomaly_Label'] = model_if.fit_predict(X_anomaly_scaled)

    # ==========================================
    # LAYOUT STRUKTUR UTAMA: SIDEBAR & MAIN PANEL
    # ==========================================

    # 1. Eksplorasi Data Singkat (Bagian Atas Dashboard)
    st.subheader("Data Overview")
    col_d1, col_d2 = st.columns([1, 3])
    with col_d1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric(label="Total Observasi Data", value=df.shape[0])
        st.metric(label="Total Atribut/Fitur", value=df.shape[1])
    with col_d2:
        st.write("Cuplikan Dataset Mentah (UCI Repository):")
        st.dataframe(df.head(4), use_container_width=True)

    st.divider()

    # 2. Pembagian Dua Tab Utama Algoritma
    tab_klasifikasi, tab_anomali = st.tabs(["🎯 Predictive Model (Decision Tree)", "🕵️ Anomaly Detection (Isolation Forest)"])

    # ---------------------------------------------------------
    # TAB 1: PREDICTIVE MODEL (KLASIFIKASI)
    # ---------------------------------------------------------
    with tab_klasifikasi:
        st.subheader("Analisis Klasifikasi Risiko Penggunaan Cannabis")

        # Grid Atas: Angka Metrik & Tabel Evaluasi
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            st.metric(label="Akurasi Pengujian Model", value=f"{accuracy_score(y_test, y_pred)*100:.2f}%")

            # Formating classification report menjadi DataFrame rapi
            report_dict = classification_report(y_test, y_pred, target_names=['Non-User', 'User'], output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose().round(2)

            st.write("Metrik Deteksi per Kelas:")
            st.table(report_df.iloc[:-3, :-1])

        with col_m2:
            st.write("Akurasi Distribusi Tebakan (Confusion Matrix):")
            fig1, ax1 = plt.subplots(figsize=(6, 3.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Prediksi Non-User', 'Prediksi User'],
                        yticklabels=['Aktual Non-User', 'Aktual User'], ax=ax1)
            plt.tight_layout()
            st.pyplot(fig1)
            st.info("Cara Membaca: Fokus pada diagonal utama warna gelap. Semakin tinggi angka keselarasan aktual vs prediksi, menandakan tingkat presisi model yang tinggi.")

        st.divider()

        # Grid Bawah: Pohon Keputusan
        st.write("#### Struktur Pohon Aturan Keputusan Model:")
        fig2, ax2 = plt.subplots(figsize=(16, 7))
        tree.plot_tree(model_dt, feature_names=X_class.columns, class_names=['Non-User', 'User'], filled=True, rounded=True, fontsize=8, ax=ax2)
        plt.tight_layout()
        st.pyplot(fig2)
        st.info("Cara Membaca Alur: Setiap percabangan mewakili kondisi fitur psikologis atau demografi tertentu yang memisahkan probabilitas objek menuju label akhir (Oranye = Non-User, Biru = User).")

    # ---------------------------------------------------------
    # TAB 2: ANOMALY DETECTION
    # ---------------------------------------------------------
    with tab_anomali:
        st.subheader("Deteksi Perilaku Menyimpang / Langka Ekstrem")

        anomali_count = (df_numeric['Anomaly_Label'] == -1).sum()
        normal_count = (df_numeric['Anomaly_Label'] == 1).sum()
        total_data = normal_count + anomali_count

        # Grid Atas: Ringkasan Angka Anomali
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            st.metric(label="Populasi Berperilaku Wajar", value=normal_count)
        with col_a2:
            st.metric(label="Populasi Terdeteksi Anomali", value=anomali_count)
        with col_a3:
            st.metric(label="Tingkat Kontaminasi Dataset", value=f"{(anomali_count/total_data)*100:.2f}%")

        st.divider()

        # Grid Bawah: Visualisasi Klaster PCA
        col_g1, col_g2 = st.columns([2, 1])
        with col_g1:
            st.write("Peta Sebaran Klaster Menggunakan Reduksi PCA 2D:")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_anomaly_scaled)

            fig3, ax3 = plt.subplots(figsize=(8, 4.5))
            ax3.scatter(X_pca[df_numeric['Anomaly_Label'] == 1, 0], X_pca[df_numeric['Anomaly_Label'] == 1, 1], color='#3498DB', alpha=0.4, label='Normal (Inliers)')
            ax3.scatter(X_pca[df_numeric['Anomaly_Label'] == -1, 0], X_pca[df_numeric['Anomaly_Label'] == -1, 1], color='#E74C3C', marker='x', s=70, label='Anomali (Outliers)')
            ax3.set_xlabel("Komponen Utama Analisis 1")
            ax3.set_ylabel("Komponen Utama Analisis 2")
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig3)

        with col_g2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.info("""
            **Interpretasi Klaster Anomali:**

            Algoritma *Isolation Forest* berhasil mendeteksi dan mengisolasi objek (titik silang merah) yang berada jauh dari pusat kerumunan data normal (titik biru).

            Objek-objek merah ini mengindikasikan kasus individu yang memiliki rekam jejak psikologis kontradiktif atau tingkat konsumsi silang multitoksin yang sangat ekstrem/langka di dalam data riil penelitian populasi ini.
            """)

except FileNotFoundError:
    st.error("File data tidak ditemukan. Harap pastikan file 'drug_consumption.data' berada di direktori yang sama dengan berkas skrip ini.")
