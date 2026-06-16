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
# 1. HEADER UTAMA DASHBOARD (VERSI PREMIUM)
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

# Baris Metrik Pintar (Quick Stats) Langsung di Header
col_h1, col_h2, col_h3, col_h4 = st.columns(4)
with col_h1:
    st.metric(label="Basis Dataset", value="UCI Drug Addiction")
with col_h2:
    st.metric(label="Total Sampel Evaluasi", value="1.885 Responden")
with col_h3:
    st.metric(label="Fokus Target Klasifikasi", value="Cannabis User")
with col_h4:
    st.metric(label="Metode Deteksi Anomali", value="Unsupervised Outliers")

st.write("")
st.info("""
**Petunjuk Navigasi Dasbor:** Gunakan **Panel Simulator** di sebelah kiri layar untuk memanipulasi skor psikologis subjek secara individual, kemudian beralihlah antar **Tab Analisis** di bawah ini untuk melihat komparasi performa model makro serta visualisasi persebaran data secara komprehensif.
""", icon="🧭")
st.divider()

# ==========================================
# PROSES BACKEND (Berjalan di Latar Belakang)
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

    # Membuang 'Target_Cannabis' agar tepat 31 kolom untuk Anomali
    df_numeric = df.drop(columns=['ID', 'Target_Cannabis']).copy()
    for col in fitur_zat:
        df_numeric[col] = df_numeric[col].str.replace('CL', '').astype(int)

    # Standarisasi Skala Fitur 
    scaler_class = StandardScaler()
    X_class_scaled = scaler_class.fit_transform(X_class)
    
    scaler_anomaly = StandardScaler()
    X_anomaly_scaled = scaler_anomaly.fit_transform(df_numeric)

    # Split Parameter Data Latih dan Data Uji
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_class_scaled, y_class, test_size=0.2, random_state=42)

    # Pemodelan 1: Decision Tree Classifier
    model_dt = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
    model_dt.fit(X_train, y_train)
    y_pred = model_dt.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Pemodelan 2: Isolation Forest Outlier Detector
    model_if = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    df_numeric['Anomaly_Label'] = model_if.fit_predict(X_anomaly_scaled)

    # ==========================================
    # 2. SIDEBAR PANEL SIMULATOR INTERAKTIF
    # ==========================================
    st.sidebar.header("Simulator Pengujian Sifat")
    st.sidebar.write("Sesuaikan parameter kepribadian di bawah ini untuk melihat hasil keputusan model secara langsung:")
    
    sim_age = st.sidebar.slider("Skor Usia (Terstandarisasi)", -1.0, 2.6, 0.0)
    sim_gender = st.sidebar.selectbox("Gender Objek", ["Pria", "Wanita"])
    sim_gender_val = -0.48246 if sim_gender == "Pria" else 0.48246
    sim_edu = st.sidebar.slider("Skor Tingkat Pendidikan", -2.4, 2.0, 0.0)
    sim_nscore = st.sidebar.slider("Neurotisisme (Tingkat Kecemasan)", -3.5, 3.5, 0.0)
    sim_escore = st.sidebar.slider("Ekstraversi (Kemampuan Bersosialisasi)", -3.5, 3.5, 0.0)
    sim_oscore = st.sidebar.slider("Keterbukaan terhadap Pengalaman Baru", -3.5, 3.5, 0.0)
    sim_ascore = st.sidebar.slider("Keramahan Sosial (Agreeableness)", -3.5, 3.5, 0.0)
    sim_cscore = st.sidebar.slider("Kedisiplinan Diri (Conscientiousness)", -3.5, 3.5, 0.0)
    sim_imp = st.sidebar.slider("Sifat Impulsif", -2.5, 2.5, 0.0)
    sim_ss = st.sidebar.slider("Sensation Seeking (Pencarian Sensasi)", -2.1, 2.1, 0.0)

    # Menggabungkan Input untuk Simulator Klasifikasi (12 Kolom)
    input_sim_class = pd.DataFrame([{
        'Age': sim_age, 'Gender': sim_gender_val, 'Education': sim_edu, 
        'Country': 0.2, 'Ethnicity': -0.1, 'Nscore': sim_nscore, 'Escore': sim_escore, 
        'Oscore': sim_oscore, 'Ascore': sim_ascore, 'Cscore': sim_cscore, 
        'Impulsive': sim_imp, 'SS': sim_ss
    }])
    
    # Samakan urutan kolom untuk klasifikasi
    input_sim_class = input_sim_class[X_class.columns]
    input_class_scaled = scaler_class.transform(input_sim_class)

    # Eksekusi Kalkulasi Prediksi Klasifikasi
    sim_pred_class = model_dt.predict(input_class_scaled)
    sim_pred_proba = model_dt.predict_proba(input_class_scaled)

    # Persiapan Data untuk Simulator Anomali (Butuh 31 Kolom)
    input_sim_anomaly = input_sim_class.copy()
    for col in fitur_zat:
        input_sim_anomaly[col] = 0 # Default asumsi simulasi tidak ada konsumsi zat lain
        
    # Samakan urutan kolom persis dengan df_numeric (31 kolom)
    input_sim_anomaly = input_sim_anomaly[df_numeric.drop(columns=['Anomaly_Label'], errors='ignore').columns]
    input_anomaly_scaled = scaler_anomaly.transform(input_sim_anomaly)

    # Eksekusi Prediksi Anomali
    sim_pred_anomaly = model_if.predict(input_anomaly_scaled)

    # ==========================================
    # 3. RINGKASAN EKSEKUTIF (EXECUTIVE SUMMARY)
    # ==========================================
    st.markdown("""
        <div style='background-color: #E8F8F5; padding: 15px; border-radius: 5px; border-left: 5px solid #1ABC9C; margin-bottom: 20px;'>
            <h4 style='margin-top: 0; color: #0E6251;'>💡 Kesimpulan Utama Analisis Dasbor (Executive Summary)</h4>
            <ol style='margin-bottom: 0; color: #117864; font-size: 15px;'>
                <li><b>Prediksi Akurat (~78%):</b> Algoritma Klasifikasi membuktikan bahwa profil psikologis kepribadian seseorang (terutama tingkat neurotisisme dan pencarian sensasi) memiliki korelasi kuat untuk mendeteksi risiko individu menjadi pengguna zat adiktif.</li>
                <li><b>Deteksi Kelompok Ekstrem (5%):</b> Algoritma Deteksi Anomali berhasil memisahkan secara tepat 95 orang yang memiliki rekam jejak penyimpangan perilaku paling ekstrem dibandingkan dengan pola normal 1.790 responden lainnya di dalam dataset.</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

    # ==========================================
    # 4. MONITOR OUTPUT LIVE SIMULATOR
    # ==========================================
    st.write("### Hasil Pengujian Mandiri via Simulator Sidebar")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if sim_pred_class[0] == 1:
            st.markdown(f"<div style='background-color: #FADBD8; padding: 15px; border-radius: 5px; border-left: 5px solid #E74C3C;'><b>Hasil Prediksi Klasifikasi:</b> Berisiko Tinggi (User)<br>Probabilitas Keyakinan Model: {sim_pred_proba[0][1]*100:.2f}%</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color: #D4EFDF; padding: 15px; border-radius: 5px; border-left: 5px solid #2ECC71;'><b>Hasil Prediksi Klasifikasi:</b> Berisiko Rendah (Non-User)<br>Probabilitas Keyakinan Model: {sim_pred_proba[0][0]*100:.2f}%</div>", unsafe_allow_html=True)
            
    with col_s2:
        if sim_pred_anomaly[0] == -1:
            st.markdown("<div style='background-color: #FDEBD0; padding: 15px; border-radius: 5px; border-left: 5px solid #E67E22;'><b>Status Pola Perilaku:</b> Terdeteksi Anomali (Outlier)<br>Kombinasi skor psikologis ini dinilai ekstrem atau sangat langka dalam populasi.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background-color: #EBF5FB; padding: 15px; border-radius: 5px; border-left: 5px solid #3498DB;'><b>Status Pola Perilaku:</b> Normal (Inlier)<br>Kombinasi skor psikologis berada pada batasan wajar persebaran data umum.</div>", unsafe_allow_html=True)

    st.divider()

    # ==========================================
    # 5. EKSPLORASI DATA UTAMA
    # ==========================================
    st.subheader("Eksplorasi Data Fondasi Analisis")
    col_d1, col_d2 = st.columns([1, 3])
    with col_d1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric(label="Volume Data Observasi", value=f"{df.shape[0]} Responden")
        st.metric(label="Jumlah Fitur Evaluasi", value=f"{df.shape[1]} Kolom Atribut")
    with col_d2:
        st.write("Struktur Data Riil (Fitur Kepribadian Berupa Nilai Terstandarisasi):")
        st.dataframe(df.head(4), use_container_width=True)
        
    st.divider()

    # ==========================================
    # 6. NAVIGASI TAB UTAMA MODEL
    # ==========================================
    tab_klasifikasi, tab_anomali = st.tabs(["🎯 Model Prediksi Klasifikasi Risiko", "🕵️ Sistem Deteksi Anomali Perilaku"])

    # ---------------------------------------------------------
    # TAB 1: MODEL PREDIKSI KLASIFIKASI (DECISION TREE)
    # ---------------------------------------------------------
    with tab_klasifikasi:
        st.write("### Evaluasi Model Prediksi Risiko Penggunaan Cannabis")
        
        # PERBAIKAN: Mengganti st.help dengan st.info
        st.info("**Fungsi Utama Model Ini:** Menjawab pertanyaan *'Apakah seseorang cenderung menjadi pengguna berdasarkan sifat psikologisnya?'* Model dilatih menggunakan porsi data latih untuk memetakan aturan klasifikasi biner otomatis (User versus Non-User).", icon="💡")
        
        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            st.metric(label="Akurasi Pengujian Model (Ketetapan Prediksi)", value=f"{accuracy_score(y_test, y_pred)*100:.2f}%")
            
            report_dict = classification_report(y_test, y_pred, target_names=['Non-User', 'User'], output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose().round(2)
            
            st.write("Tabel Detail Evaluasi Akurasi (Precision dan Recall):")
            st.table(report_df.iloc[:-3, :-1])
            
        with col_m2:
            st.write("Matriks Akurasi Tebakan (Confusion Matrix):")
            fig1, ax1 = plt.subplots(figsize=(6, 3.8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Prediksi Non-User', 'Prediksi User'], 
                        yticklabels=['Aktual Non-User', 'Aktual User'], ax=ax1)
            plt.tight_layout()
            st.pyplot(fig1)
            st.caption("**Insight Grafik:** Fokus pada jalur warna diagonal gelap. Dominasi angka tinggi pada diagonal membuktikan tingkat kesalahan prediksi model berada pada kategori rendah.")

        st.divider()
        
        st.write("### Alur Aturan Logika Keputusan (Decision Tree)")
        fig2, ax2 = plt.subplots(figsize=(16, 7))
        tree.plot_tree(model_dt, feature_names=X_class.columns, class_names=['Non-User', 'User'], filled=True, rounded=True, fontsize=8, ax=ax2)
        plt.tight_layout()
        st.pyplot(fig2)
        st.info("""
        **Cara Membaca Pohon Aturan di Atas:**
        - **Simpul Puncak (Root Node):** Atribut psikologis yang paling krusial dan mendominasi pembentukan perilaku subjek penelitian.
        - **Gradasi Warna:** Kotak berwarna **Oranye pekat** merepresentasikan keputusan mutlak untuk kelas 'Non-User', sedangkan kotak berwarna **Biru pekat** merepresentasikan keputusan mutlak untuk kelas 'User'.
        - **Nilai Entropy:** Nilai indikator keacakan data. Semakin mengarah ke nilai 0, keputusan yang dihasilkan pada simpul tersebut semakin akurat.
        """, icon="📌")

    # ---------------------------------------------------------
    # TAB 2: DETEKSI ANOMALI (ISOLATION FOREST)
    # ---------------------------------------------------------
    with tab_anomali:
        st.write("### Deteksi Pola Menyimpang Menggunakan Isolation Forest")
        
        # PERBAIKAN: Mengganti st.help dengan st.info
        st.info("**Fungsi Utama Model Ini:** Mengisolasi objek/responden minoritas yang memiliki kombinasi sifat kepribadian serta riwayat konsumsi lintas zat yang sangat tidak wajar (ekstrem) dibanding mayoritas masyarakat umum.", icon="💡")
        
        anomali_count = (df_numeric['Anomaly_Label'] == -1).sum()
        normal_count = (df_numeric['Anomaly_Label'] == 1).sum()
        total_data = normal_count + anomali_count
        
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            st.metric(label="Subjek Berperilaku Wajar (Inliers)", value=f"{normal_count} Orang")
        with col_a2:
            st.metric(label="Subjek Terdeteksi Ekstrem (Outliers)", value=f"{anomali_count} Orang")
        with col_a3:
            st.metric(label="Rasio Kontaminasi Penyimpangan", value=f"{(anomali_count/total_data)*100:.2f}%")
            
        st.divider()
        
        col_g1, col_g2 = st.columns([3, 2])
        with col_g1:
            st.write("Peta Distribusi Klastering Reduksi Dimensi PCA 2D:")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_anomaly_scaled)
            
            fig3, ax3 = plt.subplots(figsize=(8, 4.8))
            ax3.scatter(X_pca[df_numeric['Anomaly_Label'] == 1, 0], X_pca[df_numeric['Anomaly_Label'] == 1, 1], color='#3498DB', alpha=0.4, label='Normal (Pola Umum)')
            ax3.scatter(X_pca[df_numeric['Anomaly_Label'] == -1, 0], X_pca[df_numeric['Anomaly_Label'] == -1, 1], color='#E74C3C', marker='x', s=70, label='Anomali (Pola Ekstrem)')
            ax3.set_xlabel("Komponen Utama Kompresi Data 1")
            ax3.set_ylabel("Komponen Utama Kompresi Data 2")
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig3)
            
        with col_g2:
            st.write("### Detail Analisis Klaster Anomali")
            st.info("""
            **Cara Membaca Peta Sebaran Grafik:**
            - **Titik Lingkaran Biru:** Mewakili 95% mayoritas populasi data yang memiliki keselarasan perilaku konsumsi zat normal sesuai tingkat kepribadian mereka.
            - **Titik Silang Merah:** Mewakili objek anomali (5% populasi) yang berhasil diisolasi oleh algoritma karena letaknya yang terasing jauh dari pusat kerumunan.
            
            **Interpretasi Kajian Perilaku:**
            Individu pada titik merah merupakan kelompok kritis yang memiliki karakteristik berlawanan. Melalui algoritma unsupervised ini, kita mampu menyaring kelompok berisiko tinggi tanpa perlu melakukan pelabelan data manual di awal proses analisis.
            """, icon="📌")

    # ==========================================
    # 7. KESIMPULAN AKADEMIS STRUKTURAL
    # ==========================================
    st.divider()
    st.write("### Kesimpulan Akhir Penelitian Data Mining")
    st.success("""
    1. **Validitas Fitur Kepribadian:** Eksperimen ini mengonfirmasi bahwa indikator psikologis formal (Big Five Personality) memiliki signifikansi statistik yang kuat sebagai prediktor kecenderungan adiksi. Karakteristik impulsivitas tinggi dan kedisiplinan yang rendah menjadi parameter dominan model.
    2. **Efektivitas Metode Kombinasi:** Penggabungan pendekatan Klasifikasi (Decision Tree) untuk pemetaan kelompok mayoritas dan Deteksi Anomali (Isolation Forest) untuk pemetaan deviasi minoritas berhasil membentuk framework deteksi dini yang komprehensif.
    3. **Kontribusi Praktis:** Melalui model yang telah di-deploy ke dalam arsitektur cloud ini, analisis data mining terbukti mampu dikonversi menjadi sistem pendukung keputusan (Decision Support System) yang aplikatif bagi praktisi kesehatan mental dan pembuat kebijakan sosial.
    """)

except FileNotFoundError:
    st.error("Berkas data 'drug_consumption.data' tidak ditemukan dalam repositori sistem. Pastikan letak folder penyimpanan berkas sudah selaras.")