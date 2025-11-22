import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load Model & Encoders
# Pastikan file .pkl ada di folder yang sama dengan app.py
try:
    model = joblib.load('co2_xgb_model.pkl')
    encoders = joblib.load('label_encoders.pkl')
except FileNotFoundError:
    st.error("File model (.pkl) tidak ditemukan! Pastikan file model dan encoder ada di satu folder.")
    st.stop()

# Judul & Konfigurasi Halaman
st.set_page_config(page_title="Carbon Emission AI", page_icon="🚗", layout="centered")

st.title("🚗 Prediksi Jejak Karbon Kendaraan")
st.markdown("""
Aplikasi ini menggunakan **Artificial Intelligence (XGBoost)** untuk memprediksi emisi CO2 kendaraan.
Target: **SDG #13 Climate Action** 🌍
""")

st.divider()

# Form Input User
with st.form("prediction_form"):
    st.subheader("Spesifikasi Kendaraan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Merek Mobil (Ambil otomatis dari Encoder)
        make_options = encoders['Make'].classes_
        make = st.selectbox("Merek Mobil", make_options)
        
        # 2. Jenis Kendaraan
        vehicle_options = encoders['Vehicle Class'].classes_
        vehicle_class = st.selectbox("Jenis Bodi (Vehicle Class)", vehicle_options)
        
        # 3. Tipe Transmisi
        trans_options = encoders['Transmission'].classes_
        transmission = st.selectbox("Tipe Transmisi", trans_options)
        
        # 4. Jenis Bahan Bakar (MAPPING USER FRIENDLY)
        # Kita buat kamus: Kunci (Teks Tampil) -> Nilai (Kode Asli Dataset)
        fuel_map = {
            "Regular Gasoline (X) - Setara Pertalite/Pertamax": "X",
            "Premium Gasoline (Z) - Setara Pertamax Turbo": "Z",
            "Diesel (D) - Setara Solar/Dex": "D",
            "Ethanol (E) - E85": "E",
            "Natural Gas (N)": "N"
        }
        # User pilih teks panjangnya
        selected_fuel_label = st.selectbox("Jenis Bahan Bakar", list(fuel_map.keys()))
        # Kita ambil kode aslinya (X, Z, D, dst) buat dikirim ke AI
        fuel_type_code = fuel_map[selected_fuel_label]

    with col2:
        # 5. Spesifikasi Mesin
        engine_size = st.number_input("Ukuran Mesin (Liter)", min_value=0.0, max_value=10.0, value=2.0, step=0.1, help="Contoh: 1.5 untuk 1500cc")
        cylinders = st.slider("Jumlah Silinder", min_value=3, max_value=16, value=4)
        fuel_comb = st.number_input("Konsumsi BBM (L/100km)", min_value=1.0, max_value=50.0, value=8.5, step=0.1, help="Makin besar angka, makin boros.")

    # Tombol Submit
    submitted = st.form_submit_button("🚀 Hitung Emisi CO2")

if submitted:
    try:
        # 2. Preprocessing Input (Ubah Pilihan User jadi Angka)
        # Hati-hati: Urutan fitur harus SAMA PERSIS waktu training
        
        make_enc = encoders['Make'].transform([make])[0]
        vehicle_enc = encoders['Vehicle Class'].transform([vehicle_class])[0]
        trans_enc = encoders['Transmission'].transform([transmission])[0]
        fuel_enc = encoders['Fuel Type'].transform([fuel_type_code])[0] # Pakai kode yang sudah dimapping
        
        # Format List: ['Make', 'Vehicle Class', 'Transmission', 'Fuel Type', 'Engine Size', 'Cylinders', 'Fuel Comb']
        input_data = [[make_enc, vehicle_enc, trans_enc, fuel_enc, engine_size, cylinders, fuel_comb]]
        
        # 3. Prediksi
        prediction = model.predict(input_data)[0]
        
        # Tampilkan Hasil
        st.divider()
        st.metric(label="Estimasi Emisi CO2", value=f"{prediction:.2f} g/km")
        
        # Logika Kategori Emisi
        if prediction < 200:
            st.success("✅ **Low Emission**: Kendaraan ini cukup ramah lingkungan.")
        elif prediction < 300:
            st.warning("⚠️ **Medium Emission**: Emisi kendaraan ini tergolong sedang.")
        else:
            st.error("⛔ **High Emission**: Kendaraan ini sangat polutif!")
            st.markdown("*Saran: Pertimbangkan kendaraan hybrid atau listrik untuk mengurangi jejak karbon.*")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")