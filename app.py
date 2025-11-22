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
    st.error("File model (.pkl) tidak ditemukan! Pastikan sudah didownload dari Colab.")
    st.stop()

# Judul & Deskripsi
st.set_page_config(page_title="Carbon Emission AI", page_icon="🚗")
st.title("🚗 Prediksi Jejak Karbon Kendaraan")
st.markdown("Aplikasi MVP untuk memprediksi emisi CO2 berdasarkan spesifikasi mesin.")

# Form Input User
with st.form("prediction_form"):
    st.subheader("Spesifikasi Kendaraan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Dropdown otomatis ambil list merek dari encoder
        make_options = encoders['Make'].classes_
        make = st.selectbox("Merek Mobil (Make)", make_options)
        
        vehicle_options = encoders['Vehicle Class'].classes_
        vehicle_class = st.selectbox("Jenis Kendaraan", vehicle_options)
        
        trans_options = encoders['Transmission'].classes_
        transmission = st.selectbox("Tipe Transmisi", trans_options)
        
        fuel_options = encoders['Fuel Type'].classes_
        fuel_type = st.selectbox("Jenis Bahan Bakar", fuel_options)

    with col2:
        engine_size = st.number_input("Ukuran Mesin (Liter)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        cylinders = st.slider("Jumlah Silinder", min_value=3, max_value=16, value=4)
        fuel_comb = st.number_input("Konsumsi BBM (L/100km)", min_value=1.0, max_value=50.0, value=8.5, step=0.1)

    # Tombol Submit
    submitted = st.form_submit_button("Hitung Emisi CO2")

if submitted:
    # 2. Preprocessing Input (Ubah Pilihan User jadi Angka)
    make_enc = encoders['Make'].transform([make])[0]
    vehicle_enc = encoders['Vehicle Class'].transform([vehicle_class])[0]
    trans_enc = encoders['Transmission'].transform([transmission])[0]
    fuel_enc = encoders['Fuel Type'].transform([fuel_type])[0]
    
    # Urutan fitur HARUS SAMA PERSIS waktu training di Colab
    # ['Make', 'Vehicle Class', 'Transmission', 'Fuel Type', 'Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']
    input_data = [[make_enc, vehicle_enc, trans_enc, fuel_enc, engine_size, cylinders, fuel_comb]]
    
    # 3. Prediksi
    prediction = model.predict(input_data)[0]
    
    # Tampilkan Hasil
    st.divider()
    st.header(f"Hasil Prediksi: {prediction:.2f} g/km")
    
    # Logika sederhana untuk kategori (bisa disesuaikan)
    if prediction < 200:
        st.success("✅ Kendaraan ini cukup ramah lingkungan (Low Emission).")
    elif prediction < 300:
        st.warning("⚠️ Emisi sedang (Medium Emission).")
    else:
        st.error("⛔ Emisi sangat tinggi! Tidak direkomendasikan.")