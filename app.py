import streamlit as st
import pandas as pd

st.title("🔬 ML Pipeline")

# Adım 1: Excel Yükle
st.header("1. Excel Dosyası Yükle")
uploaded_file = st.file_uploader("Excel dosyanı seç", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success(f"✅ Dosya yüklendi! {df.shape[0]} satır, {df.shape[1]} sütun")
    st.dataframe(df.head())

    # Adım 2: Sütun Seçimi
    st.header("2. Sütunları Seç")
    
    tum_sutunlar = df.columns.tolist()
    
    bagimsiz = st.multiselect(
        "Bağımsız değişkenleri seç (X)",
        options=tum_sutunlar
    )
    
    bagimli = st.multiselect(
        "Bağımlı değişkeni seç (Y)",
        options=tum_sutunlar
    )

    if bagimsiz and bagimli:
        st.success(f"✅ {len(bagimsiz)} bağımsız, {len(bagimli)} bağımlı değişken seçildi")
        
        if st.button("Devam Et →"):
            st.session_state['df'] = df
            st.session_state['bagimsiz'] = bagimsiz
            st.session_state['bagimli'] = bagimli
            st.info("Preprocessing adımına geçiliyor...")
