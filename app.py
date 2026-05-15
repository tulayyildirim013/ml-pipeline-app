import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

st.title("🔬 ML Pipeline")

# ─────────────────────────────────────────
# ADIM 1: EXCEL YÜKLE
# ─────────────────────────────────────────
st.header("1. Excel Dosyası Yükle")
uploaded_file = st.file_uploader("Excel dosyanı seç", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success(f"✅ {df.shape[0]} satır, {df.shape[1]} sütun yüklendi")
    st.dataframe(df.head())

    # ─────────────────────────────────────────
    # ADIM 2: SÜTUN SEÇİMİ
    # ─────────────────────────────────────────
    st.header("2. Sütunları Seç")
    tum_sutunlar = df.columns.tolist()

    bagimsiz = st.multiselect("Bağımsız değişkenleri seç (X)", options=tum_sutunlar)
    bagimli  = st.multiselect("Bağımlı değişkeni seç (Y)", options=tum_sutunlar)

    if bagimsiz and bagimli:
        st.success(f"✅ {len(bagimsiz)} bağımsız, {len(bagimli)} bağımlı değişken seçildi")

        if st.button("Devam Et →"):
            st.session_state['df']       = df
            st.session_state['bagimsiz'] = bagimsiz
            st.session_state['bagimli']  = bagimli

# ─────────────────────────────────────────
# ADIM 3: PREPROCESSING
# ─────────────────────────────────────────
if 'bagimsiz' in st.session_state:
    st.header("3. Data Preprocessing")

    df       = st.session_state['df']
    bagimsiz = st.session_state['bagimsiz']
    bagimli  = st.session_state['bagimli']

    X = df[bagimsiz].copy()
    y = df[bagimli].copy()

    # ── 3.1 Duplicate ──────────────────────
    st.subheader("3.1 Duplicate Satırlar")
    dup_sayi = X.duplicated().sum()
    if dup_sayi > 0:
        st.warning(f"⚠️ {dup_sayi} duplicate satır bulundu")
        if st.checkbox("Duplicate satırları sil"):
            mask = ~X.duplicated()
            X = X[mask].reset_index(drop=True)
            y = y[mask].reset_index(drop=True)
            st.success(f"✅ Silindi. Kalan satır: {len(X)}")
    else:
        st.success("✅ Duplicate satır yok")

    # ── 3.2 Eksik Değer ────────────────────
    st.subheader("3.2 Eksik Değerler")
    eksik = X.isnull().sum()
    eksik_var = eksik[eksik > 0]
    if len(eksik_var) > 0:
        st.warning(f"⚠️ {len(eksik_var)} sütunda eksik değer var:")
        st.dataframe(eksik_var.rename("Eksik Sayısı"))
        eksik_yontem = st.selectbox(
            "Eksik değer yöntemi",
            ["Ortalama ile doldur", "Medyan ile doldur", "Satırı sil"]
        )
    else:
        st.success("✅ Eksik değer yok")
        eksik_yontem = "Ortalama ile doldur"

    # ── 3.3 Outlier ────────────────────────
    st.subheader("3.3 Outlier Tespiti")
    outlier_yontem = st.selectbox("Outlier yöntemi", ["IQR", "Z-Score", "Yok"])
    if outlier_yontem == "IQR":
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = ((X < Q1 - 1.5 * IQR) | (X > Q3 + 1.5 * IQR)).any(axis=1)
        st.info(f"📊 {outlier_mask.sum()} outlier satır tespit edildi")
        st.dataframe(X[outlier_mask])
    elif outlier_yontem == "Z-Score":
        z = np.abs((X - X.mean()) / X.std())
        outlier_mask = (z > 3).any(axis=1)
        st.info(f"📊 {outlier_mask.sum()} outlier satır tespit edildi")
    else:
        outlier_mask = pd.Series([False] * len(X))

    if outlier_mask.sum() > 0:
        if st.checkbox("Outlier satırları sil"):
            X = X[~outlier_mask].reset_index(drop=True)
            y = y[~outlier_mask].reset_index(drop=True)
            st.success(f"✅ Silindi. Kalan satır: {len(X)}")

    # ── 3.4 Leakage ────────────────────────
    st.subheader("3.4 Leakage Tespiti (Korelasyon)")
    esik = st.slider("Korelasyon eşiği", 0.80, 1.00, 0.95, 0.01)

    korelasyon = X.corr().abs()
    ust_ucgen = korelasyon.where(
        np.triu(np.ones(korelasyon.shape), k=1).astype(bool)
    )
    yuksek_kor = [
        col for col in ust_ucgen.columns
        if any(ust_ucgen[col] > esik)
    ]

    if yuksek_kor:
        st.warning(f"⚠️ {len(yuksek_kor)} sütun yüksek korelasyonlu:")
        st.write(yuksek_kor)
        if st.checkbox("Bu sütunları leakage olarak sil"):
            X = X.drop(columns=yuksek_kor)
            st.success(f"✅ Silindi. Kalan feature: {X.shape[1]}")
    else:
        st.success(f"✅ Eşik {esik} üzerinde leakage yok")

    # ── 3.5 Normalizasyon ──────────────────
    st.subheader("3.5 Normalizasyon")
    olcek = st.selectbox("Yöntem seç", ["StandardScaler", "MinMaxScaler", "Yok"])

    # ── Uygula Butonu ──────────────────────
    if st.button("✅ Preprocessing Uygula"):

        # Eksik değer
        if eksik_yontem == "Ortalama ile doldur":
            imp = SimpleImputer(strategy='mean')
            X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
        elif eksik_yontem == "Medyan ile doldur":
            imp = SimpleImputer(strategy='median')
            X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
        elif eksik_yontem == "Satırı sil":
            mask = X.notnull().all(axis=1)
            X = X[mask].reset_index(drop=True)
            y = y[mask].reset_index(drop=True)

        # Ölçeklendirme
        if olcek == "StandardScaler":
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        elif olcek == "MinMaxScaler":
            scaler = MinMaxScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        st.session_state['X'] = X
        st.session_state['y'] = y
        st.success(f"✅ Preprocessing tamamlandı! {X.shape[0]} satır, {X.shape[1]} feature")
        st.dataframe(X.head())


# ─────────────────────────────────────────
# ADIM 4: MODEL SEÇİMİ VE EĞİTİMİ
# ─────────────────────────────────────────
if 'X' in st.session_state:
    st.header("4. Model Seçimi ve Eğitimi")

    X = st.session_state['X']
    y = st.session_state['y']

    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    # Model seçimi
    model_secenekleri = {
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(random_state=42, verbosity=0),
        "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
    }

    secilen_modeller = st.multiselect(
        "Hangi modelleri denemek istiyorsun?",
        options=list(model_secenekleri.keys()),
        default=["Ridge", "Random Forest", "XGBoost"]
    )

    # Train/test split
    test_orani = st.slider("Test oranı", 0.10, 0.40, 0.20, 0.05)

    # Hedef seç (birden fazla Y varsa)
    if y.shape[1] > 1:
        hedef = st.selectbox("Hangi hedef değişkeni eğitelim?", options=y.columns.tolist())
        y_sec = y[hedef]
    else:
        y_sec = y.iloc[:, 0]
        hedef = y.columns[0]

    if st.button("🚀 Modelleri Eğit"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_sec, test_size=test_orani, random_state=42
        )

        sonuclar = []

        for isim in secilen_modeller:
            model = model_secenekleri[isim]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            r2  = r2_score(y_test, y_pred)
            cv  = cross_val_score(model, X, y_sec, cv=5, scoring='r2').mean()

            sonuclar.append({
                "Model": isim,
                "MAE": round(mae, 4),
                "R²": round(r2, 4),
                "CV R² (5-fold)": round(cv, 4)
            })

        sonuc_df = pd.DataFrame(sonuclar).sort_values("R²", ascending=False)
        st.session_state['sonuc_df'] = sonuc_df
        st.session_state['X_train'] = X_train
        st.session_state['X_test']  = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test']  = y_test
        st.session_state['hedef']   = hedef
        st.session_state['model_secenekleri'] = model_secenekleri
        st.session_state['secilen_modeller']  = secilen_modeller

        st.success("✅ Eğitim tamamlandı!")
        st.dataframe(sonuc_df)

        en_iyi = sonuc_df.iloc[0]['Model']
        st.info(f"🏆 En iyi model: **{en_iyi}** (R² = {sonuc_df.iloc[0]['R²']})")
