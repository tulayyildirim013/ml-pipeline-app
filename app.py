import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
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
        outlier_detay = (X < Q1 - 1.5 * IQR) | (X > Q3 + 1.5 * IQR)

    elif outlier_yontem == "Z-Score":
        z = np.abs((X - X.mean()) / X.std())
        outlier_mask = (z > 3).any(axis=1)
        outlier_detay = z > 3

    else:
        outlier_mask  = pd.Series([False] * len(X))
        outlier_detay = pd.DataFrame(False, index=X.index, columns=X.columns)

    if outlier_yontem != "Yok":
        st.info(f"📊 {outlier_mask.sum()} outlier satır tespit edildi")

    if outlier_mask.sum() > 0:
        # Kırmızı highlight
        def highlight_outlier(row):
            satir_idx = row.name
            renkler = []
            for col in X.columns:
                if outlier_detay.loc[satir_idx, col]:
                    renkler.append('background-color: #ff4b4b; color: white')
                else:
                    renkler.append('')
            return renkler

        outlier_satirlar = X[outlier_mask].copy()
        st.dataframe(
            outlier_satirlar.style.apply(highlight_outlier, axis=1)
        )

        if st.checkbox("Outlier satırları sil"):
            X = X[~outlier_mask].reset_index(drop=True)
            y = y[~outlier_mask].reset_index(drop=True)
            st.success(f"✅ Silindi. Kalan satır: {len(X)}")
           

 

    # ── 3.4 Leakage ────────────────────────
# ── 3.4 Leakage ────────────────────────
st.subheader("3.4 Leakage Tespiti (Korelasyon)")
    esik = st.slider("Korelasyon eşiği", 0.80, 1.00, 0.95, 0.01)
    corr_matrix = X.corr().abs()
    corr_matrix = corr_matrix * (1 - np.eye(len(corr_matrix)))

    yuksek_ciftler = []
    silinecek = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if val > esik:
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                yuksek_ciftler.append({
                    "Sütun 1": col_i,
                    "Sütun 2": col_j,
                    "Korelasyon": round(val, 4)
                })
                silinecek.add(col_j)

    if yuksek_ciftler:
        st.warning(f"⚠️ {len(silinecek)} sütun yüksek korelasyonlu:")
        st.dataframe(pd.DataFrame(yuksek_ciftler))
        if st.checkbox("Bu sütunları leakage olarak sil"):
            X = X.drop(columns=list(silinecek))
            st.success(f"✅ Silindi. Kalan feature: {X.shape[1]}")
    else:
        st.success(f"✅ Eşik {esik} üzerinde leakage yok")

    # Korelasyon sorgula
    st.subheader("İki Sütun Arasındaki Korelasyonu Sorgula")
    col1, col2 = st.columns(2)
    with col1:
        sutun1 = st.selectbox("1. Sütun", options=X.columns.tolist(), key="kor_s1")
    with col2:
        sutun2 = st.selectbox("2. Sütun", options=X.columns.tolist(), key="kor_s2")

    if sutun1 != sutun2:
        deger = X[sutun1].corr(X[sutun2])
        deger_abs = abs(deger)
        if deger_abs > 0.90:
            yorum = "🔴 Çok yüksek korelasyon — Leakage riski var!"
        elif deger_abs > 0.70:
            yorum = "🟠 Yüksek korelasyon — Dikkat et"
        elif deger_abs > 0.50:
            yorum = "🟡 Orta korelasyon — Normal"
        else:
            yorum = "🟢 Düşük korelasyon — Sorun yok"
        st.metric(label=f"{sutun1} ↔ {sutun2}", value=f"{deger:.4f}")
        st.info(yorum)

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

    model_secenekleri = {
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(random_state=42, verbosity=0),
        "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
    }

    secilen_model = st.selectbox(
        "Model seç",
        options=list(model_secenekleri.keys())
    )

    st.info("📊 Veri bölünmesi: %70 Train | %15 Validation | %15 Test")
    test_orani = 0.15
    val_orani  = 0.15

    if st.button("🚀 Modeli Eğit"):
        st.info(f"⏳ {secilen_model} ile 9 hedef için eğitim yapılıyor...")

        # Önce %15 test ayır
    X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42)
    
    # Kalanın %17.6'sı → toplam verinin %15'i validation olur
    X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42)
    
    st.write(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)} satır")

    sonuclar = []
    egitilmis_modeller = {}

    progress = st.progress(0)

    for i, hedef in enumerate(y.columns):
        model = model_secenekleri[secilen_model]
        
        from sklearn.base import clone
        m = clone(model)
        m.fit(X_train, y_train[hedef])
        y_pred = m.predict(X_test)

        mae = mean_absolute_error(y_test[hedef], y_pred)
        r2  = r2_score(y_test[hedef], y_pred)
        cv  = cross_val_score(m, X, y[hedef], cv=5, scoring='r2').mean()

            # Validation skoru
        y_val_pred = m.predict(X_val)
        mae_val = mean_absolute_error(y_val[hedef], y_val_pred)
        r2_val  = r2_score(y_val[hedef], y_val_pred)

        sonuclar.append({
        "Hedef": hedef,
        "Train R²": round(r2_score(y_train[hedef], m.predict(X_train)), 4),
        "Val R²":   round(r2_val, 4),
        "Test R²":  round(r2, 4),
        "Val MAE":  round(mae_val, 4),
        "Test MAE": round(mae, 4),
        "CV R²":    round(cv, 4)})

        egitilmis_modeller[hedef] = m
        progress.progress((i + 1) / len(y.columns))

        sonuc_df = pd.DataFrame(sonuclar)

        st.session_state['sonuc_df']          = sonuc_df
        st.session_state['egitilmis_modeller'] = egitilmis_modeller
        st.session_state['X_train']            = X_train
        st.session_state['X_test']             = X_test
        st.session_state['y_train']            = y_train
        st.session_state['y_test']             = y_test
        st.session_state['X_val']              = X_val
        st.session_state['y_val']              = y_val
        st.session_state['secilen_model']      = secilen_model

        st.success("✅ Eğitim tamamlandı!")
        st.dataframe(sonuc_df)

        en_iyi = sonuc_df.loc[sonuc_df['Test R²'].idxmax(), 'Hedef']
        en_iyi_r2 = sonuc_df['Test R²'].max()
        st.info(f"🏆 En yüksek R²: **{en_iyi}** (R² = {en_iyi_r2})")

# ─────────────────────────────────────────
# ADIM 5: SHAP ANALİZİ
# ─────────────────────────────────────────
if 'egitilmis_modeller' in st.session_state:
    st.header("5. SHAP Analizi ve Feature Importance")

    import shap
    import matplotlib.pyplot as plt

    egitilmis_modeller = st.session_state['egitilmis_modeller']
    X_test             = st.session_state['X_test']
    X_train            = st.session_state['X_train']

    hedef_sec = st.selectbox(
        "Hangi hedef için SHAP analizi?",
        options=list(egitilmis_modeller.keys())
    )

    if st.button("📊 SHAP Analizi Yap"):
        model = egitilmis_modeller[hedef_sec]

        st.info("⏳ SHAP hesaplanıyor...")

        # SHAP explainer
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # ── 5.1 SHAP Summary Plot ──
        st.subheader(f"5.1 SHAP Summary — {hedef_sec}")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(fig1)
        plt.close()

        # ── 5.2 Feature Importance ──
        st.subheader(f"5.2 Feature Importance — {hedef_sec}")
        importance = pd.DataFrame({
            'Feature':    X_test.columns,
            'Importance': abs(shap_values).mean(axis=0)
        }).sort_values('Importance', ascending=False)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.barh(importance['Feature'][:15], importance['Importance'][:15])
        ax2.set_xlabel("Ortalama |SHAP değeri|")
        ax2.set_title(f"Top 15 Feature — {hedef_sec}")
        ax2.invert_yaxis()
        st.pyplot(fig2)
        plt.close()

        st.session_state['shap_values'] = shap_values
        st.session_state['importance']  = importance
        st.session_state['hedef_sec']   = hedef_sec

        # ── 5.3 Eşik Değerleri ──
        st.subheader("5.3 Feature Eşik Değerleri")
        st.info("En önemli 10 feature için optimum aralıklar:")

        esik_df = pd.DataFrame()
        top10   = importance['Feature'].head(10).tolist()

        for feat in top10:
            q25 = X_train[feat].quantile(0.25)
            q75 = X_train[feat].quantile(0.75)
            esik_df = pd.concat([esik_df, pd.DataFrame([{
                'Feature': feat,
                'Min':     round(X_train[feat].min(), 4),
                'Q25':     round(q25, 4),
                'Medyan':  round(X_train[feat].median(), 4),
                'Q75':     round(q75, 4),
                'Max':     round(X_train[feat].max(), 4),
            }])], ignore_index=True)

        st.dataframe(esik_df)

        # ── 5.4 PDF Rapor İndir ──
        st.subheader("5.4 Raporu İndir")
        
        # CSV olarak indir
        csv = importance.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Feature Importance CSV İndir",
            data=csv,
            file_name=f"feature_importance_{hedef_sec}.csv",
            mime="text/csv"
        )

        esik_csv = esik_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Eşik Değerleri CSV İndir",
            data=esik_csv,
            file_name=f"esik_degerler_{hedef_sec}.csv",
            mime="text/csv"
        )

        st.success("✅ SHAP analizi tamamlandı!")


# ─────────────────────────────────────────
# ADIM 6: MODEL KARŞILAŞTIRMA
# ─────────────────────────────────────────
if 'egitilmis_modeller' in st.session_state:
    st.header("6. Model Karşılaştırma")

    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.base import clone
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    X_train = st.session_state['X_train']
    X_val   = st.session_state['X_val']
    X_test  = st.session_state['X_test']
    y_train = st.session_state['y_train']
    y_val   = st.session_state['y_val']
    y_test  = st.session_state['y_test']

    tum_modeller = {
        "Ridge":            Ridge(),
        "Random Forest":    RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42),
        "Gradient Boosting":GradientBoostingRegressor(random_state=42),
        "Extra Trees":      ExtraTreesRegressor(n_estimators=100, random_state=42),
        "XGBoost":          XGBRegressor(random_state=42, verbosity=0),
        "LightGBM":         LGBMRegressor(random_state=42, verbose=-1),
    }

    karsilastir_hedef = st.selectbox(
        "Karşılaştırma için hedef seç",
        options=y_test.columns.tolist(),
        key="karsilastir_hedef"
    )

    if st.button("🔍 Tüm Modelleri Karşılaştır"):
        karsilastir_sonuc = []
        progress = st.progress(0)

        for i, (isim, model) in enumerate(tum_modeller.items()):
            m = clone(model)
            m.fit(X_train, y_train[karsilastir_hedef])

            y_val_pred  = m.predict(X_val)
            y_test_pred = m.predict(X_test)

            karsilastir_sonuc.append({
                "Model":    isim,
                "Val R²":   round(r2_score(y_val[karsilastir_hedef], y_val_pred), 4),
                "Test R²":  round(r2_score(y_test[karsilastir_hedef], y_test_pred), 4),
                "Val MAE":  round(mean_absolute_error(y_val[karsilastir_hedef], y_val_pred), 4),
                "Test MAE": round(mean_absolute_error(y_test[karsilastir_hedef], y_test_pred), 4),
            })
            progress.progress((i + 1) / len(tum_modeller))

        kar_df = pd.DataFrame(karsilastir_sonuc).sort_values("Test R²", ascending=False)
        st.session_state['kar_df'] = kar_df
        st.success("✅ Karşılaştırma tamamlandı!")
        st.dataframe(kar_df)

        # Grafik
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(kar_df['Model'], kar_df['Test R²'], color='steelblue')
        ax.set_ylabel("Test R²")
        ax.set_title(f"Model Karşılaştırma — {karsilastir_hedef}")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=15)
        st.pyplot(fig)
        plt.close()

        en_iyi = kar_df.iloc[0]['Model']
        st.info(f"🏆 En iyi model: **{en_iyi}** (Test R² = {kar_df.iloc[0]['Test R²']})")

# ─────────────────────────────────────────
# ADIM 7: TAHMİN SAYFASI
# ─────────────────────────────────────────
if 'egitilmis_modeller' in st.session_state:
    st.header("7. Yeni Veri Tahmini")

    egitilmis_modeller = st.session_state['egitilmis_modeller']
    X                  = st.session_state['X']

    st.info("Aşağıya yeni verinin değerlerini gir, tüm hedefler için tahmin yapılacak.")

    # Her feature için input
    yeni_veri = {}
    cols = st.columns(3)

    for i, feat in enumerate(X.columns):
        with cols[i % 3]:
            val = st.number_input(
                feat,
                value=float(X[feat].mean()),
                key=f"input_{feat}"
            )
            yeni_veri[feat] = val

    if st.button("🎯 Tahmin Et"):
        yeni_df = pd.DataFrame([yeni_veri])
        tahmin_sonuc = []

        for hedef, model in egitilmis_modeller.items():
            tahmin = model.predict(yeni_df)[0]
            tahmin_sonuc.append({
                "Hedef":   hedef,
                "Tahmin":  round(float(tahmin), 4)
            })

        tahmin_df = pd.DataFrame(tahmin_sonuc)
        st.session_state['tahmin_df'] = tahmin_df
        st.success("✅ Tahmin tamamlandı!")
        st.dataframe(tahmin_df)

# ─────────────────────────────────────────
# ADIM 8: RAPOR İNDİR
# ─────────────────────────────────────────
if 'sonuc_df' in st.session_state:
    st.header("8. Rapor İndir")

    rapor_parcalar = []

    if 'sonuc_df' in st.session_state:
        rapor_parcalar.append("## Model Eğitim Sonuçları\n")
        rapor_parcalar.append(st.session_state['sonuc_df'].to_csv(index=False))

    if 'kar_df' in st.session_state:
        rapor_parcalar.append("\n## Model Karşılaştırma\n")
        rapor_parcalar.append(st.session_state['kar_df'].to_csv(index=False))

    if 'importance' in st.session_state:
        rapor_parcalar.append("\n## Feature Importance\n")
        rapor_parcalar.append(st.session_state['importance'].to_csv(index=False))

    if 'tahmin_df' in st.session_state:
        rapor_parcalar.append("\n## Tahmin Sonuçları\n")
        rapor_parcalar.append(st.session_state['tahmin_df'].to_csv(index=False))

    rapor_txt = "\n".join(rapor_parcalar)

    st.download_button(
        label="📥 Tam Raporu İndir (CSV)",
        data=rapor_txt.encode('utf-8'),
        file_name="ml_pipeline_rapor.csv",
        mime="text/csv"
    )

    st.success("✅ Rapor hazır!")


# ─────────────────────────────────────────
# ADIM 9: INVERSE DESIGN
# ─────────────────────────────────────────
if 'egitilmis_modeller' in st.session_state:
    st.header("9. Inverse Design (Ters Tasarım)")
    st.info("Hedef performans değerlerini gir, en uygun alaşım bileşimi bulunacak.")

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    egitilmis_modeller = st.session_state['egitilmis_modeller']
    X                  = st.session_state['X']

    # ── Hedef değerler ──
    st.subheader("9.1 Minimum Hedef Değerleri Gir")

    hedef_limitler = {}
    cols = st.columns(3)

    for i, hedef in enumerate(egitilmis_modeller.keys()):
        with cols[i % 3]:
            aktif = st.checkbox(f"{hedef}", key=f"inv_{hedef}_aktif")
            if aktif:
                min_val = st.number_input(
                    f"Min {hedef}",
                    value=0.0,
                    key=f"inv_{hedef}_min"
                )
                hedef_limitler[hedef] = min_val

    # ── Feature sınırları ──
    st.subheader("9.2 Feature Sınırları")
    st.info("Arama aralığı: eğitim verisinin min-max değerleri kullanılacak.")

    feature_sinirlar = {}
    for feat in X.columns:
        feature_sinirlar[feat] = (
            float(X[feat].min()),
            float(X[feat].max())
        )

    n_trials = st.slider("Optimizasyon denemesi", 100, 1000, 300, 100)

    if hedef_limitler and st.button("🔍 Optimize Et"):
        st.info(f"⏳ {n_trials} deneme ile optimizasyon yapılıyor...")

        def objective(trial):
            # Her feature için değer öner
            x_dict = {}
            for feat, (lo, hi) in feature_sinirlar.items():
                if lo == hi:
                    x_dict[feat] = lo
                else:
                    x_dict[feat] = trial.suggest_float(feat, lo, hi)

            x_df = pd.DataFrame([x_dict])

            # Her hedef için ceza hesapla
            toplam_ceza = 0
            for hedef, min_val in hedef_limitler.items():
                tahmin = egitilmis_modeller[hedef].predict(x_df)[0]
                if tahmin < min_val:
                    toplam_ceza += (min_val - tahmin) ** 2

            return toplam_ceza

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        # En iyi 5 sonuç
        st.success("✅ Optimizasyon tamamlandı!")

        en_iyi_params = study.best_params
        en_iyi_df     = pd.DataFrame([en_iyi_params])

        # Tahminleri göster
        st.subheader("9.3 En İyi Alaşım Bileşimi")
        st.dataframe(en_iyi_df)

        st.subheader("9.4 Bu Bileşimin Tahmini Performansı")
        performans = []
        for hedef, model in egitilmis_modeller.items():
            tahmin = model.predict(en_iyi_df)[0]
            limit  = hedef_limitler.get(hedef, None)
            durum  = ""
            if limit is not None:
                durum = "✅" if tahmin >= limit else "❌"
            performans.append({
                "Hedef":        hedef,
                "Tahmin":       round(float(tahmin), 4),
                "Min Hedef":    limit if limit else "-",
                "Durum":        durum
            })

        perf_df = pd.DataFrame(performans)
        st.dataframe(perf_df)

        # Top 5 deneme
        st.subheader("9.5 En İyi 5 Kombinasyon")
        trials_df = study.trials_dataframe()
        trials_df = trials_df.sort_values('value').head(5)
        st.dataframe(trials_df)

        # İndir
        csv = en_iyi_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 En İyi Bileşimi İndir",
            data=csv,
            file_name="inverse_design_sonuc.csv",
            mime="text/csv"
        )

        perf_csv = perf_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Performans Tahminini İndir",
            data=perf_csv,
            file_name="inverse_design_performans.csv",
            mime="text/csv"
        )
