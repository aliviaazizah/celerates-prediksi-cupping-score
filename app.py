# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from modules.prediction.preprocess_single import preprocess_single
from modules.prediction.model_loader import load_model_and_preprocessor, load_from_uploaded
from modules.prediction.recommend import quality_category, recommendation_from_category
import plotly.express as px

st.set_page_config(page_title="Coffee Score Prediction", layout="centered")
st.title("☕ Coffee Score Prediction — Single Input")

st.sidebar.header("Pengaturan")
mode = st.sidebar.radio("Pilih mode prediksi:", ["fisik", "akurat"])
st.sidebar.markdown("Jika kamu sudah punya model/preprocessor (.pkl), upload di sini (opsional).")
uploaded_model = st.sidebar.file_uploader("Upload model (.pkl)", type=["pkl","joblib"])
uploaded_pre = st.sidebar.file_uploader("Upload preprocessor (.pkl)", type=["pkl","joblib"])

st.write("Isi data kopi (1 sampel) di bawah, lalu klik **Prediksi Skor**.")

# Input form
with st.form(key="input_form"):
    if mode == "fisik":
        altitude = st.number_input("Altitude (m)", min_value=0.0, max_value=10000.0, value=1500.0, step=1.0)
        coffee_age = st.number_input("Coffee Age (days)", min_value=0.0, max_value=10000.0, value=300.0, step=1.0)
        moisture = st.number_input("Moisture %", min_value=0.0, max_value=100.0, value=12.5, step=0.1)
        c1 = st.number_input("Category One Defects", min_value=0, max_value=100, value=0, step=1)
        c2 = st.number_input("Category Two Defects", min_value=0, max_value=500, value=0, step=1)
        quakers = st.number_input("Quakers", min_value=0, max_value=100, value=0, step=1)
    else:
        uniformity = st.slider("Uniformity", 0.0, 10.0, 10.0)
        clean_cup = st.slider("Clean Cup", 0.0, 10.0, 10.0)
        sweetness = st.slider("Sweetness", 0.0, 10.0, 7.0)
        overall = st.slider("Overall", 0.0, 10.0, 8.0)
        flavor = st.slider("Flavor", 0.0, 10.0, 8.0)
        aftertaste = st.slider("Aftertaste", 0.0, 10.0, 8.0)
        balance = st.slider("Balance", 0.0, 10.0, 8.0)
        acidity = st.slider("Acidity", 0.0, 10.0, 8.0)
        aroma = st.slider("Aroma", 0.0, 10.0, 8.0)
        body = st.slider("Body", 0.0, 10.0, 8.0)

    processing = st.selectbox("Processing Method", ["Washed / Wet", "Natural / Dry", "Pulped natural / honey", "unknown"])
    variety = st.selectbox("Variety", ["Catuai", "Caturra", "Bourbon", "Typica", "Other", "unknown"])

    submit = st.form_submit_button("Prediksi Skor")

# Session state history
if "history" not in st.session_state:
    st.session_state.history = []

if submit:
    # build sample dict
    if mode == "fisik":
        sample = {
            "Altitude": altitude,
            "Coffee Age": coffee_age,
            "Moisture %": moisture,
            "Category One Defects": c1,
            "Category Two Defects": c2,
            "Quakers": quakers,
            "Processing Method": processing,
            "Variety": variety
        }
    else:
        sample = {
            "Uniformity": uniformity,
            "Clean Cup": clean_cup,
            "Sweetness": sweetness,
            "Overall": overall,
            "Flavor": flavor,
            "Aftertaste": aftertaste,
            "Balance": balance,
            "Acidity": acidity,
            "Aroma": aroma,
            "Body": body,
            "Processing Method": processing,
            "Variety": variety
        }

    # Preprocess single
    df_sample = preprocess_single(sample)

    # Load model & preprocessor: uploaded first, otherwise from models/
    model_obj = None
    preprocessor_obj = None
    try:
        if uploaded_model:
            loaded = load_from_uploaded(uploaded_model)
            if isinstance(loaded, dict):
                model_obj = loaded.get('model') or list(loaded.values())[0]
                preprocessor_obj = loaded.get('preprocessor') or preprocessor_obj
            else:
                model_obj = loaded
        if uploaded_pre:
            preprocessor_obj = load_from_uploaded(uploaded_pre)
    except Exception as e:
        st.error(f"Gagal load uploaded files: {e}")

    if model_obj is None or preprocessor_obj is None:
        m, p = load_model_and_preprocessor(mode)
        if model_obj is None:
            model_obj = m
        if preprocessor_obj is None:
            preprocessor_obj = p

    if model_obj is None:
        st.error("Model tidak ditemukan (letakkan di folder `models/` atau upload lewat sidebar).")
    else:
        if preprocessor_obj is None:
            st.warning("Preprocessor tidak ditemukan — membuat preprocessor fallback (hasil mungkin berbeda).")
            # build fallback
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            if mode == "fisik":
                features = ['Altitude', 'Coffee Age', 'Moisture %', 'Category One Defects', 'Category Two Defects', 'Quakers']
            else:
                features = ['Uniformity','Clean Cup','Sweetness','Overall','Flavor','Aftertaste','Balance','Acidity','Aroma','Body']
            cat_cols = [c for c in ['Processing Method','Variety'] if c in df_sample.columns]
            num_cols = [f for f in features if f in df_sample.columns]
            preprocessor_obj = ColumnTransformer([
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
                ('num', StandardScaler(), num_cols)
            ])
            try:
                preprocessor_obj.fit(df_sample[cat_cols + num_cols])
            except Exception as e:
                st.error(f"Gagal fit fallback preprocessor: {e}")
                st.stop()

        # Prepare transformation input: preprocessor expects categorical cols first as used in training
        if mode == "fisik":
            features = ['Altitude', 'Coffee Age', 'Moisture %', 'Category One Defects', 'Category Two Defects', 'Quakers']
        else:
            features = ['Uniformity','Clean Cup','Sweetness','Overall','Flavor','Aftertaste','Balance','Acidity','Aroma','Body']
        num_features = [f for f in features if f in df_sample.columns]
        cat_features = [c for c in ['Processing Method','Variety'] if c in df_sample.columns]
        X_input = df_sample[cat_features + num_features]

        try:
            X_trans = preprocessor_obj.transform(X_input)
        except Exception as e:
            st.error(f"Gagal transform input dengan preprocessor: {e}")
            st.stop()

        try:
            pred = model_obj.predict(X_trans)[0]
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")
            st.stop()

        cat = quality_category(pred)
        reco = recommendation_from_category(cat)

        st.metric("Predicted Score", f"{pred:.2f}")
        st.markdown(f"**Kategori:** {cat}")
        st.markdown(f"**Rekomendasi:** {reco}")

        # add to history
        rec = df_sample.copy()
        rec['Predicted_Score'] = pred
        rec['Quality_Category'] = cat
        rec['Recommendation'] = reco
        st.session_state.history.append(rec)

# show history if any
if len(st.session_state.history) > 0:
    st.subheader("Riwayat Prediksi (session)")
    hist_df = pd.concat(st.session_state.history, ignore_index=True)
    st.dataframe(hist_df)

    csv = hist_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Riwayat Prediksi (CSV)", csv, "riwayat_prediksi.csv", "text/csv")

# feature importance section (only shown if model loaded earlier)
st.markdown("---")
st.subheader("Feature importance (jika tersedia dari model)")

# try to show importance for the last used model if present in session
try:
    # try to find model in local variable if previous predict ran
    # For simplicity, we attempt to load model for selected mode
    model_obj2, preproc2 = load_model_and_preprocessor(mode)
    if model_obj2 is None:
        st.info("Tidak ada model tersedia untuk menampilkan feature importance.")
    else:
        # get feature names
        try:
            feat_names = list(preproc2.get_feature_names_out())
        except:
            # fallback names
            if mode == "fisik":
                feat_names = ['Processing Method', 'Variety'] + ['Altitude','Coffee Age','Moisture %','Category One Defects','Category Two Defects','Quakers']
            else:
                feat_names = ['Processing Method', 'Variety'] + ['Uniformity','Clean Cup','Sweetness','Overall','Flavor','Aftertaste','Balance','Acidity','Aroma','Body']

        importance = None
        if hasattr(model_obj2, "feature_importances_"):
            importance = model_obj2.feature_importances_
        elif hasattr(model_obj2, "coef_"):
            coef = np.array(model_obj2.coef_)
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)
            importance = np.abs(coef)

        if importance is not None:
            # align lengths
            if len(feat_names) != len(importance):
                try:
                    feat_names = list(preproc2.get_feature_names_out())
                except:
                    feat_names = [f"f_{i}" for i in range(len(importance))]
            feat_df = pd.DataFrame({"feature": feat_names, "importance": importance})
            feat_df['feature'] = feat_df['feature'].astype(str).str.replace('cat__','').str.replace('num__','')
            feat_df = feat_df.sort_values("importance", ascending=True).tail(50)
            fig = px.bar(feat_df, x='importance', y='feature', orientation='h', title="Feature importance")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Model tidak menyediakan feature_importances_ atau coef_.")
except Exception as e:
    st.warning(f"Gagal menampilkan feature importance: {e}")
