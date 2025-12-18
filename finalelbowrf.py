import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from scipy.stats import chi2_contingency

# Modelleme
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Veri İşleme
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report, precision_score, recall_score, f1_score)

# Ayarlar
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Miuul Alışveriş Analizi (Final)", page_icon="🛍️", layout="wide")

# CSS
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1 { color: #1e3a8a; }
    div[data-testid="stMetric"] { background-color: #ffffff; border: 1px solid #e5e7eb; border-radius: 5px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. YARDIMCI FONKSİYONLAR (SENİN KODUNUN AYNISI)
# =============================================================================

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
    return temp_df

def add_conditional_freq_feature(df, group_col, cat_col, prefix=None, smoothing=1.0):
    if prefix is None: prefix = f"P_{cat_col}_given_{group_col}"
    ct = pd.crosstab(df[group_col], df[cat_col])
    ct_smoothed = ct + smoothing
    probs = ct_smoothed.div(ct_smoothed.sum(axis=1), axis=0)
    feat = df[[group_col, cat_col]].apply(lambda r: probs.loc[r[group_col], r[cat_col]], axis=1)
    df[f"{prefix}"] = feat
    return df

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

# =============================================================================
# 2. DATA PROCESSING PIPELINE (SENİN MANTIĞIN)
# =============================================================================
def process_data_pipeline(df):
    """
    Orijinal koddaki 'process_data' fonksiyonunun birebir aynısıdır.
    """
    df_eng = df.copy()

    # Target
    if 'SUBSCRIPTION_STATUS' in df_eng.columns:
        df_eng['TEMP_TARGET'] = df_eng['SUBSCRIPTION_STATUS'].map({"Yes": 1, "No": 0})
    else:
        df_eng['TEMP_TARGET'] = 0 

    # --- 1. Temel Değişkenler ---
    df_eng['TOTAL_SPEND_WEIGHTED_NEW'] = df_eng['PREVIOUS_PURCHASES'] * df_eng['PURCHASE_AMOUNT_(USD)']
    df_eng['SPEND_PER_PURCHASE_NEW'] = df_eng['PURCHASE_AMOUNT_(USD)'] / (df_eng['PREVIOUS_PURCHASES'] + 1)
    
    freq_map = {'Weekly': 52, 'Bi-Weekly': 26, 'Fortnightly': 26, 'Quarterly': 4, 'Annually': 1, 'Monthly': 12, 'Every 3 Months': 4}
    df_eng['FREQUENCY_VALUE_NEW'] = df_eng['FREQUENCY_OF_PURCHASES'].map(freq_map)

    pay_map = {'Cash': 'Cash', 'Credit Card': 'Card', 'Debit Card': 'Card', 'PayPal': 'Online', 'Venmo': 'Online', 'Bank Transfer': 'Online'}
    df_eng['PAYMENT_TYPE_NEW'] = df_eng['PAYMENT_METHOD'].map(pay_map)

    # --- 2. Featurelar ---
    df_eng["AGE_NEW"] = pd.cut(df_eng["AGE"], bins=[0, 30, 45, 56, 200], labels=["18-30", "31-45", "46-56", "57-70"])
    df_eng["PURCHASE_AMOUNT_(USD)_NEW"] = pd.qcut(df_eng["PURCHASE_AMOUNT_(USD)"], q=4, labels=["Low", "Mid", "High", "Very High"])
    df_eng["LOYALTY_LEVEL_NEW"] = pd.cut(df_eng["PREVIOUS_PURCHASES"], bins=[0, 13, 25, 38, 200], labels=["Low", "Mid", "High", "Very High"], include_lowest=True)

    # --- Leakage Featurelar ---
    df_eng["SUB_FREQ_NEW"] = (df_eng["TEMP_TARGET"].astype(str) + "_" + df_eng["FREQUENCY_OF_PURCHASES"].astype(str))
    df_eng["PROMO_NO_SUB_NEW"] = ((df_eng["PROMO_CODE_USED"] == "Yes") & (df_eng["TEMP_TARGET"] == 0)).astype(int)
    df_eng["SHIP_SUB_NEW"] = (df_eng["SHIPPING_TYPE"].astype(str) + "_" + df_eng["TEMP_TARGET"].astype(str))

    # --- Diğer Featurelar ---
    df_eng["SEASON_CATEGORY_NEW"] = df_eng["SEASON"].astype(str) + "_" + df_eng["CATEGORY"].astype(str)
    df_eng["SEASON_COLOR_NEW"] = df_eng["SEASON"].astype(str) + "_" + df_eng["COLOR"].astype(str)
    df_eng["ITEM_CATEGORY_NEW"] = df_eng["CATEGORY"].astype(str) + "_" + df_eng["ITEM_PURCHASED"].astype(str)
    df_eng["HIGH_REVIEW_RATING_NEW"] = (df_eng["REVIEW_RATING"] >= 4).astype(int)
    df_eng["SPEND_RATING_NEW"] = df_eng["PURCHASE_AMOUNT_(USD)"] * df_eng["REVIEW_RATING"]

    top_locations = df_eng["LOCATION"].value_counts().nlargest(10).index
    df_eng["LOCATION_GROUPED_NEW"] = df_eng["LOCATION"].where(df_eng["LOCATION"].isin(top_locations), "Other")

    # --- 3. İklim Gruplaması ---
    cold_states = ["Alaska", "North Dakota", "South Dakota", "Minnesota", "Wisconsin", "Michigan", "Montana", "Wyoming", "Maine", "Vermont", "New Hampshire"]
    cool_states = ["Massachusetts", "Connecticut", "Rhode Island", "New York", "Pennsylvania", "New Jersey", "Ohio", "Indiana", "Illinois", "Iowa", "Nebraska", "Kansas", "Colorado", "Utah", "Idaho", "Washington", "Oregon"]
    warm_states = ["Virginia", "Maryland", "Delaware", "Kentucky", "Missouri", "West Virginia", "North Carolina", "Tennessee", "Arkansas", "Oklahoma"]
    hot_states = ["Florida", "Texas", "Louisiana", "Mississippi", "Alabama", "Georgia", "South Carolina", "Arizona", "Nevada", "New Mexico", "California"]
    tropical_states = ["Hawaii"]

    def climate_group(state):
        if state in cold_states: return "Cold"
        elif state in cool_states: return "Cool"
        elif state in warm_states: return "Warm"
        elif state in hot_states: return "Hot"
        elif state in tropical_states: return "Tropical"
        else: return "Unknown"

    df_eng["CLIMATE_GROUP_NEW"] = df_eng["LOCATION"].apply(climate_group)
    
    top_locations = df_eng["LOCATION"].value_counts().nlargest(10).index
    df_eng["TOP_LOCATION_NEW"] = df_eng["LOCATION"].where(df_eng["LOCATION"].isin(top_locations), "Other")

    # --- 4. Aggregation Featureları ---
    df_eng = add_conditional_freq_feature(df_eng, "CLIMATE_GROUP_NEW", "CATEGORY", prefix="P_CATEGORY_given_CLIMATE_NEW", smoothing=1.0)
    df_eng = add_conditional_freq_feature(df_eng, "CLIMATE_GROUP_NEW", "SIZE", prefix="P_SIZE_given_CLIMATE_NEW", smoothing=1.0)
    df_eng = add_conditional_freq_feature(df_eng, "CLIMATE_GROUP_NEW", "SEASON", prefix="P_SEASON_given_CLIMATE_NEW", smoothing=1.0)

    df_eng["CLIMATE_ITEM_FIT_SCORE_NEW"] = (
        df_eng["P_CATEGORY_given_CLIMATE_NEW"] *
        df_eng["P_SIZE_given_CLIMATE_NEW"] *
        df_eng["P_SEASON_given_CLIMATE_NEW"]
    )

    climate_spend_mean = df_eng.groupby("CLIMATE_GROUP_NEW")["PURCHASE_AMOUNT_(USD)"].transform("mean")
    df_eng["PURCHASE_AMT_REL_CLIMATE_NEW"] = df_eng["PURCHASE_AMOUNT_(USD)"] / climate_spend_mean

    df_eng["CLIMATE_LOYALTY_NEW"] = (df_eng["CLIMATE_GROUP_NEW"].astype(str) + "_" + df_eng["LOYALTY_LEVEL_NEW"].astype(str))
    df_eng["LOYALTY_SCORE_NEW"] = pd.qcut(df_eng["PREVIOUS_PURCHASES"], q=4, labels=[1, 2, 3, 4]).astype(int)

    cat_spend_mean = df_eng.groupby('CATEGORY')['PURCHASE_AMOUNT_(USD)'].transform('mean')
    df_eng['REL_SPEND_CAT_NEW'] = df_eng['PURCHASE_AMOUNT_(USD)'] / cat_spend_mean

    age_spend_mean = df_eng.groupby('AGE_NEW')['PURCHASE_AMOUNT_(USD)'].transform('mean')
    df_eng['REL_SPEND_AGE_NEW'] = df_eng['PURCHASE_AMOUNT_(USD)'] / age_spend_mean

    loc_freq_mean = df_eng.groupby('CLIMATE_GROUP_NEW')['FREQUENCY_VALUE_NEW'].transform('mean')
    df_eng['REL_FREQ_CLIMATE_NEW'] = df_eng['FREQUENCY_VALUE_NEW'] / loc_freq_mean
    
    df_eng["PROMO_X_LOYALTY"] = ((df_eng["PROMO_CODE_USED"] == "Yes").astype(int) * df_eng["LOYALTY_SCORE_NEW"])
    df_eng["PROMO_X_FREQ"] = ((df_eng["PROMO_CODE_USED"] == "Yes").astype(int) * df_eng["FREQUENCY_VALUE_NEW"])

    # Target Drop
    if 'TEMP_TARGET' in df_eng.columns: df_eng.drop(columns=['TEMP_TARGET'], inplace=True)

    # --- Encoding ---
    # DİKKAT: PROMO_CODE_USED listeden çıkarıldı (senin kodundaki gibi)
    drop_cols = ['CUSTOMER_ID', 'ITEM_PURCHASED', 'LOCATION', 'COLOR', 'SIZE',
                 'FREQUENCY_OF_PURCHASES', 'PAYMENT_METHOD', 'SHIPPING_TYPE',
                 'PURCHASE_AMOUNT_(USD)', 'PREVIOUS_PURCHASES', 'REVIEW_RATING',
                 'AGE', 'DISCOUNT_APPLIED', 'SEASON', 'PROMO_CODE_USED']
    
    cols_to_drop = [c for c in drop_cols if c in df_eng.columns]
    df_model = df_eng.drop(columns=cols_to_drop)

    # Auto Encoding
    cat_cols = [col for col in df_model.columns if df_model[col].dtype == 'O' or df_model[col].dtype.name == 'category']
    binary_cols = [col for col in cat_cols if df_model[col].nunique() <= 2]
    multi_cols = [col for col in cat_cols if df_model[col].nunique() > 2]

    le = LabelEncoder()
    for col in binary_cols:
        df_model[col] = le.fit_transform(df_model[col])

    df_encoded = pd.get_dummies(df_model, columns=multi_cols, drop_first=False)
    
    return df_eng, df_encoded

# =============================================================================
# UYGULAMA ARAYÜZÜ
# =============================================================================

st.title("🛍️ Alışveriş Davranışları: Hibrit Analitik Paneli")
st.markdown("""
Bu panel; **Segmentasyon (Unsupervised)** ve **Abonelik Tahmini (Supervised)** yöntemlerini 
Precision odaklı bir strateji ile birleştirir.
""")

# --- SIDEBAR: VERİ YÜKLEME ---
st.sidebar.header("📂 Veri Yönetimi")
uploaded_file = st.sidebar.file_uploader("CSV Dosyanızı Buraya Sürükleyin", type=["csv"])

if uploaded_file is None:
    st.info("Analize başlamak için lütfen 'shopping_behavior_updated.csv' dosyasını yükleyin.")
    st.stop()

# --- VERİ YÜKLEME VE İŞLEME (CACHE) ---
@st.cache_data
def get_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.upper().str.replace(" ", "_").str.strip()
    return df

df_raw = get_data(uploaded_file)

# GLOBAL DEĞİŞKENLERİ SAKLAMAK İÇİN SESSION STATE
if 'best_threshold' not in st.session_state:
    st.session_state['best_threshold'] = 0.50 # Varsayılan

# Sekmeler
tab_eda, tab_seg, tab_model, tab_sim = st.tabs(["📊 EDA", "🧩 Segmentasyon", "🎯 Model (RF+Precision)", "🧪 Simülatör"])

# A) VERİ İŞLEME (PIPELINE)
with st.spinner('Veri işleniyor, modeller eğitiliyor ve optimal eşik değeri aranıyor...'):
    
    # 1. Rare & Correlation
    df_rare = rare_encoder(df_raw, 0.01)
    if 'DISCOUNT_APPLIED' in df_rare.columns and 'PROMO_CODE_USED' in df_rare.columns:
        cv_score = cramers_v(df_rare['DISCOUNT_APPLIED'], df_rare['PROMO_CODE_USED'])
        if cv_score > 0.8:
            df_rare.drop(columns=['DISCOUNT_APPLIED'], inplace=True)
            
    # 2. Main Pipeline
    df_eng, df_encoded = process_data_pipeline(df_rare)

# --- TAB 1: EDA ---
with tab_eda:
    st.header("Veri Genel Bakış")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Müşteri Sayısı", df_raw.shape[0])
    k2.metric("Ortalama Yaş", f"{df_raw['AGE'].mean():.1f}")
    k3.metric("Abonelik Oranı", f"%{(df_raw['SUBSCRIPTION_STATUS']=='Yes').mean()*100:.1f}")
    k4.metric("Ortalama Harcama", f"${df_raw['PURCHASE_AMOUNT_(USD)'].mean():.1f}")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Harcama Dağılımı")
        fig, ax = plt.subplots()
        sns.histplot(df_raw['PURCHASE_AMOUNT_(USD)'], kde=True, ax=ax, color='skyblue')
        st.pyplot(fig)
    with c2:
        st.subheader("Kategori Dağılımı")
        fig, ax = plt.subplots()
        sns.countplot(y=df_raw['CATEGORY'], order=df_raw['CATEGORY'].value_counts().index, ax=ax, palette='viridis')
        st.pyplot(fig)

# --- TAB 2: SEGMENTASYON ---
with tab_seg:
    st.header("K-Means Müşteri Segmentasyonu")
    
    segmentation_cols = [
        'PURCHASE_AMOUNT_(USD)', 'PREVIOUS_PURCHASES', 
        'FREQUENCY_VALUE_NEW', 'PROMO_CODE_USED', 
        'SPEND_PER_PURCHASE_NEW', 'LOYALTY_SCORE_NEW', 'CLIMATE_LOYALTY_NEW'
    ]
    
    X_seg = df_encoded[[c for c in segmentation_cols if c in df_encoded.columns]].copy()
    X_seg.fillna(0, inplace=True)
    
    scaler_seg = StandardScaler()
    X_scaled = scaler_seg.fit_transform(X_seg)
    
    # Elbow
    wcss = []
    k_range = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        wcss.append(km.inertia_)
        
    # Optimal K
    p1 = np.array([k_range[0], wcss[0]])
    p2 = np.array([k_range[-1], wcss[-1]])
    dists = [np.abs(np.cross(p2-p1, p1-np.array([k_range[i], wcss[i]]))) / np.linalg.norm(p2-p1) for i in range(len(wcss))]
    optimal_k = k_range[np.argmax(dists)]
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.info(f"**Optimal Küme Sayısı (K): {optimal_k}**")
        fig_elb, ax = plt.subplots()
        plt.plot(k_range, wcss, 'bo--')
        plt.axvline(optimal_k, color='r', linestyle='--')
        plt.title("Elbow Method")
        st.pyplot(fig_elb)
        
    with c2:
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        pca = PCA(n_components=2)
        comps = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(comps, columns=['PC1', 'PC2'])
        df_pca['Cluster'] = clusters
        
        fig_pca, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca, palette='viridis', ax=ax)
        plt.title(f"Segment Dağılımı (K={optimal_k})")
        st.pyplot(fig_pca)
        
    # Profil
    df_report = df_eng.copy()
    df_report['Cluster'] = clusters
    df_report['PROMO_USED_VAL'] = df_report['PROMO_CODE_USED'].apply(lambda x: 1 if x=='Yes' else 0)
    
    st.subheader("Segment Profilleri")
    profile = df_report.groupby('Cluster')[['AGE', 'TOTAL_SPEND_WEIGHTED_NEW', 'CLIMATE_ITEM_FIT_SCORE_NEW', 'PROMO_USED_VAL']].mean()
    profile['PROMO_USED_VAL'] = profile['PROMO_USED_VAL'] * 100
    st.dataframe(profile.style.background_gradient(cmap='Blues'))

# --- TAB 3: MODEL (RF + PRECISION + THRESHOLD) ---
with tab_model:
    st.header("Random Forest Modeli (Precision Odaklı)")
    
    # 1. Veri Hazırlığı & Feature Selection (0.01 threshold)
    leakage_cols = [c for c in df_encoded.columns if 'SUB_FREQ_NEW' in c or 'PROMO_NO_SUB_NEW' in c or 'SHIP_SUB_NEW' in c]
    X_temp = df_encoded.drop(columns=['SUBSCRIPTION_STATUS'] + leakage_cols)
    y = df_encoded['SUBSCRIPTION_STATUS']
    
    # Feature Selection (RF ile)
    rf_sel = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_sel.fit(X_temp, y)
    importances = pd.Series(rf_sel.feature_importances_, index=X_temp.columns)
    keep_cols = importances[importances >= 0.01].index.tolist()
    
    X = X_temp[keep_cols] # Sadece önemli featurelar
    
    # Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    scaler_model = StandardScaler()
    X_train_s = scaler_model.fit_transform(X_train)
    X_test_s = scaler_model.transform(X_test)
    
    # 2. Model Eğitimi (Random Forest - Precision)
    with st.spinner("Random Forest 'Precision' hedefiyle optimize ediliyor..."):
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [8, 12, 16],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        grid = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), 
                            rf_params, cv=3, scoring='precision', n_jobs=-1)
        grid.fit(X_train_s, y_train)
        final_model = grid.best_estimator_
        
        y_proba = final_model.predict_proba(X_test_s)[:, 1]
    
    # 3. Threshold Optimizasyonu
    st.subheader("Otomatik Threshold (Eşik Değeri) Optimizasyonu")
    
    def eval_thr(y_true, y_prob, thr):
        y_p = (y_prob >= thr).astype(int)
        return {"thr": thr, "precision": precision_score(y_true, y_p, zero_division=0), "recall": recall_score(y_true, y_p, zero_division=0)}
        
    thresholds = np.linspace(0.05, 0.95, 19)
    res = [eval_thr(y_test, y_proba, t) for t in thresholds]
    df_thr = pd.DataFrame(res)
    
    # Hedef Recall: %80 (En az %80'ini yakala, ama Precision'ı maksimize et)
    target_recall = 0.80
    candidates = df_thr[df_thr["recall"] >= target_recall].sort_values("precision", ascending=False)
    
    if not candidates.empty:
        best_thr = candidates.iloc[0]["thr"]
        best_prec = candidates.iloc[0]["precision"]
        best_rec = candidates.iloc[0]["recall"]
    else:
        best_thr = 0.50
        best_prec = 0.0
        best_rec = 0.0
    
    # Session State'e kaydet (Simülatör için)
    st.session_state['best_threshold'] = best_thr
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Önerilen Threshold", f"{best_thr:.2f}")
    c2.metric("Beklenen Precision", f"%{best_prec*100:.1f}")
    c3.metric("Beklenen Recall", f"%{best_rec*100:.1f}")
    
    st.line_chart(df_thr.set_index("thr")[['precision', 'recall']])
    
    # Feature Importance
    st.subheader("Feature Importance (Model Kararını Etkileyenler)")
    imp = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    fig_imp, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=imp.values, y=imp.index, palette='viridis', ax=ax)
    st.pyplot(fig_imp)

# --- TAB 4: SİMÜLATÖR ---
with tab_sim:
    st.header(f"Canlı Tahmin Simülatörü (Threshold: {st.session_state['best_threshold']:.2f})")
    
    with st.form("sim_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.slider("Yaş", 18, 70, 30)
        gender = c2.selectbox("Cinsiyet", df_raw['GENDER'].unique())
        spend = c3.number_input("Harcama Tutarı ($)", 1, 1000, 100)
        prev = c1.number_input("Geçmiş Alışveriş", 0, 100, 5)
        freq = c2.selectbox("Sıklık", df_raw['FREQUENCY_OF_PURCHASES'].unique())
        rating = c3.slider("Rating", 1.0, 5.0, 4.0)
        cat = c1.selectbox("Kategori", df_raw['CATEGORY'].unique())
        loc = c2.selectbox("Lokasyon", df_raw['LOCATION'].unique())
        promo = c3.selectbox("Promosyon Kullandı mı?", ["Yes", "No"])
        
        # Diğer zorunlu alanlar
        item = df_raw['ITEM_PURCHASED'].mode()[0]
        color = df_raw['COLOR'].mode()[0]
        size = df_raw['SIZE'].mode()[0]
        season = df_raw['SEASON'].mode()[0]
        pay = df_raw['PAYMENT_METHOD'].mode()[0]
        ship = df_raw['SHIPPING_TYPE'].mode()[0]
        
        btn = st.form_submit_button("Tahmin Et")
        
    if btn:
        # 1. Tek satırlık DataFrame
        input_row = pd.DataFrame({
            'CUSTOMER_ID': [999999],
            'AGE': [age], 'GENDER': [gender], 'ITEM_PURCHASED': [item],
            'CATEGORY': [cat], 'PURCHASE_AMOUNT_(USD)': [spend],
            'LOCATION': [loc], 'SIZE': [size], 'COLOR': [color],
            'SEASON': [season], 'REVIEW_RATING': [rating],
            'SHIPPING_TYPE': [ship], 'DISCOUNT_APPLIED': ['No'],
            'PROMO_CODE_USED': [promo], 'PREVIOUS_PURCHASES': [prev],
            'PAYMENT_METHOD': [pay], 'FREQUENCY_OF_PURCHASES': [freq],
            'SUBSCRIPTION_STATUS': ['No']
        })
        
        # 2. Pipeline'dan geçir (Encoding)
        full_df = pd.concat([df_raw, input_row], axis=0, ignore_index=True)
        _, full_encoded = process_data_pipeline(full_df) # df_rare yerine full_df gönderiyoruz ki logic işlesin
        
        # 3. Son satırı al ve Feature Selection'a uyumlu hale getir
        user_row = full_encoded.iloc[[-1]].drop(columns=['SUBSCRIPTION_STATUS'] + leakage_cols)
        user_row = user_row.reindex(columns=X.columns, fill_value=0) # Feature selection ile seçilenlere eşitle
        
        # 4. Scale ve Tahmin
        user_s = scaler_model.transform(user_row)
        prob = final_model.predict_proba(user_s)[0][1]
        
        # 5. Karar (Optimize edilmiş threshold ile)
        thr = st.session_state['best_threshold']
        
        st.divider()
        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            if prob >= thr:
                st.success(f"## ✅ ABONE OLUR\n### İhtimal: %{prob*100:.1f}")
                st.caption(f"(Model %{thr*100:.0f} üzeri emin olduğu için bu kararı verdi)")
            else:
                st.error(f"## ❌ ABONE OLMAZ\n### İhtimal: %{prob*100:.1f}")
        with col_r2:
            st.info("**Karar Açıklaması:**")
            st.write(f"- Bu müşteri {age} yaşında ve {loc} lokasyonundan.")
            st.write(f"- Harcama Skoru ve Sadakat verileri incelendi.")
            st.progress(prob)
