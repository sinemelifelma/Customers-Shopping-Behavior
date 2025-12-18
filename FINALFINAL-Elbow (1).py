################################################################
# Alışveriş Davranışları Tahmini (TÜM GÖRSELLER AKTİF)
################################################################

######################################
# KÜTÜPHANELER
######################################

import os
import warnings
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Modelleme
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Veri Ön İşleme ve Değerlendirme
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report, precision_score, recall_score, f1_score)
# =============================================================================
# 1. AYARLAR (SETTINGS)
# =============================================================================
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings('ignore')
logging.getLogger('lightgbm').setLevel(logging.ERROR)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
sns.set(style="whitegrid")


# =============================================================================
# 2. EDA VE YARDIMCI FONKSİYONLAR
# =============================================================================
def check_df(dataframe):
    print("\n##################### GENEL RESİM (CHECK DF) #####################")
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    num_cols = dataframe.select_dtypes(include=["float64", "int64"]).columns
    print(dataframe[num_cols].quantile([0, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).T)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"\nObservations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, cat_but_car, num_cols


def checking_outlier(list_feature, df):
    print("\n##################### OUTLIER ANALİZİ #####################")
    outlier_info = []
    for feature in list_feature:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)][feature]

        if len(outliers) > 0:
            outlier_info.append({"Feature": feature, "Outlier Count": len(outliers)})

    if len(outlier_info) == 0:
        print("✅ No outliers detected in the selected features.")
    else:
        print(pd.DataFrame(outlier_info))


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
    return temp_df


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


def add_conditional_freq_feature(df, group_col, cat_col, prefix=None, smoothing=1.0):
    if prefix is None:
        prefix = f"P_{cat_col}_given_{group_col}"
    ct = pd.crosstab(df[group_col], df[cat_col])
    ct_smoothed = ct + smoothing
    probs = ct_smoothed.div(ct_smoothed.sum(axis=1), axis=0)
    feat = df[[group_col, cat_col]].apply(lambda r: probs.loc[r[group_col], r[cat_col]], axis=1)
    df[f"{prefix}"] = feat
    return df


# --- PRINT FONKSİYONLARI ---
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("##########################################")
    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


# =============================================================================
# 3. YÜKLEME VE İŞLEME FONKSİYONLARI (Feature Engineering)
# =============================================================================

def load_data(file_path):
    print("\n##################### 1. VERİ YÜKLEME #####################")
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.upper().str.replace(" ", "_").str.strip()
        print(f"> Veri seti yüklendi. Boyut: {df.shape}")
        return df
    except FileNotFoundError:
        print("HATA: Dosya bulunamadı! Lütfen dosya yolunu kontrol edin.")
        return None


def process_data(df):
    print("\n##################### 4. FEATURE ENGINEERING #####################")
    df_eng = df.copy()

    # Abonelik tahmini için target değişkeninde çalışmak yerine geçici bir target oluşturuyoruz.
    df_eng['TEMP_TARGET'] = df_eng['SUBSCRIPTION_STATUS'].map({"Yes": 1, "No": 0})

    # =====================================================
    # 1. TEMEL DEĞİŞKENLER
    # =====================================================
    df_eng['TOTAL_SPEND_WEIGHTED_NEW'] = df_eng['PREVIOUS_PURCHASES'] * df_eng['PURCHASE_AMOUNT_(USD)']
    df_eng['SPEND_PER_PURCHASE_NEW'] = df_eng['PURCHASE_AMOUNT_(USD)'] / (df_eng['PREVIOUS_PURCHASES'] + 1)

    freq_map = {'Weekly': 52, 'Bi-Weekly': 26, 'Fortnightly': 26, 'Quarterly': 4, 'Annually': 1, 'Monthly': 12,
                'Every 3 Months': 4}
    df_eng['FREQUENCY_VALUE_NEW'] = df_eng['FREQUENCY_OF_PURCHASES'].map(freq_map)

    pay_map = {'Cash': 'Cash', 'Credit Card': 'Card', 'Debit Card': 'Card', 'PayPal': 'Online', 'Venmo': 'Online',
               'Bank Transfer': 'Online'}
    df_eng['PAYMENT_TYPE_NEW'] = df_eng['PAYMENT_METHOD'].map(pay_map)

    # =====================================================
    # 2. ÖMER/SİNEM FEATURELAR
    # =====================================================
    df_eng["AGE_NEW"] = pd.cut(df_eng["AGE"], bins=[0, 30, 50, 56, df_eng["AGE"].max()],
                               labels=["18-30", "31-45", "46-56", "57-70"])
    df_eng["PURCHASE_AMOUNT_(USD)_NEW"] = pd.qcut(df_eng["PURCHASE_AMOUNT_(USD)"], q=4,
                                                  labels=["Low", "Mid", "High", "Very High"])
    df_eng["LOYALTY_LEVEL_NEW"] = pd.cut(df_eng["PREVIOUS_PURCHASES"], bins=[0, 13, 25, 38, 50],
                                         labels=["Low", "Mid", "High", "Very High"], include_lowest=True)

    # --- LEAKAGE FEATURELAR - Model kurulurken çıkarılacak segmentation kullanılacak ---
    df_eng["SUB_FREQ_NEW"] = (df_eng["TEMP_TARGET"].astype(str) + "_" + df_eng["FREQUENCY_OF_PURCHASES"].astype(str))
    df_eng["PROMO_NO_SUB_NEW"] = ((df_eng["PROMO_CODE_USED"] == "Yes") & (df_eng["TEMP_TARGET"] == 0)).astype(int)
    df_eng["SHIP_SUB_NEW"] = (df_eng["SHIPPING_TYPE"].astype(str) + "_" + df_eng["TEMP_TARGET"].astype(str))

    # --- SİNEM FEATURELAR ---
    df_eng["SEASON_CATEGORY_NEW"] = df_eng["SEASON"].astype(str) + "_" + df_eng["CATEGORY"].astype(str)
    df_eng["SEASON_COLOR_NEW"] = df_eng["SEASON"].astype(str) + "_" + df_eng["COLOR"].astype(str)
    df_eng["ITEM_CATEGORY_NEW"] = df_eng["CATEGORY"].astype(str) + "_" + df_eng["ITEM_PURCHASED"].astype(str)
    df_eng["HIGH_REVIEW_RATING_NEW"] = (df_eng["REVIEW_RATING"] >= 4).astype(int)
    df_eng["SPEND_RATING_NEW"] = df_eng["PURCHASE_AMOUNT_(USD)"] * df_eng["REVIEW_RATING"]

    top_locations = df_eng["LOCATION"].value_counts().nlargest(10).index
    df_eng["LOCATION_GROUPED_NEW"] = df_eng["LOCATION"].where(df_eng["LOCATION"].isin(top_locations), "Other")

    # =====================================================
    # 3. SİNEM İKLİM GRUPLAMASI
    # =====================================================
    cold_states = ["Alaska", "North Dakota", "South Dakota", "Minnesota", "Wisconsin", "Michigan", "Montana", "Wyoming",
                   "Maine", "Vermont", "New Hampshire"]
    cool_states = ["Massachusetts", "Connecticut", "Rhode Island", "New York", "Pennsylvania", "New Jersey", "Ohio",
                   "Indiana", "Illinois", "Iowa", "Nebraska", "Kansas", "Colorado", "Utah", "Idaho", "Washington",
                   "Oregon"]
    warm_states = ["Virginia", "Maryland", "Delaware", "Kentucky", "Missouri", "West Virginia", "North Carolina",
                   "Tennessee", "Arkansas", "Oklahoma"]
    hot_states = ["Florida", "Texas", "Louisiana", "Mississippi", "Alabama", "Georgia", "South Carolina", "Arizona",
                  "Nevada", "New Mexico", "California"]
    tropical_states = ["Hawaii"]

    def climate_group(state):
        if state in cold_states:
            return "Cold"
        elif state in cool_states:
            return "Cool"
        elif state in warm_states:
            return "Warm"
        elif state in hot_states:
            return "Hot"
        elif state in tropical_states:
            return "Tropical"
        else:
            return "Unknown"

    df_eng["CLIMATE_GROUP_NEW"] = df_eng["LOCATION"].apply(climate_group)

    top_locations = df_eng["LOCATION"].value_counts().nlargest(10).index

    df_eng["TOP_LOCATION_NEW"] = df_eng["LOCATION"].where(df_eng["LOCATION"].isin(top_locations), "Other")


    # =====================================================
    # 4. SİNEM AGGREGATION FEATURELARI
    # =====================================================
    df_eng = add_conditional_freq_feature(df_eng, "CLIMATE_GROUP_NEW", "CATEGORY",
                                          prefix="P_CATEGORY_given_CLIMATE_NEW", smoothing=1.0)
    df_eng = add_conditional_freq_feature(df_eng, "CLIMATE_GROUP_NEW", "SIZE", prefix="P_SIZE_given_CLIMATE_NEW",
                                          smoothing=1.0)
    df_eng = add_conditional_freq_feature(df_eng, "CLIMATE_GROUP_NEW", "SEASON", prefix="P_SEASON_given_CLIMATE_NEW",
                                          smoothing=1.0)

    df_eng["CLIMATE_ITEM_FIT_SCORE_NEW"] = (
            df_eng["P_CATEGORY_given_CLIMATE_NEW"] *
            df_eng["P_SIZE_given_CLIMATE_NEW"] *
            df_eng["P_SEASON_given_CLIMATE_NEW"]
    )

    climate_spend_mean = df_eng.groupby("CLIMATE_GROUP_NEW")["PURCHASE_AMOUNT_(USD)"].transform("mean")
    df_eng["PURCHASE_AMT_REL_CLIMATE_NEW"] = df_eng["PURCHASE_AMOUNT_(USD)"] / climate_spend_mean

    df_eng["CLIMATE_LOYALTY_NEW"] = (
            df_eng["CLIMATE_GROUP_NEW"].astype(str) + "_" + df_eng["LOYALTY_LEVEL_NEW"].astype(str)
    )

    df_eng["LOYALTY_SCORE_NEW"] = pd.qcut(df_eng["PREVIOUS_PURCHASES"], q=4, labels=[1, 2, 3, 4]).astype(int)

    cat_spend_mean = df_eng.groupby('CATEGORY')['PURCHASE_AMOUNT_(USD)'].transform('mean')
    df_eng['REL_SPEND_CAT_NEW'] = df_eng['PURCHASE_AMOUNT_(USD)'] / cat_spend_mean

    age_spend_mean = df_eng.groupby('AGE_NEW')['PURCHASE_AMOUNT_(USD)'].transform('mean')
    df_eng['REL_SPEND_AGE_NEW'] = df_eng['PURCHASE_AMOUNT_(USD)'] / age_spend_mean

    loc_freq_mean = df_eng.groupby('CLIMATE_GROUP_NEW')['FREQUENCY_VALUE_NEW'].transform('mean')
    df_eng['REL_FREQ_CLIMATE_NEW'] = df_eng['FREQUENCY_VALUE_NEW'] / loc_freq_mean

    df_eng["PROMO_X_LOYALTY"] = (
            (df_eng["PROMO_CODE_USED"] == "Yes").astype(int) * df_eng["LOYALTY_SCORE_NEW"]
    )

    df_eng["PROMO_X_FREQ"] = (
            (df_eng["PROMO_CODE_USED"] == "Yes").astype(int) * df_eng["FREQUENCY_VALUE_NEW"]
    )

    '''promo_rate_by_cluster = (
        df_eng.groupby("CLUSTER_ID")["PROMO_CODE_USED"]
        .apply(lambda x: (x == "Yes").mean())
    )

    df_eng["CLUSTER_PROMO_RATE"] = df_eng["CLUSTER_ID"].map(promo_rate_by_cluster)'''


    # Geçici targeti drop ettik
    df_eng.drop(columns=['TEMP_TARGET'], inplace=True)

    # --- ENCODING ---
    drop_cols = ['CUSTOMER_ID', 'ITEM_PURCHASED', 'LOCATION', 'COLOR', 'SIZE',
                 'FREQUENCY_OF_PURCHASES', 'PAYMENT_METHOD', 'SHIPPING_TYPE',
                 'PURCHASE_AMOUNT_(USD)', 'PREVIOUS_PURCHASES', 'REVIEW_RATING',
                 'AGE', 'DISCOUNT_APPLIED', 'SEASON', 'PROMO_CODE_USED']

    cols_to_drop = [c for c in drop_cols if c in df_eng.columns]
    df_model = df_eng.drop(columns=cols_to_drop)

    # =========================================================================
    # *** ENCODING***
    # =========================================================================

    cat_cols = [col for col in df_model.columns if df_model[col].dtype == 'O' or df_model[col].dtype.name == 'category']
    binary_cols = [col for col in cat_cols if df_model[col].nunique() <= 2]
    multi_cols = [col for col in cat_cols if df_model[col].nunique() > 2]

    print(f"\n> Encoding: {len(binary_cols)} Binary, {len(multi_cols)} Multi-Class Kolon bulundu.")

    le = LabelEncoder()
    for col in binary_cols:
        df_model[col] = le.fit_transform(df_model[col])

    df_encoded = pd.get_dummies(df_model, columns=multi_cols, drop_first=False)

    print(f">>> İşlem Sonrası Değişken Sayısı: {df_encoded.shape[1]}") #enconding ler tamamlandıktan sonra toplamda kaç feature olduğu bilgisi.
    return df_eng, df_encoded


# =============================================================================
# 4. ANALİZ VE MODELLEME PIPELINE
# =============================================================================
def run_analysis_pipeline(df_raw, verbose_eda=False):
    # --- KLASÖR AYARLARI -> Görsel çıktılarını istediğimiz bir klasör içerisine yazdırması için ---
    output_dir = "RF Visuals"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n> '{output_dir}' klasörü oluşturuldu/kontrol edildi.")

    # --- Exploratory Data Analysis (EDA) ---
    print("\n##################### 2. EDA & VERİ HAZIRLIĞI #####################")

    check_df(df_raw)
    cat_cols, cat_but_car, num_cols = grab_col_names(df_raw)
    missing_values_table(df_raw)
    if 'CUSTOMER_ID' in num_cols: num_cols.remove('CUSTOMER_ID')
    checking_outlier(num_cols, df_raw)

    if verbose_eda:
        print("\n-------- DETAYLI DEĞİŞKEN ANALİZİ (VERBOSE MODE ON) --------")
        for col in cat_cols: cat_summary(df_raw, col, plot=True)
        for col in num_cols: num_summary(df_raw, col, plot=True)
        print("------------------------------------------------------------\n")

    df_rare = rare_encoder(df_raw, 0.01)

    print("\n--- Korelasyon ve İlişki Analizi ---")
    if 'DISCOUNT_APPLIED' in df_rare.columns and 'PROMO_CODE_USED' in df_rare.columns:
        cv_score = cramers_v(df_rare['DISCOUNT_APPLIED'], df_rare['PROMO_CODE_USED'])
        if cv_score > 0.8:
            print(">>> Yüksek ilişki tespit edildi! 'DISCOUNT_APPLIED' drop edildi.")
            df_rare.drop(columns=['DISCOUNT_APPLIED'], inplace=True)

    df_eng, df_encoded = process_data(df_rare)

    # --- SEGMENTASYON (K-Means) ---
    print("\n##################### 3. SEGMENTASYON ANALİZİ #####################")
    # Segmentasyonda kullanacağımız columnları buraya ekleyebiliriz. Şimdilik bu şekilde tuttum.
    segmentation_features_numeric = [
        'PURCHASE_AMOUNT_(USD)',
        'PREVIOUS_PURCHASES',
        'FREQUENCY_VALUE_NEW',
        'PROMO_CODE_USED',
        'SPEND_PER_PURCHASE_NEW',
        'LOYALTY_SCORE_NEW',
        'CLIMATE_LOYALTY_NEW'
    ]

    # df_encoded'dan çekiyoruz
    X_seg = df_encoded[[c for c in segmentation_features_numeric if c in df_encoded.columns]].copy()
    X_seg.fillna(0, inplace=True)



    print(f"> Segmentasyon sadece {len(segmentation_features_numeric)} adet sayısal değişken ile yapılıyor.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seg)

    print("> Optimal K aranıyor (Sadece Elbow Method)...")

    # --- ELBOW METHOD HESAPLAMA ---
    wcss = []
    k_range = range(2, 11)

    for k in k_range:
        kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        wcss.append(kmeans_test.inertia_)
        print(f"   K={k} -> WCSS: {kmeans_test.inertia_:.2f}")

    # ELBOW POINT HESAPLAMA (Geometrik Yöntem)
    p1 = np.array([k_range[0], wcss[0]])
    p2 = np.array([k_range[-1], wcss[-1]])
    distances = []
    for i in range(len(wcss)):
        p = np.array([k_range[i], wcss[i]])
        # Noktanın doğruya olan dik uzaklığı
        dist = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distances.append(dist)

    # En uzak nokta = Dirsek noktası
    optimal_k = k_range[np.argmax(distances)]
    print(f"\n> Optimal Küme Sayısı (Elbow Method - Geometrik): {optimal_k}")

    # Grafik: Elbow
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, marker='o', linestyle='--', color='b')
    plt.title(f'Elbow Method (Optimal K={optimal_k})')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'elbow_method.png'))
    plt.show()
    plt.close()

    # KMeans
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # PCA Görselleştirme
    pca = PCA(n_components=2)
    comps = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(comps, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca, palette='viridis', s=80, alpha=0.8)
    plt.title(f'Müşteri Segmentleri (PCA - K={optimal_k})')
    plt.savefig(os.path.join(output_dir, 'pca_segments.png'))
    plt.show()
    plt.close()

    print("\n##################### SEGMENT PROFİLLERİ #####################")
    df_report = df_eng.copy()
    df_report['Cluster'] = clusters

    numeric_cols = ['AGE', 'TOTAL_SPEND_WEIGHTED_NEW', 'CLIMATE_ITEM_FIT_SCORE_NEW', 'REL_SPEND_CAT_NEW']
    segment_stats = df_report.groupby('Cluster')[numeric_cols].mean()
    fav_climate = df_report.groupby('Cluster')['CLIMATE_GROUP_NEW'].agg(lambda x: x.mode()[0])

    print(
        f"{'ID':<3} | {'Yaş':<4} | {'Harcama($)':<10} | {'FitScore':<8} | {'RelSpend':<8} | {'Promo%':<6} | {'İklim':<10}")
    print("-" * 80)
    for cluster_id, row in segment_stats.iterrows():
        print(
            f"{cluster_id:<3} | {row['AGE']:<4.1f} | {row['TOTAL_SPEND_WEIGHTED_NEW']:<10.1f} | {row['CLIMATE_ITEM_FIT_SCORE_NEW']:<8.4f} | {row['REL_SPEND_CAT_NEW']:<8.2f} | {fav_climate[cluster_id]:<10}")

    # --- MODELLEME ÖNCESİ LEAKAGE TEMİZLİĞİ ---
    leakage_cols = [c for c in df_encoded.columns if
                    'SUB_FREQ_NEW' in c or 'PROMO_NO_SUB_NEW' in c or 'SHIP_SUB_NEW' in c]

    X_temp = df_encoded.drop(columns=['SUBSCRIPTION_STATUS'] + leakage_cols)
    y = df_encoded['SUBSCRIPTION_STATUS']
    print(f"\n>>> Modelleme için Leakage yapan {len(leakage_cols)} kolon çıkarıldı.")

    print("\n>>> Feature Selection Başlıyor (Threshold: 0.01)...")

    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X_temp, y)

    importances = pd.Series(rf_selector.feature_importances_, index=X_temp.columns)

    threshold = 0.01
    low_importance_cols = importances[importances < threshold].index.tolist()
    keep_cols = importances[importances >= threshold].index.tolist()

    print(f"   - Toplam Değişken: {len(X_temp.columns)}")
    print(f"   - Elenen (Gürültülü) Değişkenler (<{threshold}): {len(low_importance_cols)} adet")
    print(f"   - Kalan (Güçlü) Değişkenler: {len(keep_cols)} adet")

    if len(low_importance_cols) > 0:
        print(f"   - Örnek Elenenler: {low_importance_cols[:5]}...")
    X = X_temp[keep_cols]


    # Stratify ile dengeli split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    scaler_model = StandardScaler()
    X_train_s = scaler_model.fit_transform(X_train)
    X_test_s = scaler_model.transform(X_test)

    # --- MODEL KARŞILAŞTIRMA (5-FOLD CV) - Sinemden aynı şekilde ekledim. ---
    print("\n##################### 4. MODEL KARŞILAŞTIRMA (5-FOLD CV) #####################")
    models = [("LogisticRegression", LogisticRegression(max_iter=1000)),
              ("RandomForest", RandomForestClassifier(random_state=42, class_weight='balanced')),
              ("XGBoost", XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42)),
              ("LightGBM", LGBMClassifier(random_state=42, verbose=-1))]

    best_model_name = ""
    best_model_score = -1

    print(f"{'Model':<20} | {'CV AUC (Mean)':<15} | {'Std Dev':<10}")
    print("-" * 50)

    for name, model in models:
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()

        print(f"{name:<20} | {mean_score:<15.4f} | {std_score:<10.4f}")

        if mean_score > best_model_score:
            best_model_score = mean_score
            best_model_name = name
            best_model_instance = model

    print(f"\n>>> KAZANAN MODEL: {best_model_name} (AUC: {best_model_score:.4f})")

    # --- FİNAL MODEL OPTİMİZASYONU ---
    print(f"\n##################### 5. FİNAL MODEL ({best_model_name}) OPTİMİZASYONU #####################")

    if best_model_name == "RandomForest":
        params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [8, 12, 16],  # Biraz daha derinleşmesine izin veriyoruz
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        estimator = RandomForestClassifier(random_state=42, class_weight='balanced')
    elif best_model_name == "XGBoost":
        params = {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'scale_pos_weight': [1, 2]  # Dengesiz veri setleri için kritik
        }
        estimator = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42)
    elif best_model_name == "LightGBM":
        params = {'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [100, 200], 'num_leaves': [20, 31, 50]}
        estimator = LGBMClassifier(random_state=42, verbose=-1)
    else:  # LogReg
        params = {'C': [0.1, 1, 10]}
        estimator = LogisticRegression(max_iter=1000)

    print(f">>> {best_model_name} için Detaylı GridSearch çalışıyor...")
    grid = GridSearchCV(estimator, params, cv=5, scoring='precision', n_jobs=-1)
    grid.fit(X_train_s, y_train)
    final_model = grid.best_estimator_
    print(f"En İyi Parametreler: {grid.best_params_}")

    y_pred = final_model.predict(X_test_s)
    if hasattr(final_model, "predict_proba"):
        y_proba = final_model.predict_proba(X_test_s)[:, 1]
        final_auc = roc_auc_score(y_test, final_model.predict_proba(X_test_s)[:, 1])
    else:
        final_auc = roc_auc_score(y_test, y_pred)

        # -------------------------------------------------------------------------
        # EKLENEN KISIM: THRESHOLD OPTIMIZATION (SINEM'IN KODU)
        # -------------------------------------------------------------------------
        print("\n##################### THRESHOLD (EŞİK DEĞERİ) OPTİMİZASYONU #####################")

        # Olasılıkları hesapla (Eğer daha önce hesaplanmadıysa)
        if hasattr(final_model, "predict_proba"):
            y_proba = final_model.predict_proba(X_test_s)[:, 1]
        else:
            # Desteklemeyen modeller için (SVM vb.) 0/1 tahmini kullanılır
            y_proba = y_pred

        def eval_threshold(y_true, y_prob_vals, thr):
            y_pred_thr = (y_prob_vals >= thr).astype(int)
            return {
                "thr": thr,
                "precision": precision_score(y_true, y_pred_thr, zero_division=0),
                "recall": recall_score(y_true, y_pred_thr, zero_division=0),
                "f1": f1_score(y_true, y_pred_thr, zero_division=0),
                "cm": confusion_matrix(y_true, y_pred_thr)
            }

        thresholds = np.linspace(0.05, 0.95, 19)
        rows = []

        for thr in thresholds:
            r = eval_threshold(y_test, y_proba, thr)
            rows.append([r["thr"], r["precision"], r["recall"], r["f1"]])

        thr_df = pd.DataFrame(rows, columns=["thr", "precision", "recall", "f1"])

        # Hedef Recall: %90 (Abonelerin %90'ını kaçırmamak istiyoruz)
        target_recall = 0.80

        candidates = thr_df[thr_df["recall"] >= target_recall].copy()

        if not candidates.empty:
            candidates = candidates.sort_values("precision", ascending=False)
            best_thr = candidates.iloc[0]["thr"]
            print(f"\n>>> Hedeflenen Recall (>{target_recall}) için En İyi Eşik Değeri: {best_thr:.2f}")
            print("\n>>> Aday Eşik Değerleri Tablosu (İlk 5):")
            print(candidates.head(5).to_string(index=False))

            print(f"\n>>> Seçilen Eşik Değeri ({best_thr:.2f}) için Confusion Matrix:")
            cm_best = eval_threshold(y_test, y_proba, best_thr)["cm"]
            print(cm_best)
        else:
            print(f"\n>>> UYARI: Recall > {target_recall} sağlayan bir eşik değeri bulunamadı.")

        print("-" * 50)
        # -------------------------------------------------------------------------

    print("\n>>> 6.1. FİNAL PERFORMANS RAPORU:")
    print(classification_report(y_test, y_pred))
    print(f"> Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"> AUC: {final_auc:.4f}")
    print(confusion_matrix(y_test, y_pred))

    # --- GRAFİK VE DETAYLI ANALİZ (ODDS RATIO) ---
    print("\n##################### 6. DEĞİŞKEN ÖNEM DÜZEYLERİ (FEATURE IMPORTANCE) #####################")
    plt.figure(figsize=(12, 10))
    # Bu kısım da, Sinem'in dosyasındaki son bölüm, Odds ratio vs olan kısımlar.
    # DURUM 1: Ağaç Bazlı Modeller
    if hasattr(final_model, 'feature_importances_'):
        importances = final_model.feature_importances_
        feature_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(100)

        print("\n>>> En Önemli 100 Değişken:")
        print(feature_imp.head(100).to_string())

        sns.barplot(x=feature_imp.values, y=feature_imp.index, palette='viridis')
        plt.title(f'Feature Importance ({best_model_name})')


    # DURUM 2: Logistic Regression
    elif hasattr(final_model, 'coef_'):
        importances = final_model.coef_[0]
        feature_imp = pd.DataFrame({'Feature': X.columns, 'Coef': importances})

        # Odds Ratio Hesaplama
        feature_imp["Odds_Ratio"] = np.exp(feature_imp["Coef"])
        feature_imp['Abs_Coef'] = feature_imp['Coef'].abs()

        # Sıralama
        feature_imp = feature_imp.sort_values(by='Abs_Coef', ascending=False)

        print("\n>>> Aboneliği En Çok ARTIRAN 5 Özellik (Pozitif Katsayı):")
        print(feature_imp[feature_imp['Coef'] > 0].head(5)[['Feature', 'Coef', 'Odds_Ratio']].to_string(index=False))

        print("\n>>> Aboneliği En Çok AZALTAN 5 Özellik (Negatif Katsayı):")
        print(feature_imp[feature_imp['Coef'] < 0].head(5)[['Feature', 'Coef', 'Odds_Ratio']].to_string(index=False))

        # Grafik (tüm featurelar)
        plot_df = feature_imp.head(10).copy()
        plot_df['Type'] = np.where(plot_df['Coef'] > 0, 'Pos', 'Neg')
        sns.barplot(x='Coef', y='Feature', data=plot_df, palette='coolwarm')
        plt.axvline(0, color='black', linestyle='--')
        plt.title(f'Feature Coefficients ({best_model_name}) - Pozitif/Negatif Etki')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.show()
    plt.close()

    # RF için Ağaç Görseli
    if best_model_name == "RandomForest":
        plt.figure(figsize=(24, 12))
        plot_tree(final_model.estimators_[0], feature_names=X.columns, class_names=['No', 'Yes'],
                  filled=True, max_depth=grid.best_params_['max_depth'], fontsize=10)
        plt.savefig(os.path.join(output_dir, 'decision_tree_logic.png'), dpi=300)
        plt.show()
        plt.close()




    # --- .PKL KAYDETME ---
    joblib.dump(final_model, os.path.join(output_dir, 'final_model.pkl'))
    joblib.dump(scaler_model, os.path.join(output_dir, 'scaler.pkl'))
    print(f"\n> Model ve Scaler '{output_dir}' klasörüne .pkl olarak kaydedildi.")

    print(f"\n> Analiz tamamlandı! Grafikler '{output_dir}' klasörüne kaydedildi.")

    print(f"\n> Analiz tamamlandı! Grafikler '{output_dir}' klasörüne kaydedildi.")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    file_path = 'shopping_behavior_updated.csv'
    df = load_data(file_path)
    show_eda_details = True

    if df is not None:
        run_analysis_pipeline(df, verbose_eda=show_eda_details)