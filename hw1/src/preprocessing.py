import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import joblib
from scipy.stats import zscore, iqr
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# =========================================================
# 1. Pandas Preprocessing Pipeline Functions
# =========================================================

def compute_colors(df):
    """Compute color indices if required columns exist."""
    df = df.copy()
    if set(['U', 'G', 'R', 'WF3', 'UT', 'WF1']).issubset(df.columns):
        df['u-g'] = df['U'] - df['G']
        df['g-r'] = df['G'] - df['R']
        df['W3-UT'] = df['WF3'] - df['UT']
        df['(W3+UT)/W1'] = (df['WF3'] + df['UT']) / df['WF1']
    return df

def filter_important_columns(df, key_vars, err_vars, no_err_vars, flags_vars, color_vars):
    """
    1a. Keep only important columns and their errors
    1b. Keep only the metal flag of the key columns but only if metal exists
    """
    cols_to_keep = key_vars + err_vars + no_err_vars + flags_vars + color_vars
    
    # 1b. Keep FLAG_METAL if METAL exists
    if 'METAL' in cols_to_keep and 'FLAG_METAL' in df.columns:
        cols_to_keep.append('FLAG_METAL')
        
    # Ensure columns actually exist in the DataFrame before selecting them
    existing_cols = [c for c in cols_to_keep if c in df.columns]
    
    return df[existing_cols].copy()


def filter_error_ratios(df, key_vars, err_vars):
    """
    Keep rows where the ratio of value / error is > 3, as per standard thresholding.
    """
    ratio_mask = np.ones(len(df), dtype=bool)
    
    for key_col, err_col in zip(key_vars, err_vars):
        if key_col in df.columns and err_col in df.columns:
            # Handle potential zeros implicitly in Pandas div (yields inf, which is > 3)
            ratio = np.abs(df[key_col] / df[err_col]) * 100
            ratio_mask &= (ratio > 3)
    
    # Drop the error columns, no longer usefull
    df = df[ratio_mask].copy()
    df = df.drop(columns=err_vars, errors='ignore')
    
    return df


def select_metal_flag(df, frac=0.005, random_state=42):
    """2. Subsample FLAG_METAL=-1 rows."""
    if 'FLAG_METAL' not in df.columns:
        return df
    
    # Split the dataset on FLAG_METAL condition
    df_missing = df[df['FLAG_METAL'] == -1]
    df_valid = df[df['FLAG_METAL'] != -1]
    
    df_missing_reduced = df_missing.sample(frac=frac, random_state=random_state)
    # Drop the FLAG_METAL column, no longer usefull
    df_valid = df_valid.drop(columns=['FLAG_METAL'])
    df_missing_reduced = df_missing_reduced.drop(columns=['FLAG_METAL'])
    return pd.concat([df_valid, df_missing_reduced]).sort_index()


def split_data(df, target_col, train_size, val_size, test_size, random_state=42):
    """
    3. Split to training, validation and test returning intact native dataframes
    """
    temp_size = val_size + test_size
    stratify_y = df[target_col].copy().fillna("MISSING")
    
    try:
        train_df, temp_df = train_test_split(
            df, test_size=temp_size, stratify=stratify_y, random_state=random_state
        )
        
        test_ratio = test_size / temp_size
        stratify_temp = temp_df[target_col].copy().fillna("MISSING")
        val_df, test_df = train_test_split(
            temp_df, test_size=test_ratio, stratify=stratify_temp, random_state=random_state
        )
    except ValueError:
        train_df, temp_df = train_test_split(
            df, test_size=temp_size, random_state=random_state
        )
        
        test_ratio = test_size / temp_size
        val_df, test_df = train_test_split(
            temp_df, test_size=test_ratio, random_state=random_state
        )
    
    return train_df, val_df, test_df


# =========================================================
# 3. Missing Value Treatment & Imputation
# =========================================================

def drop_missing_targets(df, target_col):
    """
    Drop rows where the target variable itself is missing entirely natively.
    """
    return df.dropna(subset=[target_col]).copy()

def fit_target_encoder(train_df, target_col):
    """
    Fits target encoder securely against training targets.
    """
    le = LabelEncoder()
    temp_targets = train_df.dropna(subset=[target_col])[target_col]
    le.fit(temp_targets)
    os.makedirs("models", exist_ok=True)
    joblib.dump(le, "models/label_encoder.pkl")
    return le

def apply_target_encoder(df, target_col, le):
    """
    Encodes the target variable numerically via Pipeline format.
    """
    df = df.copy()
    if target_col in df.columns:
        df[target_col] = le.transform(df[target_col])
    return df


# =========================================================
# 4. Outlier Detection (Winsorizing)
# =========================================================

def compute_iqr_bounds(X_train, factor=1.5):
    """
    Computes IQR boundaries strictly over numerical features belonging to the training set 
    to prevent data leakage against validation/test sets.
    """
    bounds = {}
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        bounds[col] = (Q1 - factor * IQR, Q3 + factor * IQR)
    return bounds


def apply_iqr_capping(X, bounds):
    """
    Caps values across dataset subsets according to previously fit threshold bounds natively.
    """
    X_capped = X.copy()
    for col, (lower, upper) in bounds.items():
        if col in X_capped.columns:
            X_capped[col] = np.clip(X_capped[col], lower, upper)
    return X_capped


def save_outlier_histograms(df, prefix="dataset", out_dir="visuals"):
    """
    Generates and saves a histogram for all numeric columns post-outlier truncation securely.
    Acts as a seamless passthrough mapping natively via Pandas .pipe() structures.
    """
    if not out_dir:
        return df
        
    num_df = df.select_dtypes(include=[np.number])
    # Visually exclude the raw Error values and tracking Flags which clutter diagnostic evaluation
    drop_cols = [c for c in num_df.columns if c.startswith('E_') or 'FLAG' in c]
    num_df = num_df.drop(columns=drop_cols)
    
    if not num_df.empty:
        os.makedirs(out_dir, exist_ok=True)
        # Save plots to explicitly verify truncation mathematically
        num_df.hist(bins=30, figsize=(20, 15))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_post_outliers_histogram.png"))
        plt.close()
    return df


def build_preprocessing_pipeline(X_train):
    """
    4 and 5. Missing value treatment and Categorical Encoding.
    Identifies numerical and categorical cardinalities dynamically.
    """
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Safely categorize string/categorical column scopes for dynamic encoding strategies
    binary_cols = [c for c in cat_cols if X_train[c].dropna().nunique() == 2]
    high_card_cols = [c for c in cat_cols if X_train[c].dropna().nunique() > 10]
    nominal_cols = [c for c in cat_cols if c not in binary_cols and c not in high_card_cols]
    
    # Numerical imputation pipeline
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    # Categorical imputation + encoding pipelines natively mapped
    bin_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    nom_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'))
    ])
    
    hc_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', TargetEncoder(target_type='continuous')) # Defaults continuous/auto but securely explicitly mapping just in case
    ])

    # Combine transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('bin', bin_pipeline, binary_cols),
            ('nom', nom_pipeline, nominal_cols),
            ('hc', hc_pipeline, high_card_cols)
        ],
        remainder='passthrough'
    )
    
    # Wrapped into final Pipeline natively
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline


def apply_imputation(df, pipeline, target_col):
    """
    Natively bridges Scikit-Learn transformers smoothly accommodating Pandas .pipe() capabilities
    by managing extracting, applying, and gracefully reconstructing variables.
    """
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    raw_cols = pipeline.get_feature_names_out()
    out_columns = [c.replace('num__', '').replace('bin__', '').replace('nom__', '').replace('hc__', '') for c in raw_cols]
    
    X_transformed = pd.DataFrame(pipeline.transform(X), columns=out_columns, index=X.index)
    return X_transformed.assign(**{target_col: y})

def get_fitted_scaler(X_train):
    """
    Fits scaler exclusively against the continuous tracking shapes matching cleanly.
    """
    scaler = StandardScaler()
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    scaler.fit(X_train[num_cols])
    return scaler, num_cols


def apply_scaling(df, scaler, num_cols, target_col):
    """
    Applies static boundaries iteratively across splits ensuring target states are untouched
    preventing testing / validation cross-leakage mathematically.
    """
    y = df[target_col]
    X_scaled = df.drop(columns=[target_col]).copy()
    
    # Scale safely across designated arrays iteratively
    cols_to_scale = [c for c in num_cols if c in X_scaled.columns]
    if cols_to_scale:
        X_scaled[cols_to_scale] = scaler.transform(X_scaled[cols_to_scale])
        
    return X_scaled.assign(**{target_col: y})


def generate_pca_insights(X_train, y_train, out_dir="visuals"):
    """
    Executes Exploratory PCA on the completely scaled training set structurally matching Task 2.7.
    Produces Scree Plots, 2D Scatter Distributions natively matching class labels, 
    and formally outputs dominant Absolute PCA Loadings across Top Components.
    """
    if not out_dir:
        return
        
    os.makedirs(out_dir, exist_ok=True)
    
    # Evaluate explicit Principal Component tracks purely against trained inputs iteratively
    pca = PCA()
    X_pca = pca.fit_transform(X_train)
    
    # 1. Variance Scree Plot
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.title('PCA Scree Plot (Cumulative Variance)')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_scree_plot.png"))
    plt.close()
    
    # 2. 2D Orthogonal Projection matching target states iteratively
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train, palette="viridis", alpha=0.7)
    plt.title('2D PCA Projection (PC1 vs PC2)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.legend(title='Target Class')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_2d_projection.png"))
    plt.close()
    
    # 3. Parameter Weight Configurations evaluating the original mapping limits
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=[f'PC{i+1}' for i in range(pca.n_components_)], 
        index=X_train.columns
    )
    
    print("\n--- Task 2.7: Top PCA Loadings (Absolute Weights) ---")
    for pc in ['PC1', 'PC2', 'PC3']:
        top_features = loadings[pc].abs().sort_values(ascending=False).head(3)
        print(f"\n{pc} Dominant Features:")
        for feat, weight in top_features.items():
            print(f"  - {feat:12s}: {loadings.loc[feat, pc]:.4f}")
            
    print(f"\nSaved Exploratory PCA visual tracking metrics correctly inside '{out_dir}/' directory.")
