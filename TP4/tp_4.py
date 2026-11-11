
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, PolynomialFeatures, TargetEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, TimeSeriesSplit, cross_val_score
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 0. LOAD DATA + CREATE THE PROXY TARGET
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
DATA_PATH = "hybrid_student_performance_1200.csv"

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)
print("\nCGPA categories:")
print(df["cgpa_category"].value_counts(dropna=False))

# These are the midpoints of the category intervals.
# This creates an estimated proxy, NOT a real measured final score.
cgpa_mapping = {
    "5.0 – 6.9": (5.0 + 6.9) / 2,
    "7.0 – 8.4": (7.0 + 8.4) / 2,
    "8.5 – 9.4": (8.5 + 9.4) / 2,
    "9.5 – 10.0": (9.5 + 10.0) / 2,
}

df["final_score"] = df["cgpa_category"].map(cgpa_mapping)

if df["final_score"].isna().any():
    unknown = df.loc[df["final_score"].isna(), "cgpa_category"].unique()
    raise ValueError(f"Some cgpa_category values were not mapped: {unknown}")

# Parse timestamp and sort chronologically.
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp", "final_score"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Convert study-hours categories to approximate numeric values.
study_hours_mapping = {
    "Less than 1 hour": 0.5,
    "1–2 hours": 1.5,
    "More than 2 hours": 2.5,
}
df["study_hours_numeric"] = df["study_hours_daily"].map(study_hours_mapping)

if df["study_hours_numeric"].isna().any():
    unknown_hours = df.loc[
        df["study_hours_numeric"].isna(), "study_hours_daily"
    ].unique()
    raise ValueError(f"Unmapped study_hours_daily values: {unknown_hours}")

print("\nDataset shape after target/date preparation:", df.shape)
print("Missing proxy targets:", df["final_score"].isna().sum())


# ============================================================
# 1. COMMON FUNCTIONS
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def chronological_split(data, test_size=TEST_SIZE):
    """First 80% for training, final 20% for testing (already time-sorted)."""
    split_index = int(len(data) * (1 - test_size))
    train = data.iloc[:split_index].copy()
    test = data.iloc[split_index:].copy()
    return train, test


def get_feature_columns(data):
    """Remove identifiers, timestamps and target-derived source column."""
    excluded = {
        "student_id",
        "timestamp",
        "cgpa_category",  # target is derived from this: must not be a feature
        "final_score",
        "study_hours_daily",  # replaced by numeric version below
    }
    features = data.drop(
        columns=[c for c in excluded if c in data.columns]
    ).copy()
    return features


def make_preprocessor(X, scale_numeric=False):
    """Preprocessing is learned inside each training fold/pipeline."""
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])


def print_metrics(name, y_train, train_pred, y_test, test_pred):
    train_rmse = rmse(y_train, train_pred)
    test_rmse = rmse(y_test, test_pred)
    print(f"\n{name}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE : {test_rmse:.4f}")
    print(f"Train R²  : {r2_score(y_train, train_pred):.4f}")
    print(f"Test R²   : {r2_score(y_test, test_pred):.4f}")
    print(f"Generalization gap (test - train RMSE): "
          f"{test_rmse - train_rmse:.4f}")


# ============================================================
# PART A — DATA INSPECTION
# ============================================================

print("\n" + "=" * 60)
print("PART A — DATA INSPECTION")
print("=" * 60)

print("\nA1. Shape:")
print(df.shape)

print("\nA2. Data types:")
print(df.dtypes)

print("\nA3. Descriptive statistics:")
print(df.describe(include="all").T)

print("\nA4. Missing values:")
missing = df.isna().sum()
print(missing[missing > 0] if (missing > 0).any() else "No missing values.")

# Duplicate check with and without student_id.
dups_with_id = df.duplicated().sum()

cols_without_id = [c for c in df.columns if c != "student_id"]
dups_without_id = df[cols_without_id].duplicated().sum()

print("\nA5. Duplicate rows:")
print("Duplicates including student_id:", dups_with_id)
print("Duplicates excluding student_id:", dups_without_id)

# Numeric correlations with the proxy target.
numeric_df = df.select_dtypes(include=np.number)
correlations = numeric_df.corr(numeric_only=True)["final_score"].sort_values(
    ascending=False
)
print("\nA6. Numeric correlations with estimated final_score:")
print(correlations)

# The assignment asks for school_code. This dataset does not have it.
# program_stream is inspected only as a categorical analogue.
if "school_code" in df.columns:
    category_column = "school_code"
    print("\nA7. Rare school_code categories (<10):")
else:
    category_column = "program_stream"
    print(
        "\nA7. No school_code column exists. "
        "Using program_stream only as a categorical analogue."
    )

category_counts = df[category_column].value_counts(dropna=False)
rare_categories = category_counts[category_counts < 10]
print(rare_categories if len(rare_categories) else "No categories have fewer than 10 rows.")

# Plot target over time.
plt.figure(figsize=(12, 5))
plt.plot(df["timestamp"], df["final_score"], marker=".", linestyle="none", alpha=0.5)
plt.title("Estimated final_score over time")
plt.xlabel("Timestamp")
plt.ylabel("Estimated final_score (CGPA-category midpoint)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("A_target_over_time.png", dpi=150)
plt.show()


# ============================================================
# PREPARE FEATURES + CHRONOLOGICAL TRAIN/TEST SPLIT
# ============================================================

# study_hours_numeric is retained as an additional numeric feature.
X = get_feature_columns(df)
y = df["final_score"].copy()

# The chronological split is reused for the model comparisons.
X_train, X_test, y_train, y_test = None, None, None, None
split_index = int(len(df) * (1 - TEST_SIZE))

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()
y_train = y.iloc[:split_index].copy()
y_test = y.iloc[split_index:].copy()

print("\nChronological split sizes:")
print("Train:", X_train.shape, " Test:", X_test.shape)


# ============================================================
# PART B — DATA LEAKAGE EXPERIMENTS
# ============================================================

print("\n" + "=" * 60)
print("PART B — DATA LEAKAGE")
print("=" * 60)

# B1. Random split versus chronological split.
# For a fair comparison, both use the same feature columns and pipeline.
X_random_train, X_random_test, y_random_train, y_random_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

random_tree = Pipeline([
    ("preprocess", make_preprocessor(X_train)),
    ("model", DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)),
])
random_tree.fit(X_random_train, y_random_train)
random_train_pred = random_tree.predict(X_random_train)
random_test_pred = random_tree.predict(X_random_test)

print_metrics(
    "B1 — Random split, Decision Tree",
    y_random_train, random_train_pred,
    y_random_test, random_test_pred,
)

chrono_tree = Pipeline([
    ("preprocess", make_preprocessor(X_train)),
    ("model", DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)),
])
chrono_tree.fit(X_train, y_train)
chrono_train_pred = chrono_tree.predict(X_train)
chrono_test_pred = chrono_tree.predict(X_test)

print_metrics(
    "B1 — Chronological split, Decision Tree",
    y_train, chrono_train_pred,
    y_test, chrono_test_pred,
)

# B2. Duplicate leakage demonstration.
# Split rows first, allowing identical records to land in both sets.
dedup_check_cols = [c for c in df.columns if c != "student_id"]
duplicate_flags = df.duplicated(subset=dedup_check_cols, keep=False)
duplicate_rows = df.loc[duplicate_flags].copy()

print("\nB2. Duplicate leakage:")
print("Rows belonging to duplicate groups (excluding student_id):",
      len(duplicate_rows))

if len(duplicate_rows) > 0:
    dup_X = get_feature_columns(df)
    dup_y = df["final_score"]

    dup_X_train, dup_X_test, dup_y_train, dup_y_test = train_test_split(
        dup_X, dup_y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Count identical feature rows that occur in both partitions.
    train_keys = set(pd.util.hash_pandas_object(dup_X_train, index=False))
    test_keys = set(pd.util.hash_pandas_object(dup_X_test, index=False))
    overlap = len(train_keys.intersection(test_keys))
    print("Identical feature-row hashes shared by train and test:", overlap)

    dup_model = Pipeline([
        ("preprocess", make_preprocessor(dup_X_train)),
        ("model", DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE)),
    ])
    dup_model.fit(dup_X_train, dup_y_train)
    pred_before = dup_model.predict(dup_X_test)
    print("Test RMSE before removing duplicates:", rmse(dup_y_test, pred_before))

    # Remove duplicate records without using student_id to identify them.
    dedup_df = df.drop_duplicates(subset=dedup_check_cols).reset_index(drop=True)
    dedup_X = get_feature_columns(dedup_df)
    dedup_y = dedup_df["final_score"]

    dX_train, dX_test, dy_train, dy_test = train_test_split(
        dedup_X, dedup_y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    dedup_model = Pipeline([
        ("preprocess", make_preprocessor(dX_train)),
        ("model", DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE)),
    ])
    dedup_model.fit(dX_train, dy_train)
    pred_after = dedup_model.predict(dX_test)
    print("Test RMSE after removing duplicates:", rmse(dy_test, pred_after))
else:
    print("No duplicate groups found; duplicate leakage cannot be demonstrated.")

# B3. Preprocessing leakage: scaler fit on all data versus train-only pipeline.
# Use numeric features only and Linear Regression so scaling is relevant.
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
X_num = X[numeric_features].copy()
X_num_train = X_num.iloc[:split_index].copy()
X_num_test = X_num.iloc[split_index:].copy()

numeric_imputer = SimpleImputer(strategy="median")
X_num_train_imp = pd.DataFrame(
    numeric_imputer.fit_transform(X_num_train),
    columns=numeric_features,
    index=X_num_train.index,
)
X_num_test_imp = pd.DataFrame(
    numeric_imputer.transform(X_num_test),
    columns=numeric_features,
    index=X_num_test.index,
)

# Incorrect: scaler learns mean/std from both train and test.
leaky_scaler = StandardScaler()
X_all_scaled = leaky_scaler.fit_transform(
    pd.concat([X_num_train_imp, X_num_test_imp])
)
X_leaky_train = X_all_scaled[:len(X_num_train_imp)]
X_leaky_test = X_all_scaled[len(X_num_train_imp):]

leaky_lr = LinearRegression()
leaky_lr.fit(X_leaky_train, y_train)
leaky_pred = leaky_lr.predict(X_leaky_test)
print("\nB3 — Incorrect scaler fit on all data; test RMSE:",
      rmse(y_test, leaky_pred))

# Correct: scaler is fitted only on training data.
correct_lr = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LinearRegression()),
])
correct_lr.fit(X_num_train, y_train)
correct_pred = correct_lr.predict(X_num_test)
print("B3 — Correct train-only scaler Pipeline; test RMSE:",
      rmse(y_test, correct_pred))
print("Note: the difference may be small; leakage is about using test "
      "distribution information, not guaranteeing a better score.")


# ============================================================
# PART C — OVERFITTING AND MODEL COMPLEXITY
# ============================================================

print("\n" + "=" * 60)
print("PART C — OVERFITTING AND GENERALIZATION")
print("=" * 60)

# C1. Decision Tree depth experiment.
depths = [1, 2, 3, 5, 10, 20, None]
depth_results = []

for depth in depths:
    model = Pipeline([
        ("preprocess", make_preprocessor(X_train)),
        ("tree", DecisionTreeRegressor(
            max_depth=depth, random_state=RANDOM_STATE
        )),
    ])
    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

    train_error = rmse(y_train, train_prediction)
    test_error = rmse(y_test, test_prediction)

    depth_results.append({
        "max_depth": str(depth),
        "train_RMSE": train_error,
        "test_RMSE": test_error,
        "generalization_gap": test_error - train_error,
        "test_R2": r2_score(y_test, test_prediction),
    })

depth_results_df = pd.DataFrame(depth_results)
print("\nC1. Decision Tree depth results:")
print(depth_results_df.to_string(index=False))

plt.figure(figsize=(9, 5))
plt.plot(depth_results_df["max_depth"], depth_results_df["train_RMSE"],
         marker="o", label="Train RMSE")
plt.plot(depth_results_df["max_depth"], depth_results_df["test_RMSE"],
         marker="o", label="Test RMSE")
plt.xlabel("Maximum tree depth")
plt.ylabel("RMSE")
plt.title("Decision Tree complexity and overfitting")
plt.legend()
plt.tight_layout()
plt.savefig("C1_tree_depth_rmse.png", dpi=150)
plt.show()

# C2. Polynomial study-hours experiment with TimeSeriesSplit.
hours_X = df[["study_hours_numeric"]].copy()
hours_y = df["final_score"].copy()

tscv = TimeSeriesSplit(n_splits=5)
degrees = [1, 2, 3, 5, 9, 15]
poly_results = []

for degree in degrees:
    poly_model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("linear", LinearRegression()),
    ])

    cv_neg_mse = cross_val_score(
        poly_model,
        hours_X,
        hours_y,
        cv=tscv,
        scoring="neg_mean_squared_error",
    )
    cv_rmse = np.sqrt(-cv_neg_mse)

    poly_results.append({
        "degree": degree,
        "mean_CV_RMSE": cv_rmse.mean(),
        "std_CV_RMSE": cv_rmse.std(),
    })

poly_results_df = pd.DataFrame(poly_results)
print("\nC2. Polynomial degree results (TimeSeriesSplit):")
print(poly_results_df.to_string(index=False))

plt.figure(figsize=(9, 5))
plt.plot(poly_results_df["degree"], poly_results_df["mean_CV_RMSE"],
         marker="o")
plt.xlabel("Polynomial degree")
plt.ylabel("Mean CV RMSE")
plt.title("Polynomial degree vs Time-Series CV RMSE")
plt.tight_layout()
plt.savefig("C2_polynomial_degree_cv.png", dpi=150)
plt.show()


# ============================================================
# PART D — ENCODING, REGULARIZATION AND FINAL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("PART D — ENCODING AND REGULARIZATION")
print("=" * 60)

# D1. Rare-category grouping.
class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, min_count=10, replacement="Other"):
        self.min_count = min_count
        self.replacement = replacement

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        self.frequent_categories_ = {}
        for column in X_df.columns:
            counts = X_df[column].value_counts(dropna=False)
            self.frequent_categories_[column] = set(
                counts[counts >= self.min_count].index
            )
        self.columns_ = X_df.columns.tolist()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.columns_).copy()
        for column in X_df.columns:
            frequent = self.frequent_categories_[column]
            X_df[column] = X_df[column].where(
                X_df[column].isin(frequent), self.replacement
            )
        return X_df


print("\nD1. Categorical column used:", category_column)
print("This is program_stream, not school_code, in the supplied dataset.")

# D2. OOF target encoding.
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

target_encoded_cols = [category_column] if category_column in categorical_cols else []
other_categorical_cols = [
    c for c in categorical_cols if c not in target_encoded_cols
]

transformers = [
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_cols),
]

if target_encoded_cols:
    target_encoding_pipe = Pipeline([
        ("rare_group", RareCategoryGrouper(min_count=10)),
        ("target_encoder", TargetEncoder(
            target_type="continuous",
            smooth="auto",
            cv=5,
            shuffle=True,
            random_state=RANDOM_STATE,
        )),
    ])
    transformers.append(
        ("target_encoded", target_encoding_pipe, target_encoded_cols)
    )

if other_categorical_cols:
    other_cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers.append(
        ("other_categories", other_cat_pipe, other_categorical_cols)
    )

target_encoder_preprocessor = ColumnTransformer(transformers)

target_encoded_model = Pipeline([
    ("preprocess", target_encoder_preprocessor),
    ("model", LinearRegression()),
])

target_encoded_model.fit(X_train, y_train)
te_prediction = target_encoded_model.predict(X_test)
print("\nD2. OOF TargetEncoder + Linear Regression")
print("Test RMSE:", rmse(y_test, te_prediction))
print("Test R²:", r2_score(y_test, te_prediction))

# D3. Polynomial degree 5 + Linear Regression / Ridge / Lasso.
# This part uses the numeric study-hours proxy, because the source column
# study_hours_daily is categorical text rather than a continuous variable.
poly_X = df[["study_hours_numeric"]].copy()
poly_y = df["final_score"].copy()

alphas = [0.001, 0.01, 0.1, 1, 10, 100]
regularization_results = []

models_to_test = [("LinearRegression", LinearRegression())]
models_to_test += [(f"Ridge alpha={a}", Ridge(alpha=a)) for a in alphas]
models_to_test += [(f"Lasso alpha={a}", Lasso(alpha=a, max_iter=20000))
                   for a in alphas]

for model_name, estimator in models_to_test:
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=5, include_bias=False)),
        ("scaler", StandardScaler()),
        ("regressor", estimator),
    ])

    scores = cross_val_score(
        model,
        poly_X,
        poly_y,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_mean_squared_error",
    )
    fold_rmse = np.sqrt(-scores)

    regularization_results.append({
        "model": model_name,
        "mean_CV_RMSE": fold_rmse.mean(),
        "std_CV_RMSE": fold_rmse.std(),
    })

regularization_df = pd.DataFrame(regularization_results).sort_values(
    "mean_CV_RMSE"
)
print("\nD3. Degree-5 polynomial regression comparison:")
print(regularization_df.to_string(index=False))

# D4. Final model comparison.

final_models = {
    "Linear Regression": Pipeline([
        ("preprocess", make_preprocessor(X_train, scale_numeric=True)),
        ("model", LinearRegression()),
    ]),
    "Ridge": Pipeline([
        ("preprocess", make_preprocessor(X_train, scale_numeric=True)),
        ("model", Ridge(alpha=1.0)),
    ]),
    "Lasso": Pipeline([
        ("preprocess", make_preprocessor(X_train, scale_numeric=True)),
        ("model", Lasso(alpha=0.1, max_iter=20000)),
    ]),
    "Decision Tree": Pipeline([
        ("preprocess", make_preprocessor(X_train)),
        ("model", DecisionTreeRegressor(
            max_depth=5, random_state=RANDOM_STATE
        )),
    ]),
    "Gradient Boosting": Pipeline([
        ("preprocess", make_preprocessor(X_train)),
        ("model", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=2,
            n_iter_no_change=5,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        )),
    ]),
}

final_results = []

for name, model in final_models.items():
    model.fit(X_train, y_train)
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

    final_results.append({
        "model": name,
        "train_RMSE": rmse(y_train, train_prediction),
        "test_RMSE": rmse(y_test, test_prediction),
        "test_R2": r2_score(y_test, test_prediction),
        "generalization_gap": (
            rmse(y_test, test_prediction) -
            rmse(y_train, train_prediction)
        ),
    })

final_results_df = pd.DataFrame(final_results).sort_values("test_RMSE")
print("\nD4. Final model comparison:")
print(final_results_df.to_string(index=False))

best_model_name = final_results_df.iloc[0]["model"]
print("\nLowest test RMSE in this run:", best_model_name)

print("\nAll plots were saved in the current working directory.")
print("Script execution completed.")
