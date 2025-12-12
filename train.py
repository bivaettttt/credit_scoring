import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)

# =====================
# НАСТРОЙКИ
# =====================
DATA_PATH = "data.csv"

# ВАЖНО: в вашей версии датасета target = 'kredit'
TARGET = "kredit"

RANDOM_STATE = 42


def to_binary_target(y_raw: pd.Series):
    """
    Универсально приводит бинарный target к 0/1 и возвращает mapping.
    Подходит для:
    - 0/1
    - 1/2
    - строки (2 уникальных значения)
    """
    y_raw = y_raw.copy()
    uniq = list(pd.Series(y_raw.dropna().unique()))

    if len(uniq) != 2:
        raise ValueError(f"Target не бинарный. Уникальные значения: {uniq}")

    # Чтобы было стабильно, сортируем, если возможно
    try:
        uniq_sorted = sorted(uniq)
    except TypeError:
        # если типы несравнимы (редко), оставим как есть
        uniq_sorted = uniq

    mapping = {uniq_sorted[0]: 0, uniq_sorted[1]: 1}
    y = y_raw.map(mapping)
    return y, mapping


def main():
    df = pd.read_csv(DATA_PATH)

    # Убираем индекс, если есть
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    if TARGET not in df.columns:
        raise ValueError(
            f"В файле нет целевой колонки '{TARGET}'. "
            f"Доступные колонки: {df.columns.tolist()}"
        )

    # Target / Features
    y_raw = df[TARGET]
    X = df.drop(columns=[TARGET])

    # Бинаризация target
    y, mapping = to_binary_target(y_raw)
    print("\nTarget mapping:", mapping)
    print("Target distribution:\n", y.value_counts(dropna=False))

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Типы признаков
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    print("\nNumeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # Предобработка
    num_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_tf, numeric_features),
            ("cat", cat_tf, categorical_features),
        ],
        remainder="drop"
    )

    # Модели
    models = {
        "DecisionTree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=6,
            min_samples_leaf=10
        ),
        "RandomForest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=300,
            min_samples_leaf=5,
            n_jobs=-1
        )
    }

    results = []

    for name, clf in models.items():
        pipe = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", clf),
        ])

        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        roc = roc_auc_score(y_test, proba)
        f1 = f1_score(y_test, pred)
        rec = recall_score(y_test, pred)
        prec = precision_score(y_test, pred)

        results.append((name, roc, f1, rec, prec, pipe))

        print(f"\n{name}")
        print(f"ROC-AUC:   {roc:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"Precision: {prec:.4f}")

        # Confusion matrix plot
        cm = confusion_matrix(y_test, pred)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        plt.title(f"Confusion Matrix – {name}")
        plt.tight_layout()
        plt.savefig(f"cm_{name}.png", dpi=200)
        plt.close()

        # ROC curve plot
        RocCurveDisplay.from_predictions(y_test, proba)
        plt.title(f"ROC Curve – {name}")
        plt.tight_layout()
        plt.savefig(f"roc_{name}.png", dpi=200)
        plt.close()

    # Выбор лучшей модели по ROC-AUC
    results.sort(key=lambda x: x[1], reverse=True)
    best_name, best_roc, best_f1, best_rec, best_prec, best_pipe = results[0]

    print("\nBEST MODEL:", best_name)
    print("BEST ROC-AUC:", round(best_roc, 4))

    # Сохранение
    joblib.dump(
        {
            "pipeline": best_pipe,
            "target_col": TARGET,
            "target_mapping": mapping,
            "feature_names": X.columns.tolist(),
            "best_model": best_name,
            "best_metrics": {
                "roc_auc": float(best_roc),
                "f1": float(best_f1),
                "recall": float(best_rec),
                "precision": float(best_prec),
            }
        },
        "model.joblib"
    )

    print("Saved model.joblib")


if __name__ == "__main__":
    main()