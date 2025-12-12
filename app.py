from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "model.joblib"
DATA_PATH = "data.csv"
TARGET = "kredit"

# Более подробные описания (tooltip). Переносы строк поддерживаются (см. CSS).
FEATURE_META = {
    "laufkont": (
        "Статус текущего счёта",
        "Код статуса текущего счёта (checking account).\n"
        "Типы (коды):\n"
        "1 — нет счёта / отрицательный баланс\n"
        "2 — 0..200 (условная шкала)\n"
        "3 — >= 200 (условная шкала)\n"
        "4 — счёта нет / неизвестно\n"
        "Примечание: коды соответствуют кодировке в датасете."
    ),
    "laufzeit": ("Срок кредита, мес.", "Количество месяцев кредита (duration)."),
    "moral": (
        "Кредитная история",
        "Код категории кредитной истории.\n"
        "Отражает качество прошлых обязательств (погашения/просрочки) в виде кода."
    ),
    "verw": (
        "Цель кредита",
        "Код назначения кредита (purpose).\n"
        "Например: техника, мебель, ремонт, авто и т.д. (в виде кода)."
    ),
    "hoehe": ("Сумма кредита", "Сумма кредита (credit amount)."),
    "sparkont": (
        "Сбережения",
        "Код уровня сбережений.\n"
        "Чем выше код — тем выше/лучше категория сбережений (в рамках кодировки датасета)."
    ),
    "beszeit": (
        "Стаж занятости",
        "Код категории стажа/занятости.\n"
        "Обычно отражает интервалы: <1 года, 1–4, 4–7, >7 и т.п. (в виде кода)."
    ),
    "rate": (
        "Доля платежа",
        "Код категории доли ежемесячного платежа относительно дохода.\n"
        "Чем выше код — тем выше нагрузка (в рамках кодировки датасета)."
    ),
    "famges": ("Семейный статус", "Код семейного положения / статуса."),
    "buerge": ("Поручительство", "Код наличия поручителя/гаранта."),
    "wohnzeit": ("Срок проживания", "Код времени проживания по текущему адресу."),
    "verm": ("Имущество", "Код категории имущества (property)."),
    "alter": ("Возраст", "Возраст заёмщика в годах."),
    "weitkred": ("Другие кредиты", "Код наличия других кредитов/обязательств."),
    "wohn": ("Тип жилья", "Код типа жилья (аренда/собственное/служебное и т.п.)."),
    "bishkred": ("Кредиты ранее", "Код/категория количества ранее полученных кредитов."),
    "beruf": ("Профессия", "Код категории профессии/квалификации."),
    "pers": ("Личный статус", "Код личного/социального статуса."),
    "telef": ("Телефон", "Код: наличие телефона / тип."),
    "gastarb": ("Иностранный работник", "Код: является ли заёмщик иностранным работником.")
}

# --- загрузка модели ---
bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
feature_names = bundle["feature_names"]
metrics = bundle.get("best_metrics", {})

# --- важности признаков (RandomForest/Tree) ---
model = pipeline.named_steps["model"]
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    importance_map = {f: float(w) for f, w in zip(feature_names, importances)}
else:
    importance_map = {f: 0.0 for f in feature_names}

TOP_N = 6
top_features = sorted(importance_map, key=lambda k: importance_map[k], reverse=True)[:TOP_N]
top_set = set(top_features)

def build_schema():
    """
    Строим схему для фронта:
    - русское название
    - подробное описание (tooltip)
    - тип, диапазон (min/max) из реального data.csv
    - importance + флаг top
    """
    df = pd.read_csv(DATA_PATH)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    schema = []
    for f in feature_names:
        series = df[f]
        mn = float(series.min())
        mx = float(series.max())

        label, desc = FEATURE_META.get(f, (f, "Числовой признак."))

        schema.append({
            "name": f,
            "label": label,
            "desc": desc,
            "min": int(mn) if mn.is_integer() else mn,
            "max": int(mx) if mx.is_integer() else mx,
            "step": 1,
            "type": "int",
            "placeholder": f"Допустимо: {int(mn)}–{int(mx)}" if mn.is_integer() and mx.is_integer() else f"Допустимо: {mn:.2f}–{mx:.2f}",
            "importance": round(float(importance_map.get(f, 0.0)), 6),
            "is_top": f in top_set
        })
    return schema

schema = build_schema()

# --- маршруты страниц ---
@app.get("/")
def index():
    return render_template(
        "index.html",
        schema=schema,
        metrics=metrics
    )

@app.get("/model")
def model_page():
    # таблица топ-признаков
    top_table = [{"feature": f, "importance": round(importance_map[f], 6)} for f in top_features]

    # картинки, которые у вас уже есть после train.py
    # Если их нет — страница всё равно откроется, просто изображения не загрузятся.
    images = {
        "roc_dt": "roc_DecisionTree.png",
        "roc_rf": "roc_RandomForest.png",
        "cm_dt": "cm_DecisionTree.png",
        "cm_rf": "cm_RandomForest.png",
    }

    return render_template(
        "model.html",
        metrics=metrics,
        top_table=top_table,
        images=images
    )

# --- отдача картинок из корня проекта ---
# (чтобы не переносить png в static/)
@app.get("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(os.getcwd(), filename)

# --- API прогнозирования ---
@app.post("/predict")
def predict():
    payload = request.get_json(force=True)
    threshold = float(payload.get("threshold", 0.5))

    row = {}
    for it in schema:
        f = it["name"]
        if f not in payload:
            return jsonify({"error": f"Missing feature: {f}"}), 400
        row[f] = payload[f]

    X = pd.DataFrame([row], columns=[it["name"] for it in schema])

    proba_bad = float(pipeline.predict_proba(X)[:, 1][0])
    decision = "REJECT" if proba_bad >= threshold else "APPROVE"

    return jsonify({
        "proba_bad": round(proba_bad, 6),
        "threshold": threshold,
        "decision": decision
    })

if __name__ == "__main__":
    app.run(debug=True)
