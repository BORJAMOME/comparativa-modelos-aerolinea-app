"""
Replica la metodologia del notebook original (comparativa de 3 modelos de
clasificacion multiclase) y guarda los artefactos que consume la app de
Streamlit: deteccion de outliers, correlacion, VIF, chequeo de leakage,
resultados de validacion cruzada y test, matrices de confusion, feature
importance, coeficientes de regresion logistica, y los 3 pipelines
entrenados (para poder predecir en vivo en el Playground).

Ejecutar una sola vez:
    py -3.10 model/train.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              f1_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "dataset_linea_aerea_multiclase_v2.xlsx"
ARTIFACTS = ROOT / "model" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

TARGET = "segmento_cliente"
RANDOM_STATE = 42
MODELO_BOOSTING = "Gradient Boosting"
MODEL_SLUGS = {"Regresión Logística": "lr", "Random Forest": "rf", "Gradient Boosting": "gb"}


def main():
    print("Cargando dataset...")
    df = pd.read_excel(DATA_PATH)
    n_filas, n_columnas_original = df.shape

    variables_numericas = df.select_dtypes(include=np.number).columns.tolist()
    variables_categoricas = [c for c in df.select_dtypes(include=["object", "category"]).columns
                              if c != TARGET]

    # -- Calidad del dato ---------------------------------------------------------
    nulos = pd.DataFrame({
        "variable": df.columns,
        "nulos": df.isnull().sum().values,
        "pct_nulos": (df.isnull().mean() * 100).round(2).values,
    })
    nulos = nulos[nulos["nulos"] > 0].sort_values("pct_nulos", ascending=False)
    nulos.to_csv(ARTIFACTS / "nulos.csv", index=False)

    class_dist = (df[TARGET].value_counts(normalize=True) * 100).round(1)

    # -- Deteccion de outliers: 4 metodos, comparativa -----------------------------
    print("Comparando metodos de deteccion de outliers...")
    resultado_iqr, resultado_z, resultado_pct = {}, {}, {}
    for v in variables_numericas:
        Q1, Q3 = df[v].quantile(0.25), df[v].quantile(0.75)
        IQR = Q3 - Q1
        resultado_iqr[v] = int(((df[v] < Q1 - 1.5 * IQR) | (df[v] > Q3 + 1.5 * IQR)).sum())

        z = np.abs(zscore(df[v].dropna()))
        resultado_z[v] = int(np.sum(z > 3))

        p1, p99 = df[v].quantile(0.01), df[v].quantile(0.99)
        resultado_pct[v] = int(((df[v] < p1) | (df[v] > p99)).sum())

    X_if = SimpleImputer(strategy="median").fit_transform(df[variables_numericas])
    X_if = StandardScaler().fit_transform(X_if)
    iforest = IsolationForest(contamination=0.03, random_state=RANDOM_STATE)
    pred_if = iforest.fit_predict(X_if)
    df["outlier_iforest"] = pred_if
    n_outliers_if = int((df["outlier_iforest"] == -1).sum())

    outlier_comparativa = pd.DataFrame({
        "variable": variables_numericas,
        "IQR": [resultado_iqr[v] for v in variables_numericas],
        "Z-score": [resultado_z[v] for v in variables_numericas],
        "Percentiles P1/P99": [resultado_pct[v] for v in variables_numericas],
    })
    outlier_comparativa["Isolation Forest (global)"] = n_outliers_if
    outlier_comparativa.to_csv(ARTIFACTS / "outlier_comparativa.csv", index=False)

    # -- Capado (winsorizing P1/P99) — nunca se sobrescribe el df original ---------
    df_sin_outliers = df.copy()
    for v in variables_numericas:
        p1, p99 = df[v].quantile(0.01), df[v].quantile(0.99)
        df_sin_outliers[v] = df[v].clip(lower=p1, upper=p99)

    # -- Correlacion + VIF (multicolinealidad) --------------------------------------
    correlacion = df[variables_numericas].corr(method="pearson").round(4)
    correlacion.to_csv(ARTIFACTS / "correlacion.csv")

    # add_constant es necesario para que el VIF sea el estándar (centrado en la
    # media): sin intercepto, variance_inflation_factor devuelve un VIF "crudo"
    # varias veces más alto, que sobrestima la multicolinealidad real.
    X_vif = df[variables_numericas].fillna(df[variables_numericas].median())
    X_vif_c = add_constant(X_vif)
    vif = pd.DataFrame({
        "variable": X_vif.columns,
        "VIF": [variance_inflation_factor(X_vif_c.values, i + 1) for i in range(X_vif.shape[1])],
    }).round(3)
    vif.to_csv(ARTIFACTS / "vif.csv", index=False)

    # -- Chequeo de leakage: ANOVA F-test / eta^2 -----------------------------------
    df_leak = df[variables_numericas].fillna(df[variables_numericas].median())
    f_stat, p_val = f_classif(df_leak, df[TARGET])
    n, k = len(df_leak), df[TARGET].nunique()
    eta2 = (f_stat * (k - 1)) / (f_stat * (k - 1) + (n - k))
    leakage = pd.DataFrame({
        "variable": variables_numericas, "F_stat": f_stat, "p_valor": p_val, "eta2": eta2,
    }).sort_values("eta2", ascending=False).round(5)
    leakage.to_csv(ARTIFACTS / "leakage.csv", index=False)

    # ==================================================================== MODELADO ==
    print("Preparando datos de modelado...")
    datos_modelo = df_sin_outliers.copy()
    X = datos_modelo.drop(columns=[TARGET, "outlier_iforest"], errors="ignore")
    y = datos_modelo[TARGET]

    variables_numericas_modelo = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    variables_categoricas_modelo = X.select_dtypes(include=["object", "category"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    preprocesador = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), variables_numericas_modelo),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), variables_categoricas_modelo),
    ])

    baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)
    baseline_stats = {
        "accuracy": float(accuracy_score(y_test, y_pred_baseline)),
        "f1_macro": float(f1_score(y_test, y_pred_baseline, average="macro")),
    }

    modelos = {
        "Regresión Logística": Pipeline([
            ("preprocesador", preprocesador),
            ("modelo", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocesador", preprocesador),
            ("modelo", RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_leaf=2,
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
        MODELO_BOOSTING: Pipeline([
            ("preprocesador", preprocesador),
            ("modelo", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }

    print("Validación cruzada (5 folds estratificados)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    resultados_cv = []
    for nombre, pipe in modelos.items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv,
                                 scoring=["accuracy", "f1_macro"], n_jobs=-1)
        resultados_cv.append({
            "modelo": nombre,
            "accuracy_media": float(scores["test_accuracy"].mean()),
            "accuracy_std": float(scores["test_accuracy"].std()),
            "f1_macro_media": float(scores["test_f1_macro"].mean()),
            "f1_macro_std": float(scores["test_f1_macro"].std()),
        })
        print(f"  {nombre}: F1-macro={resultados_cv[-1]['f1_macro_media']:.4f}")
    tabla_cv = pd.DataFrame(resultados_cv).sort_values("f1_macro_media", ascending=False)
    tabla_cv.to_csv(ARTIFACTS / "cv_results.csv", index=False)

    print("Entrenamiento final y evaluación en test...")
    resultados_test = []
    matrices_confusion = {}
    reportes_clasificacion = {}
    clases = sorted(y.unique())

    for nombre, pipe in modelos.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        resultados_test.append({
            "modelo": nombre,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        })

        cm = confusion_matrix(y_test, y_pred, labels=clases)
        matrices_confusion[nombre] = {"labels": clases, "matrix": cm.tolist()}

        report = classification_report(y_test, y_pred, output_dict=True)
        reportes_clasificacion[nombre] = report

        joblib.dump(pipe, ARTIFACTS / f"model_{MODEL_SLUGS[nombre]}.joblib")

    tabla_test = pd.DataFrame(resultados_test).sort_values("f1_macro", ascending=False)
    tabla_test.to_csv(ARTIFACTS / "test_results.csv", index=False)

    with open(ARTIFACTS / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(matrices_confusion, f, ensure_ascii=False, indent=2)
    with open(ARTIFACTS / "classification_reports.json", "w", encoding="utf-8") as f:
        json.dump(reportes_clasificacion, f, ensure_ascii=False, indent=2)

    # -- Interpretabilidad: coeficientes/odds ratios (LR) y feature importance -----
    print("Calculando interpretabilidad...")
    modelo_lr = modelos["Regresión Logística"].named_steps["modelo"]
    nombres_features = modelos["Regresión Logística"].named_steps["preprocesador"].get_feature_names_out()

    odds_ratios = pd.DataFrame(
        np.exp(modelo_lr.coef_), columns=nombres_features, index=modelo_lr.classes_,
    ).T.round(4)
    odds_ratios.to_csv(ARTIFACTS / "odds_ratios_lr.csv")

    for nombre in ["Random Forest", MODELO_BOOSTING]:
        pipe = modelos[nombre]
        importancias = pd.Series(
            pipe.named_steps["modelo"].feature_importances_,
            index=pipe.named_steps["preprocesador"].get_feature_names_out(),
        ).sort_values(ascending=False)
        fname = "feature_importance_rf.csv" if nombre == "Random Forest" else "feature_importance_gb.csv"
        importancias.reset_index().rename(columns={"index": "feature", 0: "importancia"}).to_csv(
            ARTIFACTS / fname, index=False)

    # -- Valores por defecto para el Playground (mediana num., moda categ.) --------
    defaults = {}
    for v in variables_numericas_modelo:
        defaults[v] = {
            "median": float(X[v].median()), "min": float(X[v].min()), "max": float(X[v].max()),
            "p05": float(X[v].quantile(0.05)), "p95": float(X[v].quantile(0.95)),
        }
    categorical_options = {v: sorted(X[v].dropna().unique().tolist()) for v in variables_categoricas_modelo}
    categorical_defaults = {v: X[v].mode().iloc[0] for v in variables_categoricas_modelo}

    with open(ARTIFACTS / "playground_defaults.json", "w", encoding="utf-8") as f:
        json.dump({
            "numeric": defaults,
            "categorical_options": categorical_options,
            "categorical_defaults": categorical_defaults,
            "numeric_features": variables_numericas_modelo,
            "categorical_features": variables_categoricas_modelo,
        }, f, ensure_ascii=False, indent=2)

    with open(ARTIFACTS / "model_slugs.json", "w", encoding="utf-8") as f:
        json.dump(MODEL_SLUGS, f, ensure_ascii=False, indent=2)

    # -- Estadísticas generales del dataset ------------------------------------------
    with open(ARTIFACTS / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_customers": int(n_filas),
            "n_columns_original": int(n_columnas_original),
            "n_features_used": len(variables_numericas_modelo) + len(variables_categoricas_modelo),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "class_distribution": class_dist.round(1).to_dict(),
            "baseline": baseline_stats,
            "n_outliers_iforest": n_outliers_if,
            "modelo_ganador": tabla_test.iloc[0]["modelo"],
        }, f, ensure_ascii=False, indent=2)

    print("\nListo. Resumen:")
    print(f"  {n_filas} clientes, {n_columnas_original} variables originales")
    print(f"  Baseline: acc={baseline_stats['accuracy']:.3f} f1_macro={baseline_stats['f1_macro']:.3f}")
    print(tabla_test.to_string(index=False))


if __name__ == "__main__":
    main()
