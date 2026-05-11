import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder
)

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# =========================
# MODELOS CLASSIFICAÇÃO
# =========================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# =========================
# MODELOS REGRESSÃO
# =========================

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIG
# ==========================================

st.set_page_config(
    page_title="ML Model Trainer",
    layout="wide"
)

st.title("🤖 ML Model Trainer")

st.markdown("""
Faça upload do seu dataset CSV,
treine modelos de Machine Learning
e visualize os resultados.
""")

# ==========================================
# FUNÇÃO SEGURA LEITURA CSV
# ==========================================

def load_csv(uploaded_file):

    encodings = [
        ('utf-8', ','),
        ('latin1', ';'),
        ('cp1252', ';'),
        ('ISO-8859-1', ';'),
        ('utf-8', ';'),
    ]

    for encoding, sep in encodings:

        try:

            uploaded_file.seek(0)

            df = pd.read_csv(
                uploaded_file,
                encoding=encoding,
                sep=sep
            )

            if df.shape[1] > 1:
                return df

        except:
            continue

    return None

# ==========================================
# UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📁 Upload CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = load_csv(uploaded_file)

    if df is None:
        st.error(
            "❌ Não foi possível ler o CSV."
        )
        st.stop()

    # ==========================================
    # LIMPEZA INICIAL
    # ==========================================

    df.columns = df.columns.astype(str)

    # remover duplicadas
    df = df.drop_duplicates()

    # remover colunas totalmente vazias
    df = df.dropna(axis=1, how='all')

    # remover linhas totalmente vazias
    df = df.dropna(axis=0, how='all')

    if df.empty:
        st.error("Dataset vazio.")
        st.stop()

    # ==========================================
    # PREVIEW
    # ==========================================

    st.subheader("📊 Preview")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Linhas", df.shape[0])

    with col2:
        st.metric("Colunas", df.shape[1])

    with col3:
        st.metric(
            "Valores Nulos",
            int(df.isnull().sum().sum())
        )

    st.subheader("🔎 Tipos")
    st.dataframe(df.dtypes.astype(str))

    # ==========================================
    # SIDEBAR
    # ==========================================

    st.sidebar.title("⚙️ Configurações")

    target_column = st.sidebar.selectbox(
        "Coluna Target",
        df.columns
    )

    # Detectar automaticamente
    if (
        df[target_column].dtype == 'object'
        or df[target_column].nunique() < 20
    ):
        default_problem = "Classificação"
    else:
        default_problem = "Regressão"

    problem_type = st.sidebar.selectbox(
        "Tipo de Problema",
        ["Classificação", "Regressão"],
        index=0 if default_problem == "Classificação" else 1
    )

    test_size = st.sidebar.slider(
        "Teste %",
        0.1,
        0.5,
        0.2
    )

    random_state = st.sidebar.slider(
        "Random State",
        0,
        100,
        42
    )

    # ==========================================
    # PREPARAÇÃO
    # ==========================================

    st.subheader("🔧 Preparação")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    # remover target nulo
    mask = y.notnull()

    X = X[mask]
    y = y[mask]

    # ==========================================
    # CLASSIFICAÇÃO
    # ==========================================

    if problem_type == "Classificação":

        y = y.astype(str)

        if y.nunique() < 2:
            st.error(
                "❌ Target precisa ter pelo menos 2 classes."
            )
            st.stop()

        # codificar target
        le = LabelEncoder()
        y = le.fit_transform(y)

    # ==========================================
    # REGRESSÃO
    # ==========================================

    else:

        y = pd.to_numeric(
            y,
            errors='coerce'
        )

        mask = y.notnull()

        X = X[mask]
        y = y[mask]

        if len(y) < 5:
            st.error(
                "❌ Poucos dados válidos para regressão."
            )
            st.stop()

    # ==========================================
    # FEATURES
    # ==========================================

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    # ==========================================
    # PIPELINES
    # ==========================================

    numeric_transformer = Pipeline(steps=[
        (
            'imputer',
            SimpleImputer(strategy='median')
        ),
        (
            'scaler',
            StandardScaler()
        )
    ])

    categorical_transformer = Pipeline(steps=[
        (
            'imputer',
            SimpleImputer(
                strategy='most_frequent'
            )
        ),
        (
            'onehot',
            OneHotEncoder(
                handle_unknown='ignore'
            )
        )
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'num',
                numeric_transformer,
                numeric_features
            ),
            (
                'cat',
                categorical_transformer,
                categorical_features
            )
        ]
    )

    # ==========================================
    # TRANSFORMAÇÃO
    # ==========================================

    try:

        X_processed = preprocessor.fit_transform(X)

        # remover NaN restantes
        X_processed = np.nan_to_num(X_processed)

    except Exception as e:

        st.error(
            f"Erro no preprocessamento: {e}"
        )

        st.stop()

    # ==========================================
    # TRAIN TEST SPLIT
    # ==========================================

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed,
            y,
            test_size=test_size,
            random_state=random_state
        )

    except Exception as e:

        st.error(
            f"Erro train_test_split: {e}"
        )

        st.stop()

    # ==========================================
    # MODELOS
    # ==========================================

    if problem_type == "Classificação":

        model_name = st.sidebar.selectbox(
            "Modelo",
            [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "SVM",
                "KNN",
                "Naive Bayes"
            ]
        )

        models = {

            "Logistic Regression":
                LogisticRegression(max_iter=2000),

            "Decision Tree":
                DecisionTreeClassifier(),

            "Random Forest":
                RandomForestClassifier(),

            "Gradient Boosting":
                GradientBoostingClassifier(),

            "SVM":
                SVC(),

            "KNN":
                KNeighborsClassifier(),

            "Naive Bayes":
                GaussianNB(),
        }

    else:

        model_name = st.sidebar.selectbox(
            "Modelo",
            [
                "Linear Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "SVR",
                "KNN"
            ]
        )

        models = {

            "Linear Regression":
                LinearRegression(),

            "Decision Tree":
                DecisionTreeRegressor(),

            "Random Forest":
                RandomForestRegressor(),

            "Gradient Boosting":
                GradientBoostingRegressor(),

            "SVR":
                SVR(),

            "KNN":
                KNeighborsRegressor(),
        }

    model = models[model_name]

    # ==========================================
    # TREINAR
    # ==========================================

    st.subheader("🚀 Treinamento")

    if st.button("Treinar Modelo"):

        try:

            with st.spinner("Treinando..."):

                model.fit(
                    X_train,
                    y_train
                )

                predictions = model.predict(X_test)

            st.success(
                "✅ Modelo treinado!"
            )

        except Exception as e:

            st.error(
                f"Erro treinamento: {e}"
            )

            st.stop()

        # ==========================================
        # CLASSIFICAÇÃO
        # ==========================================

        if problem_type == "Classificação":

            try:

                accuracy = accuracy_score(
                    y_test,
                    predictions
                )

                st.metric(
                    "Acurácia",
                    f"{accuracy:.4f}"
                )

                st.subheader(
                    "📋 Classification Report"
                )

                report = classification_report(
                    y_test,
                    predictions,
                    output_dict=True
                )

                st.dataframe(
                    pd.DataFrame(report).transpose()
                )

                st.subheader(
                    "📌 Matriz de Confusão"
                )

                cm = confusion_matrix(
                    y_test,
                    predictions
                )

                fig, ax = plt.subplots()

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    ax=ax
                )

                st.pyplot(fig)

            except Exception as e:

                st.error(
                    f"Erro métricas classificação: {e}"
                )

        # ==========================================
        # REGRESSÃO
        # ==========================================

        else:

            try:

                mse = mean_squared_error(
                    y_test,
                    predictions
                )

                mae = mean_absolute_error(
                    y_test,
                    predictions
                )

                r2 = r2_score(
                    y_test,
                    predictions
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "MSE",
                        f"{mse:.4f}"
                    )

                with col2:
                    st.metric(
                        "MAE",
                        f"{mae:.4f}"
                    )

                with col3:
                    st.metric(
                        "R²",
                        f"{r2:.4f}"
                    )

                fig, ax = plt.subplots(
                    figsize=(6, 4)
                )

                ax.scatter(
                    y_test,
                    predictions
                )

                ax.set_xlabel(
                    "Real"
                )

                ax.set_ylabel(
                    "Predito"
                )

                ax.set_title(
                    "Real vs Predito"
                )

                st.pyplot(fig)

            except Exception as e:

                st.error(
                    f"Erro métricas regressão: {e}"
                )

        # ==========================================
        # FEATURE IMPORTANCE
        # ==========================================

        if hasattr(model, "feature_importances_"):

            try:

                st.subheader(
                    "⭐ Feature Importance"
                )

                cat_names = []

                if len(categorical_features) > 0:

                    cat_names = list(
                        preprocessor.named_transformers_[
                            'cat'
                        ]['onehot'].get_feature_names_out(
                            categorical_features
                        )
                    )

                feature_names = (
                    numeric_features +
                    cat_names
                )

                importance_df = pd.DataFrame({

                    "Feature":
                        feature_names,

                    "Importance":
                        model.feature_importances_

                })

                importance_df = importance_df.sort_values(
                    by="Importance",
                    ascending=False
                )

                st.dataframe(
                    importance_df.head(20)
                )

                fig, ax = plt.subplots(
                    figsize=(10, 6)
                )

                sns.barplot(
                    data=importance_df.head(10),
                    x="Importance",
                    y="Feature",
                    ax=ax
                )

                st.pyplot(fig)

            except Exception as e:

                st.warning(
                    f"Erro feature importance: {e}"
                )

        # ==========================================
        # DOWNLOAD
        # ==========================================

        try:

            st.subheader(
                "⬇️ Download"
            )

            output_df = pd.DataFrame({
                "Real": y_test,
                "Predito": predictions
            })

            csv = output_df.to_csv(
                index=False
            ).encode('utf-8')

            st.download_button(
                label="📥 Baixar CSV",
                data=csv,
                file_name="predicoes.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.warning(
                f"Erro download: {e}"
            )

else:

    st.info(
        "📌 Faça upload de um CSV."
    )

