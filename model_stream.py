import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

# Modelos de Classificação
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Modelos de Regressão
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="ML Model Trainer",
    layout="wide"
)

st.title("🤖 ML Model Trainer")

st.markdown(
    """
    Faça upload do seu dataset, escolha o modelo de Machine Learning,
    treine e visualize os resultados diretamente na tela.
    """
)

# =========================
# Upload do Dataset
# =========================

uploaded_file = st.file_uploader(
    "📂 Faça upload do seu arquivo CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Preview do Dataset")
    st.dataframe(df.head())

    st.subheader("📌 Informações Gerais")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Linhas", df.shape[0])

    with col2:
        st.metric("Colunas", df.shape[1])

    with col3:
        st.metric("Valores Nulos", int(df.isnull().sum().sum()))

    st.subheader("🔎 Tipos de Dados")
    st.dataframe(df.dtypes.astype(str))

    # =========================
    # Configuração do Modelo
    # =========================

    st.sidebar.title("⚙️ Configurações")

    problem_type = st.sidebar.selectbox(
        "Tipo de Problema",
        ["Classificação", "Regressão"]
    )

    target_column = st.sidebar.selectbox(
        "Selecione a coluna alvo (target)",
        df.columns
    )

    test_size = st.sidebar.slider(
        "Tamanho do conjunto de teste",
        0.1, 0.5, 0.2
    )

    random_state = st.sidebar.slider(
        "Random State",
        0, 100, 42
    )

    # =========================
    # Preparação dos Dados
    # =========================

    st.subheader("🔧 Preparação dos Dados")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Identificar colunas numéricas e categóricas
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Criar transformadores
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', pd.get_dummies)
    ])

    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # One-hot encoding manual
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # =========================
    # Escolha do Modelo
    # =========================

    if problem_type == "Classificação":

        model_name = st.sidebar.selectbox(
            "Escolha o Modelo",
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
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "SVM": SVC(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
        }

    else:

        model_name = st.sidebar.selectbox(
            "Escolha o Modelo",
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
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "SVR": SVR(),
            "KNN": KNeighborsRegressor(),
        }

    model = models[model_name]

    # =========================
    # Treinamento do Modelo
    # =========================

    st.subheader("🚀 Treinamento do Modelo")

    if st.button("Treinar Modelo"):
        with st.spinner("Treinando modelo..."):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            st.success("✅ Modelo treinado com sucesso!")

            # =========================
            # Exibição de Métricas
            # =========================

            if problem_type == "Classificação":
                st.subheader("📊 Métricas de Classificação")

                accuracy = accuracy_score(y_test, predictions)

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Acurácia", f"{accuracy:.4f}")

                with col2:
                    st.subheader("Matriz de Confusão")
                    cm = confusion_matrix(y_test, predictions)
                    st.write(cm)

                st.subheader("📋 Relatório de Classificação")
                report = classification_report(y_test, predictions, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

            else:
                st.subheader("📊 Métricas de Regressão")

                mse = mean_squared_error(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("MSE", f"{mse:.4f}")

                with col2:
                    st.metric("MAE", f"{mae:.4f}")

                with col3:
                    st.metric("R²", f"{r2:.4f}")

                st.subheader("📉 Valores Reais vs Preditos")

                result_df = pd.DataFrame({
                    "Real": y_test,
                    "Predito": predictions
                })

                st.dataframe(result_df.head(20))

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(y_test, predictions)
                ax.set_xlabel("Valores Reais")
                ax.set_ylabel("Predições")
                ax.set_title("Real vs Predito")
                st.pyplot(fig)

            # =========================
            # Feature Importance
            # =========================

            if hasattr(model, 'feature_importances_'):

                st.subheader("⭐ Importância das Features")

                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)

                st.dataframe(importance_df.head(20))

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    data=importance_df.head(10),
                    x='Importance',
                    y='Feature',
                    ax=ax
                )
                ax.set_title("Top 10 Features Mais Importantes")
                st.pyplot(fig)

            # =========================
            # Download das previsões
            # =========================

            st.subheader("⬇️ Download das Previsões")

            output_df = pd.DataFrame({
                'Real': y_test,
                'Predito': predictions
            })

            csv = output_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📥 Baixar CSV",
                data=csv,
                file_name='predicoes.csv',
                mime='text/csv'
            )

else:
    st.info("📌 Faça upload de um arquivo CSV para começar.")

