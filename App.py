from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app_core import (
    AppDataError,
    build_result_dataframe,
    generate_template_dataframe,
    load_artifacts,
    predict_clusters,
)


st.set_page_config(
    page_title="Previsao de Agrupamentos",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_resource
def get_artifacts():
    return load_artifacts(Path(__file__).resolve().parent)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def main() -> None:
    st.title("Clusterizacao de consumidores com KMeans")
    st.write(
        "Envie um arquivo CSV para classificar os registros em clusters e baixar "
        "o resultado completo."
    )

    with st.expander("Descricao dos grupos", expanded=True):
        st.markdown(
            """
- **Grupo 0**: perfil jovem com interesse em moda, musica e aparencia.
- **Grupo 1**: perfil ligado a esportes e interesses culturais como banda e rock.
- **Grupo 2**: perfil equilibrado, com interesses em musica, danca e moda.
            """
        )

    try:
        artifacts = get_artifacts()
    except FileNotFoundError as exc:
        st.error("Nao foi possivel carregar os arquivos do modelo.")
        st.exception(exc)
        st.stop()
    except Exception as exc:  # Defensive guard for deployment diagnostics.
        st.error("Falha inesperada ao carregar os artefatos.")
        st.exception(exc)
        st.stop()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        uploaded_file = st.file_uploader(
            "Upload do CSV",
            type=["csv"],
            help="O arquivo deve conter as colunas esperadas pela pipeline.",
        )

    with col_right:
        template_df = generate_template_dataframe()
        st.download_button(
            label="Baixar CSV modelo",
            data=dataframe_to_csv_bytes(template_df),
            file_name="template_previsao_clusters.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if uploaded_file is None:
        st.info("Aguardando upload do arquivo CSV.")
        return

    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Pre-visualizacao do arquivo enviado")
        st.dataframe(df.head(10), use_container_width=True)

        clusters = predict_clusters(df, artifacts)
        result_df = build_result_dataframe(df, clusters)

        st.subheader("Resultado da clusterizacao")
        st.dataframe(result_df.head(10), use_container_width=True)

        st.subheader("Distribuicao dos clusters")
        distribution = result_df["cluster"].value_counts().sort_index()
        st.bar_chart(distribution)

        st.download_button(
            label="Baixar resultado completo",
            data=dataframe_to_csv_bytes(result_df),
            file_name="resultado_clusterizacao.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except AppDataError as exc:
        st.error("Erro de validacao no arquivo enviado.")
        st.exception(exc)
    except Exception as exc:
        st.error("Falha inesperada durante o processamento.")
        st.exception(exc)


if __name__ == "__main__":
    main()
