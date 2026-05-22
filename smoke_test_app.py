"""Simple smoke test for the Streamlit prediction pipeline core."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app_core import INPUT_COLUMNS, build_result_dataframe, load_artifacts, predict_clusters


def load_sample_input(base_path: Path) -> pd.DataFrame:
    local_dataset = base_path / "dados_clusterizados.csv"
    if local_dataset.exists():
        df = pd.read_csv(local_dataset)
        if "cluster" in df.columns:
            df = df.drop(columns=["cluster"])
        return df[INPUT_COLUMNS].head(20)

    url = (
        "https://raw.githubusercontent.com/alura-cursos/"
        "Clusterizacao-dados-sem-rotulo/main/Dados/dados_mkt.csv"
    )
    return pd.read_csv(url).head(20)


def main() -> None:
    base_path = Path(__file__).resolve().parent
    artifacts = load_artifacts(base_path)
    sample_df = load_sample_input(base_path)

    clusters = predict_clusters(sample_df, artifacts)
    result_df = build_result_dataframe(sample_df, clusters)

    assert len(result_df) == len(sample_df), "Resultado com quantidade incorreta"
    assert "cluster" in result_df.columns, "Coluna cluster ausente"
    assert result_df["cluster"].dtype.kind in {"i", "u"}, "Cluster nao inteiro"

    unique_clusters = sorted(result_df["cluster"].unique().tolist())
    print("Smoke test OK. Clusters encontrados:", unique_clusters)


if __name__ == "__main__":
    main()

