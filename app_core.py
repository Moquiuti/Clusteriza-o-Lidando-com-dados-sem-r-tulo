"""Core utilities for cluster prediction app.

This module keeps the prediction pipeline isolated from Streamlit UI,
which makes validation and testing easier.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd


INPUT_COLUMNS: List[str] = [
    "sexo",
    "idade",
    "numero_de_amigos",
    "basquete",
    "futebol_americano",
    "futebol",
    "softbol",
    "voleibol",
    "natacao",
    "animacao",
    "beisebol",
    "tenis",
    "esportes",
    "fofo",
    "danca",
    "banda",
    "marcha",
    "musica",
    "rock",
    "cabelo",
    "vestido",
    "shopping",
    "compras",
    "roupas",
    "nossa_marca",
    "marca_concorrente",
    "bebidas",
]

CLUSTER_DESCRIPTIONS: Dict[int, str] = {
    0: "Perfil jovem com interesse em moda, musica e aparencia.",
    1: "Perfil mais ligado a esportes e interesses culturais como banda/rock.",
    2: "Perfil equilibrado, com interesses em musica, danca e moda.",
}


class AppDataError(ValueError):
    """Raised when uploaded data is incompatible with prediction pipeline."""


@dataclass(frozen=True)
class Artifacts:
    encoder: object
    scaler: object
    kmeans: object


DEFAULT_ARTIFACT_PATHS = {
    "encoder": "encoder.pkl",
    "scaler": "scaler.pkl",
    "kmeans": "kmeans.pkl",
}


def load_artifacts(base_path: Path | str = ".") -> Artifacts:
    base = Path(base_path)

    missing = [
        name
        for name, rel_path in DEFAULT_ARTIFACT_PATHS.items()
        if not (base / rel_path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Arquivos de modelo ausentes: " + ", ".join(sorted(missing))
        )

    return Artifacts(
        encoder=joblib.load(base / DEFAULT_ARTIFACT_PATHS["encoder"]),
        scaler=joblib.load(base / DEFAULT_ARTIFACT_PATHS["scaler"]),
        kmeans=joblib.load(base / DEFAULT_ARTIFACT_PATHS["kmeans"]),
    )


def _validate_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in INPUT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise AppDataError(
            "Colunas obrigatorias ausentes no CSV: " + ", ".join(missing_columns)
        )


def _normalize_and_validate_sex(df: pd.DataFrame, encoder: object) -> pd.Series:
    sexo = df["sexo"].astype(str).str.strip().str.upper()
    valid_categories = {str(value).upper() for value in encoder.categories_[0]}
    invalid_values = sorted(set(sexo) - valid_categories)

    if invalid_values:
        invalid_preview = ", ".join(invalid_values[:5])
        raise AppDataError(
            "Valores invalidos na coluna 'sexo': "
            f"{invalid_preview}. Use apenas: {', '.join(sorted(valid_categories))}."
        )

    return sexo


def preprocess_input(
    raw_df: pd.DataFrame,
    encoder: object,
    scaler: object,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _validate_columns(raw_df)

    working_df = raw_df.copy()
    working_df["sexo"] = _normalize_and_validate_sex(working_df, encoder)

    encoded_values = encoder.transform(working_df[["sexo"]])
    encoded_df = pd.DataFrame(
        encoded_values,
        columns=encoder.get_feature_names_out(["sexo"]),
        index=working_df.index,
    )

    model_input_df = pd.concat(
        [working_df.drop(columns=["sexo"]), encoded_df],
        axis=1,
    )

    model_input_df = model_input_df.apply(pd.to_numeric, errors="coerce")
    if model_input_df.isna().any().any():
        bad_columns = model_input_df.columns[model_input_df.isna().any()].tolist()
        raise AppDataError(
            "Existem valores nao numericos ou vazios em colunas numericas: "
            + ", ".join(bad_columns)
        )

    scaled_array = scaler.transform(model_input_df)
    scaled_df = pd.DataFrame(
        scaled_array,
        columns=model_input_df.columns,
        index=model_input_df.index,
    )

    return scaled_df, working_df


def predict_clusters(raw_df: pd.DataFrame, artifacts: Artifacts) -> pd.Series:
    scaled_df, _ = preprocess_input(raw_df, artifacts.encoder, artifacts.scaler)
    predictions = artifacts.kmeans.predict(scaled_df)
    return pd.Series(predictions, name="cluster", index=raw_df.index)


def build_result_dataframe(raw_df: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    result_df = raw_df.copy()
    result_df.insert(0, "cluster", clusters.astype(int))
    result_df.insert(
        1,
        "descricao_cluster",
        result_df["cluster"].map(CLUSTER_DESCRIPTIONS).fillna("Perfil nao mapeado"),
    )
    return result_df


def generate_template_dataframe() -> pd.DataFrame:
    # Template helps users understand the exact shape required by the app.
    base = {column: 0 for column in INPUT_COLUMNS}
    base["sexo"] = "F"
    return pd.DataFrame([base])

