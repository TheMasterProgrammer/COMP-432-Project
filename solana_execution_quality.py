
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from solana.rpc.api import Client
    from solana.rpc.types import TxOpts
    from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
    from solders.instruction import Instruction
    from solders.keypair import Keypair
    from solders.message import MessageV0
    from solders.pubkey import Pubkey
    from solders.system_program import TransferParams, transfer
    from solders.transaction import VersionedTransaction
    SOLANA_SDK_AVAILABLE = True
except Exception:
    Client = Any  
    TxOpts = Any 
    Instruction = Any
    Keypair = Any 
    Pubkey = Any 
    VersionedTransaction = Any  
    MessageV0 = Any 
    SOLANA_SDK_AVAILABLE = False


BASE_NUMERIC_FEATURES = [
    "cu_limit",
    "cu_price_micro_lamports",
    "priority_fee_lamports",
    "fee_cap_lamports",
    "recent_fee_min",
    "recent_fee_median",
    "recent_fee_p90",
    "recent_fee_max",
    "local_recent_fee_min",
    "local_recent_fee_median",
    "local_recent_fee_p90",
    "local_recent_fee_max",
    "recent_tps",
    "recent_slots_per_second",
    "message_size_bytes",
    "instruction_count",
    "account_count",
    "writable_account_count",
    "memo_size_bytes",
    "extra_transfer_count",
    "lamports",
    "time_of_day_sin",
    "time_of_day_cos",
    "day_of_week",
]

BASE_CATEGORICAL_FEATURES = [
    "send_method",
    "pricing_policy",
    "tx_variant",
]

REAL_DATA_REQUIRED_COLUMNS = [
    "timestamp_utc",
    "send_method",
    "pricing_policy",
    "tx_variant",
    "cu_limit",
    "cu_price_micro_lamports",
    "priority_fee_lamports",
    "fee_cap_lamports",
    "recent_fee_min",
    "recent_fee_median",
    "recent_fee_p90",
    "recent_fee_max",
    "local_recent_fee_min",
    "local_recent_fee_median",
    "local_recent_fee_p90",
    "local_recent_fee_max",
    "recent_tps",
    "recent_slots_per_second",
    "message_size_bytes",
    "instruction_count",
    "account_count",
    "writable_account_count",
    "memo_size_bytes",
    "extra_transfer_count",
    "lamports",
    "landed_within_1_slots",
    "landed_within_2_slots",
    "landed_within_3_slots",
    "latency_ms",
]


@dataclass
class SplitData:
    """Container for train, validation, and test splits."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass
class ModelResult:
    """Structured model evaluation output."""

    model_name: str
    target_name: str
    metrics: dict[str, float]
    fitted_model: Pipeline


@dataclass
class NetworkSnapshot:
    """Compact snapshot of recent fee and performance signals."""

    recent_fee_min: float
    recent_fee_median: float
    recent_fee_p90: float
    recent_fee_max: float
    recent_tps: float
    recent_slots_per_second: float


MEMO_PROGRAM_ID = "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"


def build_memo_instruction(memo_text: str) -> Instruction:
    """Build a Solana memo instruction."""
    ensure_solana_sdk()
    return Instruction(
        program_id=Pubkey.from_string(MEMO_PROGRAM_ID),
        accounts=[],
        data=memo_text.encode("utf-8"),
    )


def choose_fee_cap_lamports(rng: random.Random) -> int:
    """Sample a fee cap to create variation in the dataset."""
    return rng.choices(
        population=[5, 10, 20, 30, 50, 80, 120, 200],
        weights=[0.06, 0.10, 0.20, 0.20, 0.18, 0.14, 0.08, 0.04],
        k=1,
    )[0]


def choose_pricing_policy(rng: random.Random) -> str:
    """Sample a pricing policy to avoid a constant pricing_policy column."""
    return rng.choices(
        population=["recent_fee", "sample_under_cap", "cap_max"],
        weights=[0.30, 0.50, 0.20],
        k=1,
    )[0]


def choose_tx_variant(rng: random.Random) -> dict[str, Any]:
    """
    Choose a harmless transaction shape variant.

    Returns a dict with:
        tx_variant: str
        memo_text: str
        extra_transfer_count: int
    """
    variant = rng.choices(
        population=[
            "plain",
            "memo_small",
            "memo_medium",
            "memo_large",
            "memo_small_plus_extra_transfer",
        ],
        weights=[0.22, 0.22, 0.22, 0.18, 0.16],
        k=1,
    )[0]

    if variant == "plain":
        return {
            "tx_variant": variant,
            "memo_text": "",
            "extra_transfer_count": 0,
        }

    if variant == "memo_small":
        return {
            "tx_variant": variant,
            "memo_text": "x" * 16,
            "extra_transfer_count": 0,
        }

    if variant == "memo_medium":
        return {
            "tx_variant": variant,
            "memo_text": "m" * 64,
            "extra_transfer_count": 0,
        }

    if variant == "memo_large":
        return {
            "tx_variant": variant,
            "memo_text": "l" * 160,
            "extra_transfer_count": 0,
        }

    return {
        "tx_variant": variant,
        "memo_text": "e" * 32,
        "extra_transfer_count": 1,
    }


def snapshot_network_for_accounts(
    client: SolanaRpcClient,
    writable_accounts: list[str],
) -> dict[str, float]:
    """
    Collect account-conditioned prioritization fee statistics.

    Falls back to zeros if the RPC returns no account-specific fee samples.
    """
    fee_samples = client.get_recent_prioritization_fees(writable_accounts=writable_accounts)
    fees = [float(item.get("prioritizationFee", 0.0)) for item in fee_samples]
    if not fees:
        fees = [0.0]

    return {
        "local_recent_fee_min": float(np.min(fees)),
        "local_recent_fee_median": float(np.median(fees)),
        "local_recent_fee_p90": float(np.quantile(fees, 0.90)),
        "local_recent_fee_max": float(np.max(fees)),
    }


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def ensure_solana_sdk() -> None:
    """Raise a helpful error when the Solana Python SDK is missing."""
    if not SOLANA_SDK_AVAILABLE:
        raise ImportError(
            "Solana collection helpers require the 'solana' and 'solders' packages. "
            "Install them with: pip install solana solders"
        )


class SolanaRpcClient:
    """Small JSON-RPC helper for Solana HTTP endpoints."""

    def __init__(self, rpc_url: str, timeout_seconds: int = 20) -> None:
        self.rpc_url = rpc_url
        self.timeout_seconds = timeout_seconds

    def call(self, method: str, params: list[Any] | None = None) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or [],
        }
        response = requests.post(
            self.rpc_url,
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"RPC error for {method}: {data['error']}")
        return data["result"]

    def get_recent_prioritization_fees(
        self,
        writable_accounts: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        params: list[Any] = []
        if writable_accounts:
            params.append(writable_accounts)
        return self.call("getRecentPrioritizationFees", params)

    def get_recent_performance_samples(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return self.call("getRecentPerformanceSamples", [limit])

    def get_signature_statuses(
        self,
        signatures: list[str],
        search_transaction_history: bool = True,
    ) -> list[dict[str, Any] | None]:
        result = self.call(
            "getSignatureStatuses",
            [signatures, {"searchTransactionHistory": search_transaction_history}],
        )
        return result["value"]

    def get_signature_status(self, signature: str) -> dict[str, Any] | None:
        return self.get_signature_statuses([signature])[0]

    def get_slot(self, commitment: str = "processed") -> int:
        return int(self.call("getSlot", [{"commitment": commitment}]))

    def snapshot_network(self) -> NetworkSnapshot:
        fee_samples = self.get_recent_prioritization_fees()
        performance_samples = self.get_recent_performance_samples(limit=20)

        fees = [float(item.get("prioritizationFee", 0.0)) for item in fee_samples]
        if not fees:
            fees = [0.0]

        sample_num_transactions = [
            float(item.get("numTransactions", 0.0)) for item in performance_samples
        ]
        sample_num_slots = [float(item.get("numSlots", 0.0)) for item in performance_samples]
        sample_period_seconds = [
            float(item.get("samplePeriodSecs", 60.0)) for item in performance_samples
        ]

        total_period_seconds = max(sum(sample_period_seconds), 1.0)
        recent_tps = sum(sample_num_transactions) / total_period_seconds
        recent_slots_per_second = sum(sample_num_slots) / total_period_seconds

        return NetworkSnapshot(
            recent_fee_min=float(np.min(fees)),
            recent_fee_median=float(np.median(fees)),
            recent_fee_p90=float(np.quantile(fees, 0.90)),
            recent_fee_max=float(np.max(fees)),
            recent_tps=recent_tps,
            recent_slots_per_second=recent_slots_per_second,
        )


class RawLogDatasetBuilder:
    """Augment sender logs into a modeling dataset."""

    def __init__(self, rpc_url: str) -> None:
        self.client = SolanaRpcClient(rpc_url=rpc_url)

    def snapshot_network_features(
        self,
        writable_accounts: list[str] | None = None,
    ) -> dict[str, float]:
        fee_samples = self.client.get_recent_prioritization_fees(writable_accounts)
        performance_samples = self.client.get_recent_performance_samples(limit=20)

        fees = [float(item.get("prioritizationFee", 0.0)) for item in fee_samples]
        if not fees:
            fees = [0.0]

        sample_num_transactions = [
            float(item.get("numTransactions", 0.0)) for item in performance_samples
        ]
        sample_num_slots = [float(item.get("numSlots", 0.0)) for item in performance_samples]
        sample_period_seconds = [
            float(item.get("samplePeriodSecs", 60.0)) for item in performance_samples
        ]

        total_period_seconds = max(sum(sample_period_seconds), 1.0)
        recent_tps = sum(sample_num_transactions) / total_period_seconds
        recent_slots_per_second = sum(sample_num_slots) / total_period_seconds

        return {
            "recent_fee_min": float(np.min(fees)),
            "recent_fee_median": float(np.median(fees)),
            "recent_fee_p90": float(np.quantile(fees, 0.90)),
            "recent_fee_max": float(np.max(fees)),
            "recent_tps": recent_tps,
            "recent_slots_per_second": recent_slots_per_second,
        }

    def attach_labels_from_statuses(
        self,
        frame: pd.DataFrame,
        signature_column: str = "signature",
        sent_slot_column: str = "sent_slot",
        sent_time_column: str = "sent_at_utc",
    ) -> pd.DataFrame:
        statuses = self.client.get_signature_statuses(frame[signature_column].tolist())
        enriched = frame.copy()
        enriched[sent_time_column] = pd.to_datetime(enriched[sent_time_column], utc=True)

        confirmed_slots: list[float] = []
        confirmation_status: list[str | None] = []
        has_error: list[int] = []

        for status in statuses:
            if status is None:
                confirmed_slots.append(np.nan)
                confirmation_status.append(None)
                has_error.append(1)
                continue
            confirmed_slots.append(float(status.get("slot", np.nan)))
            confirmation_status.append(status.get("confirmationStatus"))
            has_error.append(0 if status.get("err") in (None, {}) else 1)

        enriched["landed_slot"] = confirmed_slots
        enriched["confirmation_status"] = confirmation_status
        enriched["rpc_status_has_error"] = has_error
        enriched["slot_delta"] = enriched["landed_slot"] - enriched[sent_slot_column]
        for k in (1, 2, 3):
            enriched[f"landed_within_{k}_slots"] = (
                enriched["slot_delta"].notna() & (enriched["slot_delta"] <= k)
            ).astype(int)
        return enriched


def compute_priority_fee_lamports(
    cu_limit: int | float,
    cu_price_micro_lamports: int | float,
) -> int:
    return int(math.ceil(float(cu_limit) * float(cu_price_micro_lamports) / 1_000_000.0))


def add_calendar_features(
    frame: pd.DataFrame,
    timestamp_column: str = "timestamp_utc",
) -> pd.DataFrame:
    enriched = frame.copy()
    enriched[timestamp_column] = pd.to_datetime(enriched[timestamp_column], utc=True)
    seconds = (
        enriched[timestamp_column].dt.hour * 3600
        + enriched[timestamp_column].dt.minute * 60
        + enriched[timestamp_column].dt.second
    )
    enriched["time_of_day_sin"] = np.sin(2.0 * np.pi * seconds / 86_400.0)
    enriched["time_of_day_cos"] = np.cos(2.0 * np.pi * seconds / 86_400.0)
    enriched["day_of_week"] = enriched[timestamp_column].dt.dayofweek.astype(int)
    return enriched



def time_based_split(
    frame: pd.DataFrame,
    timestamp_column: str = "timestamp_utc",
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> SplitData:
    ordered = frame.sort_values(timestamp_column).reset_index(drop=True)
    train_end = int(len(ordered) * train_fraction)
    val_end = int(len(ordered) * (train_fraction + val_fraction))
    return SplitData(
        train=ordered.iloc[:train_end].copy(),
        val=ordered.iloc[train_end:val_end].copy(),
        test=ordered.iloc[val_end:].copy(),
    )


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def build_classification_models(
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Pipeline]:
    linear_pre = build_preprocessor(numeric_features, categorical_features, scale_numeric=True)
    tree_pre = build_preprocessor(numeric_features, categorical_features, scale_numeric=False)

    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", linear_pre),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1200,
                        class_weight="balanced",
                        n_jobs=None,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", tree_pre),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=12,
                        min_samples_leaf=8,
                        random_state=42,
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("preprocessor", tree_pre),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=400,
                        max_depth=14,
                        min_samples_leaf=6,
                        random_state=42,
                        n_jobs=-1,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }


def build_regression_models(
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Pipeline]:
    tree_pre = build_preprocessor(numeric_features, categorical_features, scale_numeric=False)

    return {
        "random_forest_regressor": Pipeline(
            [
                ("preprocessor", tree_pre),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=14,
                        min_samples_leaf=8,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees_regressor": Pipeline(
            [
                ("preprocessor", tree_pre),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=400,
                        max_depth=16,
                        min_samples_leaf=6,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def train_and_evaluate_classifiers(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target_name: str,
    feature_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, np.ndarray]]:
    x_train = train_frame[feature_columns]
    y_train = train_frame[target_name].astype(int)
    x_test = test_frame[feature_columns]
    y_test = test_frame[target_name].astype(int)

    models = build_classification_models(numeric_features, categorical_features)
    rows: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}
    probability_map: dict[str, np.ndarray] = {}

    for model_name, pipeline in models.items():
        fitted = clone(pipeline)
        fitted.fit(x_train, y_train)
        probabilities = fitted.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        rows.append(
            {
                "model": model_name,
                "target": target_name,
                "roc_auc": roc_auc_score(y_test, probabilities),
                "average_precision": average_precision_score(y_test, probabilities),
                "brier_score": brier_score_loss(y_test, probabilities),
                "positive_rate": float(predictions.mean()),
                "baseline_positive_rate": float(y_test.mean()),
            }
        )
        fitted_models[model_name] = fitted
        probability_map[model_name] = probabilities

    table = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return table, fitted_models, probability_map


def train_and_evaluate_regressors(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    target_name: str = "latency_ms",
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, np.ndarray]]:
    x_train = train_frame[feature_columns]
    y_train = train_frame[target_name].astype(float)
    x_test = test_frame[feature_columns]
    y_test = test_frame[target_name].astype(float)

    models = build_regression_models(numeric_features, categorical_features)
    rows: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}
    prediction_map: dict[str, np.ndarray] = {}

    for model_name, pipeline in models.items():
        fitted = clone(pipeline)
        fitted.fit(x_train, y_train)
        predictions = fitted.predict(x_test)
        rows.append(
            {
                "model": model_name,
                "target": target_name,
                "mae_ms": mean_absolute_error(y_test, predictions),
                "median_ae_ms": median_absolute_error(y_test, predictions),
                "r2": r2_score(y_test, predictions),
            }
        )
        fitted_models[model_name] = fitted
        prediction_map[model_name] = predictions

    table = pd.DataFrame(rows).sort_values("mae_ms").reset_index(drop=True)
    return table, fitted_models, prediction_map


def build_policy_table(
    test_frame: pd.DataFrame,
    probability_column: str,
    thresholds: Iterable[float] = (0.50, 0.60, 0.70, 0.80, 0.90),
    target_column: str = "landed_within_3_slots",
    latency_column: str = "latency_ms",
    fee_column: str = "priority_fee_lamports",
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        subset = test_frame[test_frame[probability_column] >= threshold].copy()
        acceptance_rate = 0.0 if len(test_frame) == 0 else len(subset) / len(test_frame)
        if subset.empty:
            rows.append(
                {
                    "threshold": float(threshold),
                    "acceptance_rate": acceptance_rate,
                    "success_rate": np.nan,
                    "median_latency_ms": np.nan,
                    "avg_priority_fee_lamports": np.nan,
                }
            )
            continue

        rows.append(
            {
                "threshold": float(threshold),
                "acceptance_rate": acceptance_rate,
                "success_rate": float(subset[target_column].mean()),
                "median_latency_ms": float(subset[latency_column].median()),
                "avg_priority_fee_lamports": float(subset[fee_column].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_congestion_regimes(
    frame: pd.DataFrame,
    target_column: str = "landed_within_3_slots",
) -> pd.DataFrame:
    working = frame.copy()
    def _choose_signal(column_names: list[str]) -> tuple[pd.Series, str]:
        for column_name in column_names:
            if column_name not in working.columns:
                continue
            series = pd.to_numeric(working[column_name], errors="coerce")
            if series.dropna().nunique() > 1:
                return series, column_name
        return pd.Series(np.nan, index=working.index, dtype=float), "none"

    signal, signal_source = _choose_signal(
        ["recent_fee_p90", "recent_tps", "recent_slots_per_second"]
    )
    valid_signal = signal.dropna()

    if valid_signal.empty:
        working["regime"] = "unknown"
    else:
        ranked_signal = signal.rank(method="first")
        working["regime"] = pd.qcut(
            ranked_signal,
            q=3,
            labels=["low", "medium", "high"],
            duplicates="drop",
        )
        working["regime"] = working["regime"].astype(object)
        working.loc[signal.isna(), "regime"] = "unknown"

    regime_order = ["low", "medium", "high", "flat", "unknown"]
    summary = (
        working.groupby("regime", observed=True)
        .agg(
            n=(target_column, "size"),
            success_rate=(target_column, "mean"),
            median_latency_ms=("latency_ms", "median"),
            avg_priority_fee_lamports=("priority_fee_lamports", "mean"),
        )
        .reset_index()
    )
    summary["regime_signal"] = signal_source
    summary["regime"] = pd.Categorical(summary["regime"], categories=regime_order, ordered=True)
    return summary.sort_values("regime").reset_index(drop=True)


def plot_probability_histogram(
    probabilities: np.ndarray,
    target: pd.Series,
    title: str,
) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(probabilities[target.to_numpy() == 0], bins=30, alpha=0.65, label="negative")
    plt.hist(probabilities[target.to_numpy() == 1], bins=30, alpha=0.65, label="positive")
    plt.title(title)
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_regression_scatter(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray,
    title: str,
) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(actual, predicted, alpha=0.35)
    low = float(min(np.min(actual), np.min(predicted)))
    high = float(max(np.max(actual), np.max(predicted)))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.title(title)
    plt.xlabel("Actual latency (ms)")
    plt.ylabel("Predicted latency (ms)")
    plt.tight_layout()
    plt.show()


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def generate_keypair_file(
    output_path: str | Path = Path.home() / ".config" / "solana" / "id.json",
    overwrite: bool = False,
) -> str:
    """Generate a Solana keypair JSON file and return the public key as a string."""
    ensure_solana_sdk()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        existing = load_keypair(output_path)
        return str(existing.pubkey())
    kp = Keypair()
    output_path.write_text(json.dumps(list(bytes(kp))))
    return str(kp.pubkey())


def load_keypair(keypair_path: Path) -> Keypair:
    ensure_solana_sdk()
    values = json.loads(keypair_path.read_text())
    if not isinstance(values, list):
        raise ValueError("Keypair file must contain a JSON array of integers.")
    return Keypair.from_bytes(bytes(values))


def rolling_quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(ordered[lo])
    frac = idx - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def choose_cu_limit(rng: random.Random) -> int:
    return rng.choices(
        population=[120_000, 160_000, 200_000, 250_000, 300_000],
        weights=[0.10, 0.20, 0.35, 0.20, 0.15],
        k=1,
    )[0]


def choose_micro_lamports_price(
    rng: random.Random,
    snapshot: NetworkSnapshot,
    cu_limit: int,
    fee_cap_lamports: int,
    pricing_policy: str = "sample_under_cap",
) -> int:
    """
    Choose a CU price in micro-lamports.

    pricing_policy:
        - "recent_fee": original behavior, based on recent_fee_p90 and capped
        - "sample_under_cap": sample a target priority fee under the hard cap
        - "cap_max": always push as high as possible under the cap
    """
    max_price_from_cap = max(1, int(math.floor(fee_cap_lamports * 1_000_000 / cu_limit)))

    if pricing_policy == "recent_fee":
        ratios = [0.55, 0.75, 0.90, 1.00, 1.15, 1.35, 1.60]
        weights = [0.05, 0.12, 0.18, 0.24, 0.18, 0.14, 0.09]
        target_price = max(
            1,
            int(round(snapshot.recent_fee_p90 * rng.choices(ratios, weights=weights, k=1)[0])),
        )
        return int(min(target_price, max_price_from_cap))

    if pricing_policy == "cap_max":
        return int(max_price_from_cap)

    if pricing_policy == "sample_under_cap":
        target_priority_fee_lamports = rng.choices(
            population=[
                max(1, int(round(fee_cap_lamports * 0.10))),
                max(1, int(round(fee_cap_lamports * 0.25))),
                max(1, int(round(fee_cap_lamports * 0.50))),
                max(1, int(round(fee_cap_lamports * 0.75))),
                max(1, int(round(fee_cap_lamports * 1.00))),
            ],
            weights=[0.12, 0.18, 0.30, 0.24, 0.16],
            k=1,
        )[0]

        target_price = max(
            1,
            int(math.floor(target_priority_fee_lamports * 1_000_000 / cu_limit)),
        )
        return int(min(target_price, max_price_from_cap))

    raise ValueError(f"Unsupported pricing_policy: {pricing_policy}")


def build_transaction(
    payer: Keypair,
    recipient: Pubkey,
    recent_blockhash: Any,
    cu_limit: int,
    cu_price_micro_lamports: int,
    lamports: int,
    memo_text: str = "",
    extra_transfer_count: int = 0,
) -> tuple[VersionedTransaction, int, int, int, int, list[str], int, int]:
    ensure_solana_sdk()

    instructions: list[Any] = [
        set_compute_unit_limit(cu_limit),
        set_compute_unit_price(cu_price_micro_lamports),
        transfer(
            TransferParams(
                from_pubkey=payer.pubkey(),
                to_pubkey=recipient,
                lamports=lamports,
            )
        ),
    ]

    if memo_text:
        instructions.append(build_memo_instruction(memo_text))

    for _ in range(extra_transfer_count):
        instructions.append(
            transfer(
                TransferParams(
                    from_pubkey=payer.pubkey(),
                    to_pubkey=recipient,
                    lamports=lamports,
                )
            )
        )

    message = MessageV0.try_compile(
        payer=payer.pubkey(),
        instructions=instructions,
        address_lookup_table_accounts=[],
        recent_blockhash=recent_blockhash,
    )
    tx = VersionedTransaction(message, [payer])

    account_keys = list(message.account_keys)
    writable_count = 0
    writable_accounts: list[str] = []
    for idx in range(len(account_keys)):
        if message.is_maybe_writable(idx):
            writable_count += 1
            writable_accounts.append(str(account_keys[idx]))

    message_size_bytes = len(bytes(tx))
    instruction_count = len(instructions)
    account_count = len(account_keys)
    memo_size_bytes = len(memo_text.encode("utf-8")) if memo_text else 0

    return (
        tx,
        message_size_bytes,
        instruction_count,
        account_count,
        writable_count,
        writable_accounts,
        memo_size_bytes,
        extra_transfer_count,
    )


def poll_for_landing(
    rpc: SolanaRpcClient,
    signature: str,
    sent_slot: int,
    started_monotonic: float,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = started_monotonic + timeout_seconds
    last_status: dict[str, Any] | None = None

    while time.monotonic() < deadline:
        status = rpc.get_signature_status(signature)
        last_status = status
        if status is not None:
            landed_slot = status.get("slot")
            latency_ms = (time.monotonic() - started_monotonic) * 1000.0
            slot_delta = None
            if landed_slot is not None:
                slot_delta = int(landed_slot) - int(sent_slot)
            err = status.get("err")
            return {
                "landed_slot": float(landed_slot) if landed_slot is not None else None,
                "latency_ms": float(latency_ms),
                "slot_delta": float(slot_delta) if slot_delta is not None else None,
                "confirmation_status": status.get("confirmationStatus"),
                "rpc_status_has_error": 0 if err in (None, {}) else 1,
            }
        time.sleep(poll_interval_seconds)

    return {
        "landed_slot": None,
        "latency_ms": float(timeout_seconds * 1000.0),
        "slot_delta": None,
        "confirmation_status": None if last_status is None else last_status.get("confirmationStatus"),
        "rpc_status_has_error": 1,
    }


def ensure_csv_header(path: Path, fieldnames: list[str]) -> None:
    if path.exists():
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def append_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(row)


def collect_real_transactions(
    rpc_url: str,
    keypair_path: str | Path,
    output: str | Path = "real_transactions.csv",
    recipient: str | None = None,
    num_samples: int = 200,
    lamports: int = 1,
    fee_cap_lamports: int | None = None,
    poll_interval_seconds: float = 0.50,
    timeout_seconds: float = 45.0,
    sleep_seconds: float = 1.0,
    random_seed: int = 42,
    print_progress: bool = True,
    pricing_policy: str | None = None,
) -> Path:
    """Collect a richer real transaction dataset and append rows to a CSV file."""
    ensure_solana_sdk()
    rng = random.Random(random_seed)
    rpc = SolanaRpcClient(rpc_url)
    client = Client(rpc_url)
    payer = load_keypair(Path(keypair_path))
    recipient_pubkey = Pubkey.from_string(recipient) if recipient else payer.pubkey()
    output = Path(output)

    fieldnames = [
        "timestamp_utc",
        "send_method",
        "pricing_policy",
        "tx_variant",
        "cu_limit",
        "cu_price_micro_lamports",
        "priority_fee_lamports",
        "fee_cap_lamports",
        "recent_fee_min",
        "recent_fee_median",
        "recent_fee_p90",
        "recent_fee_max",
        "local_recent_fee_min",
        "local_recent_fee_median",
        "local_recent_fee_p90",
        "local_recent_fee_max",
        "recent_tps",
        "recent_slots_per_second",
        "message_size_bytes",
        "instruction_count",
        "account_count",
        "writable_account_count",
        "memo_size_bytes",
        "extra_transfer_count",
        "lamports",
        "landed_within_1_slots",
        "landed_within_2_slots",
        "landed_within_3_slots",
        "latency_ms",
        "slot_delta",
        "confirmation_status",
        "rpc_status_has_error",
        "signature",
        "sent_slot",
        "landed_slot",
    ]
    ensure_csv_header(output, fieldnames)

    for idx in range(num_samples):
        sampled_fee_cap = fee_cap_lamports if fee_cap_lamports is not None else choose_fee_cap_lamports(rng)
        sampled_pricing_policy = pricing_policy if pricing_policy is not None else choose_pricing_policy(rng)
        tx_shape = choose_tx_variant(rng)
        cu_limit = choose_cu_limit(rng)

        snapshot = rpc.snapshot_network()
        cu_price = choose_micro_lamports_price(
            rng=rng,
            snapshot=snapshot,
            cu_limit=cu_limit,
            fee_cap_lamports=sampled_fee_cap,
            pricing_policy=sampled_pricing_policy,
        )
        priority_fee = compute_priority_fee_lamports(cu_limit, cu_price)

        latest_blockhash_resp = client.get_latest_blockhash()
        recent_blockhash = latest_blockhash_resp.value.blockhash

        (
            tx,
            message_size_bytes,
            instruction_count,
            account_count,
            writable_count,
            writable_accounts,
            memo_size_bytes,
            extra_transfer_count,
        ) = build_transaction(
            payer=payer,
            recipient=recipient_pubkey,
            recent_blockhash=recent_blockhash,
            cu_limit=cu_limit,
            cu_price_micro_lamports=cu_price,
            lamports=lamports,
            memo_text=tx_shape["memo_text"],
            extra_transfer_count=tx_shape["extra_transfer_count"],
        )

        local_fee_snapshot = snapshot_network_for_accounts(
            client=rpc,
            writable_accounts=writable_accounts,
        )

        sent_at_utc = utc_now_iso()
        sent_slot = rpc.get_slot("processed")
        started_monotonic = time.monotonic()
        send_resp = client.send_transaction(
            tx,
            opts=TxOpts(skip_preflight=False, preflight_commitment="processed"),
        )

        signature = str(send_resp.value) if hasattr(send_resp, "value") else str(send_resp)
        landing = poll_for_landing(
            rpc=rpc,
            signature=signature,
            sent_slot=sent_slot,
            started_monotonic=started_monotonic,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds,
        )

        slot_delta = landing["slot_delta"]
        row = {
            "timestamp_utc": sent_at_utc,
            "send_method": "rpc",
            "pricing_policy": sampled_pricing_policy,
            "tx_variant": tx_shape["tx_variant"],
            "cu_limit": cu_limit,
            "cu_price_micro_lamports": cu_price,
            "priority_fee_lamports": priority_fee,
            "fee_cap_lamports": sampled_fee_cap,
            "recent_fee_min": snapshot.recent_fee_min,
            "recent_fee_median": snapshot.recent_fee_median,
            "recent_fee_p90": snapshot.recent_fee_p90,
            "recent_fee_max": snapshot.recent_fee_max,
            "local_recent_fee_min": local_fee_snapshot["local_recent_fee_min"],
            "local_recent_fee_median": local_fee_snapshot["local_recent_fee_median"],
            "local_recent_fee_p90": local_fee_snapshot["local_recent_fee_p90"],
            "local_recent_fee_max": local_fee_snapshot["local_recent_fee_max"],
            "recent_tps": snapshot.recent_tps,
            "recent_slots_per_second": snapshot.recent_slots_per_second,
            "message_size_bytes": message_size_bytes,
            "instruction_count": instruction_count,
            "account_count": account_count,
            "writable_account_count": writable_count,
            "memo_size_bytes": memo_size_bytes,
            "extra_transfer_count": extra_transfer_count,
            "lamports": lamports,
            "landed_within_1_slots": int(slot_delta is not None and slot_delta <= 1),
            "landed_within_2_slots": int(slot_delta is not None and slot_delta <= 2),
            "landed_within_3_slots": int(slot_delta is not None and slot_delta <= 3),
            "latency_ms": landing["latency_ms"],
            "slot_delta": slot_delta,
            "confirmation_status": landing["confirmation_status"],
            "rpc_status_has_error": landing["rpc_status_has_error"],
            "signature": signature,
            "sent_slot": sent_slot,
            "landed_slot": landing["landed_slot"],
        }
        append_row(output, fieldnames, row)

        if print_progress:
            print(
                f"[{idx + 1}/{num_samples}] "
                f"sig={signature} "
                f"policy={sampled_pricing_policy} "
                f"variant={tx_shape['tx_variant']} "
                f"fee_cap={sampled_fee_cap} "
                f"cu_limit={cu_limit} "
                f"cu_price={cu_price} "
                f"priority_fee={priority_fee} "
                f"slot_delta={slot_delta} "
                f"latency_ms={row['latency_ms']:.1f}"
            )

        time.sleep(sleep_seconds)

    return output


def prepare_real_transactions(
    input_path: str | Path = "real_transactions.csv",
    output_path: str | Path = "real_transactions_prepared.csv",
) -> pd.DataFrame:
    """Validate and lightly prepare the collected real dataset."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path, parse_dates=["timestamp_utc"])

    missing = [col for col in REAL_DATA_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "signature" in df.columns:
        df = df.drop_duplicates(subset=["signature"]).copy()

    df = add_calendar_features(df, timestamp_column="timestamp_utc")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    df.to_csv(output_path, index=False)
    return df


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified helper for the COMP432 Solana execution-quality project."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    keypair_parser = subparsers.add_parser("generate-keypair", help="Create a Solana keypair JSON file.")
    keypair_parser.add_argument("--output", type=Path, default=Path.home() / ".config" / "solana" / "id.json")
    keypair_parser.add_argument("--overwrite", action="store_true")

    collect_parser = subparsers.add_parser("collect-real-data", help="Collect real transaction data.")
    collect_parser.add_argument("--rpc-url", required=True)
    collect_parser.add_argument("--keypair", required=True, type=Path)
    collect_parser.add_argument("--recipient", default=None)
    collect_parser.add_argument("--output", default="real_transactions.csv", type=Path)
    collect_parser.add_argument("--num-samples", default=200, type=int)
    collect_parser.add_argument("--lamports", default=1, type=int)
    collect_parser.add_argument("--fee-cap-lamports", default=None, type=int)
    collect_parser.add_argument("--poll-interval-seconds", default=0.50, type=float)
    collect_parser.add_argument("--timeout-seconds", default=45.0, type=float)
    collect_parser.add_argument("--sleep-seconds", default=1.0, type=float)
    collect_parser.add_argument("--random-seed", default=42, type=int)
    collect_parser.add_argument(
        "--pricing-policy",
        default=None,
        choices=["recent_fee", "sample_under_cap", "cap_max"],
    )

    prep_parser = subparsers.add_parser("prepare-real-data", help="Validate and prepare real_transactions.csv.")
    prep_parser.add_argument("--input", default="real_transactions.csv", type=Path)
    prep_parser.add_argument("--output", default="real_transactions_prepared.csv", type=Path)

    args = parser.parse_args()

    if args.command == "generate-keypair":
        public_key = generate_keypair_file(output_path=args.output, overwrite=args.overwrite)
        print(f"Wrote keypair to: {args.output}")
        print(f"Public key: {public_key}")
    elif args.command == "collect-real-data":
        output = collect_real_transactions(
            rpc_url=args.rpc_url,
            keypair_path=args.keypair,
            output=args.output,
            recipient=args.recipient,
            num_samples=args.num_samples,
            lamports=args.lamports,
            fee_cap_lamports=args.fee_cap_lamports,
            poll_interval_seconds=args.poll_interval_seconds,
            timeout_seconds=args.timeout_seconds,
            sleep_seconds=args.sleep_seconds,
            random_seed=args.random_seed,
            pricing_policy=args.pricing_policy,
        )
        print(f"Appended data to: {output}")
    elif args.command == "prepare-real-data":
        df = prepare_real_transactions(input_path=args.input, output_path=args.output)
        print(f"Wrote {args.output} with shape {df.shape}.")


if __name__ == "__main__":
    cli_main()
