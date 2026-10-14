"""Loading run outputs from the To-Call-or-Not-to-Call harness."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_ground_truth(bfcl_csv: str | Path) -> dict[str, str]:
    df = pd.read_csv(bfcl_csv)
    return dict(zip(df["question"], df["correct_response"]))


def question_of(row: dict) -> str:
    return row.get("query") or row.get("question") or row.get("entity") or ""
