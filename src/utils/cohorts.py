"""Select explicit question cohorts without silently reusing explored IDs."""
from pathlib import Path

from .io import load_json


def read_ids(path):
    values = load_json(path)
    ids = values.get("ids") if isinstance(values, dict) else values
    if not isinstance(ids, list) or any(not isinstance(q, str) or not q for q in ids):
        raise ValueError("An ID manifest must contain a list of nonempty question IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs in manifest")
    return ids


def select_cohort(records, dataset, base):
    path = dataset.get("id_manifest")
    exclusions = dataset.get("exclude_ids_manifest")
    if dataset.get("split") == "test" and (not path or not exclusions):
        raise ValueError("Test cohorts require explicit IDs and an explored-ID exclusion manifest")
    if not path:
        return None
    ids = read_ids(Path(base) / path)
    if not ids:
        raise ValueError("The selected cohort is empty")
    excluded = set(read_ids(Path(base) / exclusions)) if exclusions else set()
    if excluded.intersection(ids):
        raise ValueError("Evaluation IDs overlap with the declared explored IDs")
    index = {r["_id"]: r for r in records}
    if len(index) != len(records) or not set(ids) <= index.keys():
        raise ValueError("Dataset has duplicate IDs or is missing requested IDs")
    return [index[q] for q in ids]
