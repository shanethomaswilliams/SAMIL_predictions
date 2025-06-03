import csv, json, os, pathlib
from typing import Dict, Any, List
import numpy as np

# Default path to save prediction information
_CSV_PATH  = "./results/predictions.csv"
_JSON_PATH = "./results/predictions.json"

def _numpy_to_py(obj: Any):
    """
    Converts general objects to json safe datatypes for easy storage and saving
    """
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    try:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, (list, tuple)):
        return [_numpy_to_py(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _numpy_to_py(v) for k, v in obj.items()}
    return obj          


def save_prediction(study_id: str, pred: Dict[str, Any],
                    csv_path: str = _CSV_PATH,
                    json_path: str = _JSON_PATH) -> None:
    """
    Given the specific study id and predictions saves information to both json and csv format. 
    Default paths for CSV and JSON are provided but these can be manually inputted as well
    """

    # Uploading the values to json
    json_data: Dict[str, Any] = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as jf:
            json_data = json.load(jf)
    json_data[study_id] = _numpy_to_py(pred)

    with open(json_path, "w") as jf:
        json.dump(json_data, jf, indent=2)

    # Creating row for csv writing
    row: Dict[str, Any] = {
        "study_id":        study_id,
        "raw_pred_class":  pred["raw_pred_class"],
        "ema_pred_class":  pred["ema_pred_class"],
        **{f"raw_p{i}": v for i, v in
           enumerate(_numpy_to_py(pred["raw_probs"]))},
        **{f"ema_p{i}": v for i, v in
           enumerate(_numpy_to_py(pred["ema_probs"]))},
    }

    rows: List[Dict[str, Any]] = []
    header: List[str]          = []

    if os.path.exists(csv_path):
        with open(csv_path, newline="") as cf:
            reader = csv.DictReader(cf)
            header = reader.fieldnames or []
            rows   = [r for r in reader if r["study_id"] != study_id]

    header = list(dict.fromkeys(header + list(row.keys())))
    rows.append(row)

    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
