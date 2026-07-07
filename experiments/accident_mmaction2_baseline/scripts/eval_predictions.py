#!/root/autodl-tmp/traffic_accident_rnd/.venv_mmaction/bin/python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from common import ensure_dir


def compute_metrics(rows: list[dict[str, str]], threshold: float | None = None) -> dict:
    y_true = np.array([int(r['label_id']) for r in rows], dtype=int)
    score = np.array([float(r.get('prob_accident') or 0.0) for r in rows], dtype=float)
    if threshold is None:
        y_pred = np.array([int(r['pred_label']) for r in rows], dtype=int)
    else:
        y_pred = (score >= threshold).astype(int)
    labels = [0, 1]
    cm = np.zeros((2, 2), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        if truth in labels and pred in labels:
            cm[truth, pred] += 1
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    metrics = {
        'threshold': threshold,
        'accuracy': float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        'macro_precision': float(np.mean(precision)) if len(precision) else 0.0,
        'macro_recall': float(np.mean(recall)) if len(recall) else 0.0,
        'macro_f1': float(np.mean(f1)) if len(f1) else 0.0,
        'accident_precision': float(precision[1]),
        'accident_recall': float(recall[1]),
        'accident_f1': float(f1[1]),
        'false_positive_rate': float(fp / (fp + tn)) if fp + tn else 0.0,
        'false_negative_rate': float(fn / (fn + tp)) if fn + tp else 0.0,
        'confusion_matrix': cm.tolist(),
        'labels': labels,
        'per_class': {
            '0': {'name': 'non_accident', 'precision': float(precision[0]), 'recall': float(recall[0]), 'f1': float(f1[0]), 'support': int(support[0])},
            '1': {'name': 'accident', 'precision': float(precision[1]), 'recall': float(recall[1]), 'f1': float(f1[1]), 'support': int(support[1])},
        },
        'avg_latency_ms': float(np.mean([float(r.get('latency_ms') or 0.0) for r in rows])) if rows else 0.0,
    }
    if len(set(y_true.tolist())) == 2:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, score))
        except ValueError:
            metrics['roc_auc'] = None
        try:
            metrics['pr_auc'] = float(average_precision_score(y_true, score))
        except ValueError:
            metrics['pr_auc'] = None
    else:
        metrics['roc_auc'] = None
        metrics['pr_auc'] = None
    return metrics


def write_confusion_matrix(path: Path, cm: list[list[int]]) -> None:
    img = np.full((260, 300, 3), 255, np.uint8)
    cv2.putText(img, 'Confusion Matrix', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, 'Pred 0', (95, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(img, 'Pred 1', (180, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(img, 'True 0', (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(img, 'True 1', (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    arr = np.array(cm)
    for i in range(2):
        for j in range(2):
            x0, y0 = 80 + j * 85, 75 + i * 75
            cv2.rectangle(img, (x0, y0), (x0 + 80, y0 + 70), (180, 180, 180), 1)
            cv2.putText(img, str(int(arr[i, j])), (x0 + 28, y0 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.imwrite(str(path), img)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--threshold', type=float)
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.predictions).open(newline='', encoding='utf-8')))
    metrics = compute_metrics(rows, args.threshold)
    out = ensure_dir(args.output_dir)
    (out / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key in ['threshold', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'accident_precision', 'accident_recall', 'accident_f1', 'false_positive_rate', 'false_negative_rate', 'roc_auc', 'pr_auc', 'avg_latency_ms']:
            writer.writerow([key, metrics.get(key)])
    write_confusion_matrix(out / 'confusion_matrix.png', metrics['confusion_matrix'])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
