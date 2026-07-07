#!/root/autodl-tmp/traffic_accident_rnd/.venv_mmaction/bin/python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import ensure_dir
from eval_predictions import compute_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--step', type=float, default=0.01)
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.predictions).open(newline='', encoding='utf-8')))
    thresholds = []
    t = 0.0
    while t <= 1.000001:
        thresholds.append(round(t, 4))
        t += args.step
    metrics = [compute_metrics(rows, threshold=t) for t in thresholds]
    out = ensure_dir(args.output_dir)
    csv_path = out / 'threshold_sweep.csv'
    fields = ['threshold', 'accuracy', 'accident_precision', 'accident_recall', 'accident_f1', 'false_positive_rate', 'macro_f1']
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in metrics:
            writer.writerow({k: item.get(k) for k in fields})

    high_recall = [m for m in metrics if m['accident_recall'] >= 0.9]
    low_false_alarm = [m for m in metrics if m['false_positive_rate'] <= 0.1]
    best_high_recall = max(high_recall, key=lambda m: (m['accident_precision'], m['macro_f1'], -m['false_positive_rate']), default=None)
    best_low_false_alarm = max(low_false_alarm, key=lambda m: (m['accident_recall'], m['accident_precision'], m['macro_f1']), default=None)
    summary = {
        'predictions': args.predictions,
        'num_samples': len(rows),
        'step': args.step,
        'best_by_macro_f1': max(metrics, key=lambda m: m['macro_f1']) if metrics else None,
        'high_recall_operating_point': best_high_recall,
        'low_false_alarm_operating_point': best_low_false_alarm,
        'threshold_sweep_csv': str(csv_path),
    }
    (out / 'threshold_sweep_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# Threshold Sweep', '', f"- Predictions: `{args.predictions}`", f"- Samples: `{len(rows)}`", '']
    for key in ['best_by_macro_f1', 'high_recall_operating_point', 'low_false_alarm_operating_point']:
        item = summary.get(key)
        if item:
            lines.append(f"- {key}: threshold `{item['threshold']}`, accident recall `{item['accident_recall']:.3f}`, accident precision `{item['accident_precision']:.3f}`, FPR `{item['false_positive_rate']:.3f}`")
        else:
            lines.append(f'- {key}: not available')
    lines.append('')
    (out / 'threshold_sweep.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
