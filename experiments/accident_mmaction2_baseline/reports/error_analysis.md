# Error Analysis

Visualizations are stored under:

- VideoMAE: `outputs/20260707_1208_videomae_inference_demo/visualizations/error_cases/`
- SlowFast: `outputs/20260707_1210_slowfast_inference_demo/visualizations/error_cases/`
- X3D: `outputs/20260707_1212_x3d_inference_demo/visualizations/error_cases/`

Observed smoke behavior:

- VideoMAE: 20 false negatives and 20 true negatives; all accident clips missed at threshold 0.5.
- SlowFast: 3 true positives, 8 false positives, 17 false negatives, 12 true negatives.
- X3D: 20 true positives and 20 false positives; all clips predicted accident.

Likely causes: random-init 1 epoch smoke, weak pre-event negatives, short mechanical clip windows, and no independent hard negatives.

Next analysis should add hard negatives, YOLO/Track overlays, ROI crops, and threshold operating-point analysis.
