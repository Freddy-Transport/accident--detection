# Error Analysis

Visualization roots:

- VideoMAE-B: `experiments/accident_mmaction2_baseline/outputs/20260707_1343_videomae_pretrained_medium_1epoch_inference/visualizations/error_cases/`
- SlowFast-R50: `experiments/accident_mmaction2_baseline/outputs/20260707_1347_slowfast_pretrained_medium_1epoch_inference/visualizations/error_cases/`
- X3D-S: `experiments/accident_mmaction2_baseline/outputs/20260707_1349_x3d_pretrained_medium_1epoch_inference/visualizations/error_cases/`

Observed one-epoch medium behavior:

- VideoMAE-B: TP=97, FP=74, FN=53, TN=76, accident recall=0.647, FPR=0.493.
- SlowFast-R50: TP=117, FP=111, FN=33, TN=39, accident recall=0.780, FPR=0.740.
- X3D-S: TP=146, FP=143, FN=4, TN=7, accident recall=0.973, FPR=0.953.

Analysis:

- False positives are mostly weak pre-event clips from the same accident videos; this is not equivalent to operational false alarms on normal roads.
- X3D and SlowFast are biased toward accident at default threshold; VideoMAE is less extreme but still has high false positives.
- Current video-only classifier cannot distinguish many pre-event context clips from true accident windows; ROI crops and trajectory evidence should be added next.
- Next data priority is true hard negative CCTV: congestion, queueing, bus stop, temporary parking, construction, occlusion, rain/fog/night reflection, breakdown without collision.
