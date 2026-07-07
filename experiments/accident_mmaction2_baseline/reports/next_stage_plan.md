# Next Stage Plan

1. Add real CCTV hard negative data: congestion, temporary stops, bus stops, queue growth, night/rain glare, occlusion, construction, breakdowns.
2. Replace IoU fallback with StrongSORT/BoxMOT or the user's production tracker while preserving Track JSONL.
3. Train a three-class model only after hard negatives are labeled: normal / hard_negative / accident.
4. Calibrate thresholds by camera and alarm budget; current high-recall threshold 0.56 still has high false positive rate on weak negatives.
5. Add ROI filtering and track-level vehicle evidence selection so rendered boxes focus on likely conflict vehicles rather than all dense traffic boxes.
