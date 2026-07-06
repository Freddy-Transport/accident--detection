import json
from pathlib import Path

import torch

from traffic_accident_rnd.inference import build_prediction_result, summarize_track_file
from traffic_accident_rnd.model import build_r3d18, load_checkpoint, save_checkpoint, select_device


def test_build_r3d18_outputs_two_logits():
    model = build_r3d18(num_classes=2, pretrained=False)
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 3, 4, 64, 64))
    assert tuple(logits.shape) == (1, 2)


def test_checkpoint_roundtrip_preserves_metadata(tmp_path: Path):
    model = build_r3d18(num_classes=2, pretrained=False)
    checkpoint = tmp_path / "baseline.pt"
    save_checkpoint(model, checkpoint, metadata={"source": "unit-test"})
    payload = load_checkpoint(model, checkpoint, map_location="cpu")
    assert payload["metadata"]["source"] == "unit-test"


def test_select_device_falls_back_to_cpu_for_unavailable_cuda():
    assert select_device("cpu").type == "cpu"


def test_summarize_track_file_counts_frames_and_classes(tmp_path: Path):
    track_path = tmp_path / "tracks.jsonl"
    track_path.write_text(
        '{"frame_index":0,"timestamp_sec":0.0,"detections":[{"track_id":"1","class_name":"car","confidence":0.9,"bbox_xyxy":[0,0,10,10]}]}\n'
        '{"frame_index":1,"timestamp_sec":0.1,"detections":[{"track_id":"2","class_name":"truck","confidence":0.8,"bbox_xyxy":[0,0,20,20]}]}\n',
        encoding="utf-8",
    )
    summary = summarize_track_file(track_path)
    assert summary["frame_count"] == 2
    assert summary["detection_count"] == 2
    assert summary["class_counts"] == {"car": 1, "truck": 1}


def test_build_prediction_result_writes_scores_and_output(tmp_path: Path):
    output_path = tmp_path / "prediction.json"
    result = build_prediction_result(
        video_path="data/samples/demo.mp4",
        segments=[{"segment_start_sec": 1.0, "segment_end_sec": 2.0, "peak_time_sec": 1.5, "trigger_score": 2.2, "evidence": {}}],
        scores=[0.73],
        track_summary={"frame_count": 0, "detection_count": 0, "class_counts": {}},
        output_path=output_path,
    )
    assert result["candidate_count"] == 1
    assert result["predictions"][0]["accident_score"] == 0.73
    assert json.loads(output_path.read_text(encoding="utf-8"))["video_path"] == "data/samples/demo.mp4"
