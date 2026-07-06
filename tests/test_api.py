from pathlib import Path

from fastapi.testclient import TestClient

from traffic_accident_rnd.api import app


def test_health_endpoint_reports_project_root():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["project_root"] == "/root/autodl-tmp/traffic_accident_rnd"
    assert "torch_cuda_available" in payload


def test_schema_endpoint_exposes_manifest_and_track_fields():
    client = TestClient(app)
    response = client.get("/schema")
    assert response.status_code == 200
    payload = response.json()
    assert "video_id" in payload["manifest_required_fields"]
    assert "detections" in payload["track_frame_required_fields"]


def test_predict_video_endpoint_delegates_to_pipeline(monkeypatch, tmp_path: Path):
    output_path = tmp_path / "prediction.json"

    def fake_predict_video(**kwargs):
        assert kwargs["video_path"] == "/remote/video.mp4"
        assert kwargs["output_path"] == output_path
        return {"video_path": kwargs["video_path"], "candidate_count": 0, "predictions": [], "track_summary": None}

    monkeypatch.setattr("traffic_accident_rnd.api.predict_video", fake_predict_video)
    client = TestClient(app)
    response = client.post(
        "/predict/video",
        json={"video_path": "/remote/video.mp4", "output_path": str(output_path), "threshold_z": 1.5},
    )
    assert response.status_code == 200
    assert response.json()["video_path"] == "/remote/video.mp4"


def test_predict_tracks_endpoint_returns_summary(monkeypatch):
    def fake_summary(track_path):
        assert str(track_path) == "/remote/tracks.jsonl"
        return {"frame_count": 2, "detection_count": 3, "class_counts": {"car": 3}}

    monkeypatch.setattr("traffic_accident_rnd.api.summarize_track_file", fake_summary)
    client = TestClient(app)
    response = client.post("/predict/tracks", json={"track_path": "/remote/tracks.jsonl"})
    assert response.status_code == 200
    assert response.json()["detection_count"] == 3
