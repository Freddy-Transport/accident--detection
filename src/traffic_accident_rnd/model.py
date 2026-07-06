"""TorchVision R3D-18 baseline for accident/non-accident video clips."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models.video import R3D_18_Weights, r3d_18

PROJECT_ROOT = Path("/root/autodl-tmp/traffic_accident_rnd")


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def build_r3d18(*, num_classes: int = 2, pretrained: bool = False, pretrained_dir: str | Path | None = None) -> nn.Module:
    if pretrained:
        torch_home = Path(pretrained_dir or PROJECT_ROOT / "models" / "pretrained")
        torch_home.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(torch_home)
        weights = R3D_18_Weights.KINETICS400_V1
    else:
        weights = None
    model = r3d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def save_checkpoint(model: nn.Module, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata or {}}, output)


def load_checkpoint(model: nn.Module, path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict)
    return payload if isinstance(payload, dict) else {"model_state_dict": state_dict, "metadata": {}}


def run_smoke_training(
    output_path: str | Path,
    *,
    device: str = "auto",
    pretrained: bool = False,
    num_frames: int = 16,
    size: int = 112,
) -> dict[str, Any]:
    selected = select_device(device)
    model = build_r3d18(pretrained=pretrained).to(selected)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    inputs = torch.rand(1, 3, num_frames, size, size, device=selected)
    labels = torch.tensor([1], dtype=torch.long, device=selected)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    metadata = {
        "mode": "smoke",
        "architecture": "r3d_18",
        "device": str(selected),
        "pretrained": pretrained,
        "loss": float(loss.detach().cpu()),
        "num_frames": num_frames,
        "size": size,
    }
    save_checkpoint(model.cpu(), output_path, metadata=metadata)
    return metadata
