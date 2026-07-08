"""Download off-the-shelf ONNX models into ./models.

Usage: python scripts/fetch_models.py
Verifies file names by listing the repo; prints the resolved local paths.
"""
import os
from huggingface_hub import hf_hub_download, list_repo_files

# Acoustic EOU model. Prefer the most accurate CPU variant (see model-card
# benchmarks: v3.2 > v3.1 > v3.0); fall back to whatever .onnx the repo lists.
SMART_TURN_REPO = "pipecat-ai/smart-turn-v3"
SMART_TURN_PREFERENCE = [
    "smart-turn-v3.2-cpu.onnx",
    "smart-turn-v3.1-cpu.onnx",
    "smart-turn-v3.0.onnx",
]


# The lexical (semantic) branch is the homemade French rule engine
# (eou_detector/eou/lexical.py:FrenchSemanticEOU) -- no model to download.


def main():
    os.makedirs("models", exist_ok=True)
    files = [f for f in list_repo_files(SMART_TURN_REPO) if f.endswith(".onnx")]
    target = next((p for p in SMART_TURN_PREFERENCE if p in files),
                  files[0] if files else None)
    if not target:
        print(f"[smart_turn] no .onnx found in {SMART_TURN_REPO}; files={files}")
        return
    path = hf_hub_download(SMART_TURN_REPO, target, local_dir="models")
    print(f"[smart_turn] {SMART_TURN_REPO}/{target} -> {path}")


if __name__ == "__main__":
    main()
