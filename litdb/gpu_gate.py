"""Is the GPU wanted by someone else? Bulk embedding yields to everything.

Why embed through Ollama instead of in-process torch: Ollama owns VRAM
scheduling, so our 0.3 GB nomic model is evicted/loaded alongside the other
apps' vision/chat models without OOM or CPU-offload fights. In-process
torch would pin ~1.5 GB that Ollama can't see.

busy() -> reason string, or "" when the GPU is free for us:
  - PAUSE file present (manual pause: `touch /mnt/nvme3/litdb/PAUSE`)
  - Ollama has any model loaded other than the embedder. Ollama keeps a model
    5 min after its last request, which gives natural hysteresis: we resume
    only after the other job has been quiet for ~5 min.
  - ComfyUI has a running or queued prompt
  - any other GPU compute process (e.g. an upscaler, KoboldCpp)
"""
import json
import os
import subprocess
import urllib.request

from common import EMB_MODEL, OLLAMA

PAUSE_FILE = os.environ.get("PAUSE_FILE", "/mnt/nvme3/litdb/PAUSE")
COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
# GPU processes that are always resident and fine to share with.
ALLOWED_PROCS = ("venvs/comfyui/bin/python", "ollama")


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def busy():
    if os.path.exists(PAUSE_FILE):
        return "PAUSE file"
    try:
        others = [m["name"] for m in _get(OLLAMA + "/api/ps").get("models", [])
                  if not m["name"].startswith(EMB_MODEL)]
        if others:
            return "ollama:" + ",".join(others)
    except OSError as e:
        return f"ollama unreachable ({e})"
    try:
        q = _get(COMFY + "/queue")
        if q.get("queue_running") or q.get("queue_pending"):
            return "comfyui queue"
    except OSError:
        pass  # ComfyUI down = not using the GPU
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=process_name",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=10).stdout
        foreign = [p.strip() for p in out.splitlines()
                   if p.strip() and not any(a in p for a in ALLOWED_PROCS)]
        if foreign:
            return "gpu proc:" + ",".join(foreign)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


if __name__ == "__main__":
    print(busy() or "free")
