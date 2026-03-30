"""
Shared utilities for model inference scripts.
Both test_model_request.py and eval_all_studies.py import from here so they
always use identical image preprocessing, request logic, and prompt.
"""

import base64
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image


MAX_SIDE = 1120  # Llama 3.2 Vision max tile grid = 2×2 × 560 px

DEFAULT_HOST  = "localhost"
DEFAULT_PORT  = 40035
DEFAULT_MODEL = "default"

# Single prompt used by both scripts.
# Asks for step-by-step reasoning so responses are interpretable, then
# requires the structured JSON block at the end so eval_all_studies.py
# can parse it without a second request.
PROMPT = """\
You are a highly experienced chest radiologist AI.
Examine this chest X-ray carefully.

Step 1 – Observations: Describe the key radiological findings (lung fields, \
opacities, consolidations, nodules, cavities, pleural abnormalities, heart size, etc.).

Step 2 – Assessment: Explain whether the findings are consistent with active \
tuberculosis (TB), active silicosis, or another diagnosis.

Step 3 – Report: Output a JSON block with this exact schema and nothing else after it:
```json
{
  "radiologist_conclusion": "<concise one-sentence clinical conclusion>",
  "positive_findings": [
    {
      "condition": "<e.g. Tuberculosis, Silicosis, Pneumonia>",
      "status": "<active | previous | chronic>",
      "icd10": "<ICD-10 code>",
      "snomed_ct": "<SNOMED CT code>",
      "laterality": "<bilateral | left | right | N/A>",
      "location": "<anatomical location>",
      "descriptors": ["<radiological descriptor>"],
      "certainty": "<definite | probable | possible>"
    }
  ],
  "incidental_findings": [
    {
      "condition": "<condition name>",
      "status": "<active | chronic>",
      "icd10": "<ICD-10 code>",
      "snomed_ct": "<SNOMED CT code>",
      "descriptors": ["<descriptor>"],
      "certainty": "<definite | probable | possible>"
    }
  ]
}
```
Return [] for either array if nothing applies.\
"""


def image_to_data_url(path: Path) -> str:
    """Resize image to ≤ MAX_SIDE px and return a base64 JPEG data URL."""
    img = Image.open(path).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    """Send a JSON POST request and return the parsed response."""
    body = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace").strip()
        msg = f"HTTP Error {exc.code}: {exc.reason}"
        if detail:
            msg = f"{msg}\n{detail}"
        raise RuntimeError(msg) from exc


def build_payload(image_path: Path, model: str) -> dict:
    """Build the chat-completions payload for a given image file."""
    image_content = {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}}
    return {
        "model": model,
        "messages": [{"role": "user", "content": [
            image_content,
            {"type": "text", "text": PROMPT},
        ]}],
        "max_tokens":  1024,
        "temperature": 0.0,
    }
