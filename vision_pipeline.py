"""
BuildwellAI Vision Pipeline — Architectural Drawing Analysis
Uses a multimodal LLM (Gemma 4 or Skywork R1V2-38B) via vLLM for floor plan understanding.

Requirements:
  pip install vllm pillow requests
  vLLM v0.19.0+ (Gemma 4) or vLLM ≥ latest (Skywork R1V2 with trust_remote_code)

Model selection:
  - Gemma 4 (default): general multimodal, Apache 2.0
  - Skywork R1V2-38B: stronger on documents/charts/STEM diagrams; MIT licensed;
    single-GPU AWQ deployment. Better fit for architectural drawing analysis.

Usage:
  python vision_pipeline.py --image floor_plan.png [--server http://localhost:8000]
  python vision_pipeline.py --image floor_plan.png --model r1v2  # use Skywork R1V2-38B
"""
import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

import requests


VLLM_BASE_URL = "http://localhost:8000"

# Model registry: short alias → vLLM/HF model id
MODELS = {
    "gemma4-27b":   "google/gemma-4-27b-it",
    "gemma4-12b":   "google/gemma-4-12b-it",
    "gemma4-e4b":   "google/gemma-4-e4b-it",
    "r1v2":         "Skywork/Skywork-R1V2-38B",
    "r1v2-awq":     "Skywork/Skywork-R1V2-38B-AWQ",
}
DEFAULT_MODEL_ALIAS = "gemma4-27b"
GEMMA4_MODEL = MODELS[DEFAULT_MODEL_ALIAS]  # backwards compat for callers

CLASSIFY_PROMPT = """Analyze this architectural drawing and classify it.

Return a JSON object with these fields:
{
  "drawing_type": "<floor_plan|elevation|section|site_plan|detail|schedule|unknown>",
  "confidence": <0.0-1.0>,
  "scale": "<detected scale or null>",
  "north_arrow_present": <true|false>,
  "grid_lines_present": <true|false>,
  "notes": "<brief observation about the drawing>"
}

Return ONLY the JSON object, no explanation."""

EXTRACT_PARAMS_PROMPT = """This is a {drawing_type} architectural drawing. Extract building parameters for UK Building Regulations compliance assessment.

Return a JSON object with these fields (use null for fields you cannot determine):
{
  "estimated_storeys": <integer or null>,
  "estimated_gfa_m2": <float or null>,
  "building_footprint_m2": <float or null>,
  "apparent_use": "<Residential|Commercial|Mixed Use|Industrial|Education|Healthcare|unknown>",
  "construction_clues": "<observed materials, structural elements>",
  "has_basement": <true|false|null>,
  "has_atrium": <true|false|null>,
  "room_count": <integer or null>,
  "circulation_notes": "<escape routes, stair positions, corridors>",
  "fire_compartment_walls": <true|false|null>,
  "accessible_entrance": <true|false|null>,
  "extraction_confidence": <0.0-1.0>
}

Return ONLY the JSON object, no explanation."""

COMPLIANCE_RISK_PROMPT = """This is a {drawing_type} for a {building_use} building. Identify visible compliance issues or items requiring attention under UK Building Regulations.

Focus on:
- Escape route widths and travel distances (Doc B)
- Stair dimensions visible in plan (Doc K)
- Structural grid and span implications (Doc A)
- Apparent means of ventilation (Doc F)
- Accessibility — door widths, step-free routes (Doc M)

Return a JSON array of findings:
[
  {{
    "regulation": "<Doc X clause>",
    "observation": "<what was observed>",
    "riskLevel": "<low|medium|high>",
    "action": "<recommended action>"
  }}
]

Return ONLY the JSON array, no explanation."""


VALID_CONSTRUCTION_TYPES = [
    "Timber Frame",
    "Masonry",
    "Steel Frame",
    "Concrete Frame",
    "Cross Laminated Timber",
]


def normalize_construction_type(description: str) -> str:
    """Map free-text construction description to a valid ConstructionType enum value."""
    if not description:
        return "Masonry"
    desc = description.lower()
    if "clt" in desc or "cross laminated" in desc or "cross-laminated" in desc:
        return "Cross Laminated Timber"
    if "timber" in desc or "wood" in desc or "lumber" in desc:
        return "Timber Frame"
    if "steel" in desc or "structural steel" in desc or "metal frame" in desc:
        return "Steel Frame"
    if "concrete" in desc or "rc " in desc or "reinforced" in desc or "precast" in desc:
        return "Concrete Frame"
    if "masonry" in desc or "brick" in desc or "block" in desc or "stone" in desc:
        return "Masonry"
    return "Masonry"  # safe default


def call_compliance_api(building_params: dict, compliance_url: str,
                        domains: list[str] | None = None,
                        additional_context: str = "",
                        max_retries: int = 3) -> dict:
    """POST building parameters to the BuildwellAI compliance API and return the report.

    Retries on HTTP 429 (rate limited) up to max_retries times, honouring the
    Retry-After header returned by the compliance server.
    """
    if domains is None:
        domains = ["fire_safety", "structural", "ventilation", "energy"]
    query = {
        "buildingParameters": {
            "buildingUse": building_params["buildingUse"],
            "constructionType": building_params["constructionType"],
            "numberOfStoreys": building_params["numberOfStoreys"],
            "floorAreaM2": building_params["floorAreaM2"],
            "occupancyEstimate": building_params["occupancyEstimate"],
            "hasBasement": building_params["hasBasement"],
            "hasAtrium": building_params["hasAtrium"],
        },
        "domains": domains,
        "additionalContext": additional_context,
    }
    for attempt in range(max_retries):
        resp = requests.post(f"{compliance_url}/check", json=query, timeout=60)
        if resp.status_code == 429 and attempt < max_retries - 1:
            retry_after = int(resp.headers.get("Retry-After", "5"))
            print(f"      Rate limited — retrying in {retry_after}s "
                  f"(attempt {attempt + 1}/{max_retries})", file=sys.stderr)
            time.sleep(retry_after)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return resp.json()  # unreachable, satisfies type checkers


def encode_image_base64(image_path: str) -> str:
    from PIL import Image  # lazy: only needed when processing real images
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(path)
    # Resize if too large (Gemma 4 handles up to ~896px natively)
    max_dim = 1024
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_vllm(model: str, prompt: str, image_b64: str, server: str, max_tokens: int = 512) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    resp = requests.post(f"{server}/v1/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def parse_json_response(raw: str) -> dict | list:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Strip markdown code fences
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw.strip())


def classify_drawing(image_b64: str, server: str, model: str = GEMMA4_MODEL) -> dict:
    raw = call_vllm(model, CLASSIFY_PROMPT, image_b64, server)
    return parse_json_response(raw)


def extract_building_params(image_b64: str, drawing_type: str, server: str, model: str = GEMMA4_MODEL) -> dict:
    prompt = EXTRACT_PARAMS_PROMPT.format(drawing_type=drawing_type)
    raw = call_vllm(model, prompt, image_b64, server, max_tokens=768)
    return parse_json_response(raw)


def identify_compliance_risks(image_b64: str, drawing_type: str, building_use: str, server: str, model: str = GEMMA4_MODEL) -> list:
    prompt = COMPLIANCE_RISK_PROMPT.format(drawing_type=drawing_type, building_use=building_use)
    raw = call_vllm(model, prompt, image_b64, server, max_tokens=1024)
    return parse_json_response(raw)


def run_pipeline(image_path: str, server: str = VLLM_BASE_URL, model: str = GEMMA4_MODEL) -> dict:
    print(f"[1/3] Encoding image: {image_path}")
    image_b64 = encode_image_base64(image_path)

    print(f"[2/3] Classifying drawing type ({model})...")
    classification = classify_drawing(image_b64, server, model)
    drawing_type = classification.get("drawing_type", "unknown")
    print(f"      → {drawing_type} (confidence: {classification.get('confidence', '?')})")

    print(f"[3/3] Extracting building parameters...")
    params = extract_building_params(image_b64, drawing_type, server, model)
    building_use = params.get("apparent_use", "Residential")
    print(f"      → {building_use}, ~{params.get('estimated_storeys', '?')} storeys, "
          f"~{params.get('estimated_gfa_m2', '?')}m²")

    print(f"[+]   Identifying compliance risks...")
    risks = identify_compliance_risks(image_b64, drawing_type, building_use, server, model)
    print(f"      → {len(risks)} potential compliance items found")

    construction_type = normalize_construction_type(params.get("construction_clues") or "")

    result = {
        "image_path": str(image_path),
        "model": model,
        "classification": classification,
        "building_parameters": {
            "buildingUse": building_use,
            "constructionType": construction_type,
            "numberOfStoreys": params.get("estimated_storeys") or 2,
            "floorAreaM2": int(params.get("estimated_gfa_m2") or 200),
            "occupancyEstimate": max(1, int((params.get("estimated_gfa_m2") or 200) / 15)),
            "hasBasement": params.get("has_basement") or False,
            "hasAtrium": params.get("has_atrium") or False,
            "_extractedParams": params,
        },
        "compliance_risks": risks,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="BuildwellAI Vision Pipeline — multimodal drawing analysis")
    parser.add_argument("--image", required=True, help="Path to architectural drawing image")
    parser.add_argument("--server", default=VLLM_BASE_URL, help="vLLM server URL")
    parser.add_argument("--model", default=DEFAULT_MODEL_ALIAS,
                        choices=list(MODELS.keys()) + ["custom"],
                        help=f"Model alias: {', '.join(MODELS.keys())}; or 'custom' to use --model-id")
    parser.add_argument("--model-id", default=None,
                        help="Explicit HF/vLLM model id (overrides --model). Use with --model custom.")
    parser.add_argument("--output", help="Output JSON file (default: stdout)")
    parser.add_argument("--compliance-url", default=None,
                        help="BuildwellAI compliance API URL — triggers a full compliance check after extraction")
    parser.add_argument("--domains", default=None,
                        help="Comma-separated compliance domains (default: fire_safety,structural,ventilation,energy)")
    args = parser.parse_args()

    if args.model == "custom":
        if not args.model_id:
            parser.error("--model custom requires --model-id")
        model = args.model_id
    else:
        model = MODELS[args.model]

    result = run_pipeline(args.image, args.server, model)

    domains = [d.strip() for d in args.domains.split(",")] if args.domains else None

    if args.compliance_url:
        print(f"\n[+] Running compliance check via {args.compliance_url}...")
        bp = result["building_parameters"]
        risks = result.get("compliance_risks", [])
        context_parts = [f"Drawing type: {result['classification']['drawing_type']}"]
        if risks:
            context_parts.append(
                f"Vision analysis flagged {len(risks)} items: " +
                "; ".join(r.get("observation", "") for r in risks[:3])
            )
        try:
            compliance_report = call_compliance_api(
                bp, args.compliance_url, domains, ". ".join(context_parts)
            )
            result["compliance_report"] = compliance_report
            status = compliance_report.get("overallStatus", "?")
            print(f"      → Overall status: {status.upper().replace('_', ' ')}")
        except Exception as e:
            print(f"      ✗ Compliance API error: {e}", file=sys.stderr)
            result["compliance_report"] = {"error": str(e)}

    output = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(output)
        print(f"\nResults written to {args.output}")
    else:
        print("\n=== Vision Pipeline Results ===")
        print(output)

    # Print compliance-ready building parameters for use with the compliance checker
    bp = result["building_parameters"]
    print("\n=== Compliance Checker Parameters ===")
    print(f"Building Use:      {bp['buildingUse']}")
    print(f"Construction Type: {bp['constructionType']}")
    print(f"Storeys:           {bp['numberOfStoreys']}")
    print(f"Floor Area:        {bp['floorAreaM2']}m²")
    print(f"Occupancy:         {bp['occupancyEstimate']}")
    print(f"Basement:          {bp['hasBasement']}")

    if result["compliance_risks"]:
        print(f"\n=== {len(result['compliance_risks'])} Compliance Risk(s) Identified ===")
        for r in result["compliance_risks"]:
            level = r.get("riskLevel", "?").upper()
            print(f"[{level}] {r.get('regulation', '?')}: {r.get('observation', '')}")

    if "compliance_report" in result and "error" not in result["compliance_report"]:
        report = result["compliance_report"]
        print(f"\n=== Compliance Report: {report.get('overallStatus','?').upper().replace('_',' ')} ===")
        for domain in report.get("domains", []):
            print(f"  {domain['label']}: {domain['status'].replace('_',' ')}")


if __name__ == "__main__":
    main()
