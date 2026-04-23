"""
BuildwellAI Vision Pipeline — Batch Processor
Processes a directory of architectural drawings and calls the compliance API for each.

Usage:
  python batch_process.py --input-dir ./drawings --compliance-url http://localhost:3001
"""
import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from vision_pipeline import run_pipeline, VLLM_BASE_URL

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}

COMPLIANCE_DOMAINS_DEFAULT = ["fire_safety", "ventilation", "structural", "energy"]


def call_compliance_api(building_params: dict, compliance_url: str, additional_context: str = "") -> dict:
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
        "domains": COMPLIANCE_DOMAINS_DEFAULT,
        "additionalContext": additional_context,
    }
    resp = requests.post(f"{compliance_url}/check", json=query, timeout=60)
    resp.raise_for_status()
    return resp.json()


def process_image(image_path: Path, vllm_server: str, compliance_url: str | None) -> dict:
    try:
        vision_result = run_pipeline(str(image_path), vllm_server)

        result = {
            "image": image_path.name,
            "vision": vision_result,
            "compliance": None,
            "error": None,
        }

        if compliance_url:
            # Build context from vision analysis
            risks = vision_result.get("compliance_risks", [])
            context_parts = [f"Drawing type: {vision_result['classification']['drawing_type']}"]
            if risks:
                context_parts.append(f"Vision analysis found {len(risks)} potential issues: " +
                                     "; ".join(r.get("observation", "") for r in risks[:3]))
            context = ". ".join(context_parts)
            result["compliance"] = call_compliance_api(
                vision_result["building_parameters"], compliance_url, context
            )

        return result
    except Exception as e:
        return {"image": image_path.name, "vision": None, "compliance": None, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="BatchProcess architectural drawings with Gemma 4 + compliance check")
    parser.add_argument("--input-dir", required=True, help="Directory containing drawing images")
    parser.add_argument("--output-dir", default="./results", help="Output directory for JSON results")
    parser.add_argument("--vllm-server", default=VLLM_BASE_URL, help="vLLM server URL")
    parser.add_argument("--compliance-url", default=None, help="Compliance API URL (optional)")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not images:
        print(f"No supported images found in {input_dir}")
        sys.exit(1)

    print(f"Processing {len(images)} images with {args.workers} workers...")
    summary = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_image, img, args.vllm_server, args.compliance_url): img
            for img in images
        }
        for future in as_completed(futures):
            img = futures[future]
            result = future.result()
            output_file = output_dir / f"{img.stem}_result.json"
            output_file.write_text(json.dumps(result, indent=2))

            status = "ERROR" if result["error"] else "OK"
            drawing_type = result["vision"]["classification"]["drawing_type"] if result["vision"] else "?"
            print(f"[{status}] {img.name} → {drawing_type}")
            summary.append({"image": img.name, "status": status, "drawing_type": drawing_type,
                           "error": result["error"]})

    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    ok = sum(1 for s in summary if s["status"] == "OK")
    print(f"\nDone: {ok}/{len(images)} processed successfully. Results in {output_dir}")


if __name__ == "__main__":
    main()
