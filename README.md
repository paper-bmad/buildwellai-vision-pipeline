# BuildwellAI Vision Pipeline

Multimodal AI pipeline for architectural drawing analysis and UK Building Regulations compliance. Processes floor plans, elevations, and sections using Gemma 4 via vLLM, extracts building parameters, and optionally runs a full compliance check via the BuildwellAI Compliance API.

## Overview

```
Floor plan / elevation / section
         ↓
  [Gemma 4 via vLLM]
         ↓
  Classify drawing type
  Extract building params
  Identify compliance risks
         ↓
  [Compliance API — optional]
         ↓
  16-domain UK Building Regs report
```

**License:** Apache 2.0 (Gemma 4 — commercial use permitted)

## Quick Start

### 1. Start Gemma 4 vLLM server

Requires: CUDA GPU(s), vLLM ≥ 0.19.0

```bash
./serve_gemma4.sh           # 27B — best accuracy (2× GPU)
./serve_gemma4.sh 12b       # 12B — balanced (1× GPU)
./serve_gemma4.sh e4b       # E4B 4.5B — fast inference (1× GPU)
```

The server starts on `http://localhost:8000` by default. Set `PORT=XXXX` to override.

### 2. Install dependencies

```bash
pip install -r requirements.txt
# or
pip install vllm>=0.19.0 pillow>=10.0 requests>=2.31
```

### 3. Analyse a drawing

```bash
# Vision only — extracts params, identifies compliance risks
python vision_pipeline.py --image floor_plan.png

# Vision + compliance check (requires BuildwellAI Compliance API running)
python vision_pipeline.py \
  --image floor_plan.png \
  --compliance-url http://localhost:3001 \
  --domains fire_safety,structural,ventilation,energy,access
```

### 4. Batch process a directory

```bash
python batch_process.py \
  --input-dir ./drawings \
  --output-dir ./results \
  --compliance-url http://localhost:3001 \
  --workers 4
```

## CLI Reference

### `vision_pipeline.py`

| Flag | Default | Description |
|---|---|---|
| `--image` | *(required)* | Path to drawing image (PNG, JPEG, TIFF, WebP) |
| `--server` | `http://localhost:8000` | vLLM server URL |
| `--output` | stdout | Path to write JSON result |
| `--compliance-url` | *(off)* | BuildwellAI Compliance API URL — triggers full report |
| `--domains` | `fire_safety,structural,ventilation,energy` | Comma-separated domains for compliance check |

### `batch_process.py`

| Flag | Default | Description |
|---|---|---|
| `--input-dir` | *(required)* | Directory of drawing images |
| `--output-dir` | `./results` | Output directory for per-image JSON results |
| `--vllm-server` | `http://localhost:8000` | vLLM server URL |
| `--compliance-url` | *(off)* | Compliance API URL |
| `--workers` | `2` | Parallel workers |

## Output Format

### Vision pipeline result

```json
{
  "image_path": "floor_plan.png",
  "classification": {
    "drawing_type": "floor_plan",
    "confidence": 0.92,
    "scale": "1:100",
    "north_arrow_present": true,
    "grid_lines_present": true,
    "notes": "Residential floor plan, 3 storeys apparent"
  },
  "building_parameters": {
    "buildingUse": "Residential",
    "constructionType": "Masonry",
    "numberOfStoreys": 3,
    "floorAreaM2": 280,
    "occupancyEstimate": 18,
    "hasBasement": false,
    "hasAtrium": false
  },
  "compliance_risks": [
    {
      "regulation": "Doc B §B1",
      "observation": "Travel distance from far bedroom to stair ~20m",
      "risk_level": "low",
      "action": "Verify travel distance does not exceed 18m for single-direction escape"
    }
  ],
  "compliance_report": { ... }
}
```

`constructionType` is always a valid `ConstructionType` enum value (`Masonry`, `Steel Frame`, `Timber Frame`, `Concrete Frame`, `Cross Laminated Timber`) — mapped from the VLM's free-text description via keyword matching.

### Drawing types

`floor_plan` · `elevation` · `section` · `site_plan` · `detail` · `schedule` · `unknown`

## Gemma 4 Model Variants

| Variant | Params | Tensor Parallel | VRAM | Notes |
|---|---|---|---|---|
| `e4b` | 4.5B | 1× | ~12 GB | Fast; good for classification |
| `12b` | 12B | 1× | ~24 GB | Balanced accuracy/speed |
| `27b` | 27B | 2× | ~48 GB | Best accuracy; recommended for compliance |

All variants use `bfloat16` and `--enable-prefix-caching` for repeated image analysis.

## Integration with BuildwellAI Compliance API

The vision pipeline outputs `building_parameters` in the exact format expected by the compliance API. With `--compliance-url` set, it calls `POST /check` automatically after extraction.

To run the compliance server locally:

```bash
cd ../compliance-server
ANTHROPIC_API_KEY=sk-ant-... node server.js
```

See [buildwellai-compliance-server](https://github.com/paper-bmad/buildwellai-compliance-server) for setup.

## Supported Compliance Domains

`fire_safety` · `structural` · `ventilation` · `energy` · `overheating` · `acoustics` · `sap` · `drainage` · `access` · `electrical` · `security` · `site_prep` · `sanitation` · `falling` · `broadband` · `ev_charging`

## Tests

```bash
pip install pytest
python -m pytest test_pipeline.py -v
# 14 tests — no GPU required (mocks vLLM responses)
```

## HPC Deployment (SLURM)

```bash
# Start vLLM server on a GPU node
sbatch --partition=gpu --gres=gpu:2 --ntasks=1 --mem=64G \
  --wrap="./serve_gemma4.sh 27b"

# Run pipeline against the server
python vision_pipeline.py \
  --image floor_plan.png \
  --server http://<gpu-node>:8000 \
  --compliance-url http://<api-node>:3001
```
