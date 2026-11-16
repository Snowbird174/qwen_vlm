"""Lab-bench equipment completeness checking with Qwen3-VL-2B."""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("QWEN_VLM_ROOT", Path(__file__).resolve().parents[2]))
CONTRACTS_DIR = PROJECT_ROOT / "contracts"
DATA_DIR = PROJECT_ROOT / "data"
