"""
Mock Dataset API – SotaRad development server
==============================================

Serves the 10 labelled chest X-ray studies from data/results.json and the
accompanying PNG images so validator.py can be tested without the real dataset
API.

Endpoints
---------
  GET /studies?after=YYYY-MM-DD&before=YYYY-MM-DD&limit=N
      Returns all studies (ignores date params – data/results.json is used
      as-is, but acquisition_date is overridden to tomorrow so every study
      passes the validator's temporal filter when running with --mock).

  GET /images/{filename}
      Streams the PNG from the data/ directory.

Usage
-----
  python mock/dataset_api.py [--port 8100] [--data-dir path/to/data]

  Then run the validator with:
    python validator.py --mock ...
"""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse


def _build_app(data_dir: Path) -> FastAPI:
    app = FastAPI(title="SotaRad Mock Dataset API", version="1.0-mock")

    studies: list[dict] = []

    @app.on_event("startup")
    def _load():
        results_path = data_dir / "results.json"
        if not results_path.exists():
            raise RuntimeError(f"results.json not found at {results_path}")
        raw = json.loads(results_path.read_text())
        studies.extend(raw.get("studies", []))
        print(f"[mock/dataset_api] Loaded {len(studies)} studies from {results_path}")

    # Override acquisition_date to tomorrow so studies always pass the
    # validator's per-miner temporal filter when eval_delay_minutes=0.
    TOMORROW = (date.today() + timedelta(days=1)).isoformat()

    @app.get("/studies")
    def get_studies(
        after:  str | None = Query(default=None),
        before: str | None = Query(default=None),
        limit:  int        = Query(default=100, ge=1, le=1000),
    ):
        out = []
        for study in studies[:limit]:
            out.append({**study, "acquisition_date": TOMORROW})
        print(
            f"[mock/dataset_api] GET /studies  "
            f"after={after} before={before} limit={limit}  → {len(out)} studies"
        )
        return {
            "api_version": "1.0-mock",
            "cohort_id":   "mock-cohort",
            "studies":     out,
        }

    @app.get("/images/{filename}")
    def get_image(filename: str):
        path = data_dir / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
        print(f"[mock/dataset_api] GET /images/{filename}")
        return FileResponse(str(path), media_type="image/png")

    @app.get("/health")
    def health():
        return {"status": "ok", "studies_loaded": len(studies)}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="SotaRad mock dataset API server")
    parser.add_argument(
        "--port", type=int, default=8100,
        help="Port to listen on (default: 8100)",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).parent.parent / "data"),
        help="Path to the directory containing results.json and PNG images",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    print(f"[mock/dataset_api] Starting on port {args.port}  data_dir={data_dir}")

    app = _build_app(data_dir)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
