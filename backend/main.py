import os
import json
import shutil
import tempfile
import subprocess
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="SHAP Viewer Backend")

# CORS: allow developer to set origins via BACKEND_CORS_ORIGINS (comma-separated), default to allow all
cors_origins = os.environ.get("BACKEND_CORS_ORIGINS", "*")
if cors_origins == "*":
    allow_origins = ["*"]
else:
    allow_origins = [o.strip() for o in cors_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str
    params: Dict[str, Any] = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    # Command template must be provided by env var. Keep backend minimal and flexible.
    cmd_template = os.environ.get("SHAP_PIPELINE_CMD")
    if not cmd_template:
        raise HTTPException(
            status_code=501,
            detail=(
                "SHAP_PIPELINE_CMD not configured. Set environment variable to a shell command template. "
                "Use {dir} and {textfile} placeholders. Example: `export SHAP_PIPELINE_CMD=\"bash /path/to/my_pipeline.sh {dir} {textfile}\"`"
            ),
        )

    tmpdir = tempfile.mkdtemp(prefix="shap_req_")
    textfile = os.path.join(tmpdir, "input.txt")
    with open(textfile, "w", encoding="utf-8") as f:
        f.write(req.text)

    cmd = cmd_template.format(dir=tmpdir, textfile=textfile)
    try:
        # Run the user-provided command; capture output for debugging.
        proc = subprocess.run(
            cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600
        )
    except subprocess.CalledProcessError as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e.stderr.decode()[:1000]}")
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(status_code=504, detail="Pipeline timed out")

    combined_path = os.path.join(tmpdir, "combined_output.json")
    if not os.path.exists(combined_path):
        # Try to find and run the repository parse utility to create combined_output.json
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        parse_script = os.path.join(repo_root, "utils", "parse_txt_output.py")
        if os.path.exists(parse_script):
            try:
                subprocess.run(
                    f"python3 {parse_script} {tmpdir} -o {combined_path}",
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=120,
                )
            except Exception as e:
                shutil.rmtree(tmpdir, ignore_errors=True)
                raise HTTPException(status_code=500, detail=f"Failed to parse outputs: {e}")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise HTTPException(status_code=500, detail="No combined_output.json found and parse utility missing")

    try:
        with open(combined_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to read combined_output.json: {e}")

    # Return whole parsed list to frontend. Caller can pick the first sample.
    keep_tmp = os.environ.get("KEEP_TMP")
    if not keep_tmp:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return data


if __name__ == "__main__":
    # For local dev: uvicorn backend.main:app --reload
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
