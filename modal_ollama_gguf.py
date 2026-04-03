import os
import subprocess
import time
from pathlib import Path
from textwrap import dedent

import modal

APP_NAME = "ollama-gguf-hf-import"
OLLAMA_VERSION = "0.6.5"
OLLAMA_PORT = 11434

# Modal Volume 會掛載到這個路徑
VOLUME_PATH = "/models"
GGUF_DIR = f"{VOLUME_PATH}/gguf"

# Ollama 自己存模型的地方
OLLAMA_MODELS_DIR = "/root/.ollama/models"

MODEL_SPECS = [
    {
        "name": "llama31-abliterated-q4km",
        "repo_id": "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF",
        "filename": "meta-llama-3.1-8b-instruct-abliterated.Q4_K_M.gguf",
        "temperature": "0.7",
        "num_ctx": "8192",
    },
    {
        "name": "deepseek-r1-qwen14b-abliterated-q4km",
        "repo_id": "QuantFactory/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2-GGUF",
        "filename": "DeepSeek-R1-Distill-Qwen-14B-abliterated-v2.Q4_K_M.gguf",
        "temperature": "0.6",
        "num_ctx": "16384",
    },
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "zstd")
    .pip_install(
        "huggingface_hub>=0.30.0",
        "hf-transfer>=0.1.9",
        "openai>=1.30.0",
    )
    .run_commands(
        f"mkdir -p {OLLAMA_MODELS_DIR}",
        f"OLLAMA_VERSION={OLLAMA_VERSION} curl -fsSL https://ollama.com/install.sh | sh",
        "which ollama",
        "ollama --version",
    )
    .env(
        {
            "OLLAMA_HOST": f"0.0.0.0:{OLLAMA_PORT}",
            "OLLAMA_MODELS": OLLAMA_MODELS_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name("ollama-gguf-hf-volume", create_if_missing=True)


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(">>>", " ".join(cmd))
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=True,
    )


def _wait_for_ollama(timeout_s: int = 120) -> None:
    import urllib.request

    deadline = time.time() + timeout_s
    last_error = None

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=5) as resp:
                if resp.status == 200:
                    print("Ollama is ready.")
                    return
        except Exception as e:
            last_error = e
            time.sleep(2)

    raise RuntimeError(f"Ollama server did not become ready in time. Last error: {last_error}")


def _download_gguf(repo_id: str, filename: str, local_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded or reused: {path}")
    return path


def _write_modelfile(model_name: str, gguf_path: str, temperature: str, num_ctx: str) -> str:
    modelfile_dir = Path(GGUF_DIR) / model_name
    modelfile_dir.mkdir(parents=True, exist_ok=True)
    modelfile_path = modelfile_dir / "Modelfile"

    content = dedent(
        f"""
        FROM {gguf_path}
        PARAMETER temperature {temperature}
        PARAMETER num_ctx {num_ctx}
        """
    ).strip() + "\n"

    modelfile_path.write_text(content, encoding="utf-8")
    print(f"Wrote Modelfile: {modelfile_path}")
    print(content)
    return str(modelfile_path)


@app.cls(
    gpu="A100",
    timeout=60 * 60,
    volumes={VOLUME_PATH: volume},
    scaledown_window=10 * 60,
)
class OllamaGGUFServer:
    ollama_process: subprocess.Popen | None = None

    @modal.enter()
    def setup(self):
        os.makedirs(GGUF_DIR, exist_ok=True)
        os.makedirs(OLLAMA_MODELS_DIR, exist_ok=True)

        print("Starting Ollama server...")
        self.ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        _wait_for_ollama()

        changed = False

        for spec in MODEL_SPECS:
            gguf_path = os.path.join(GGUF_DIR, spec["filename"])

            if not os.path.exists(gguf_path):
                print(f"Downloading {spec['repo_id']} / {spec['filename']}")
                _download_gguf(
                    repo_id=spec["repo_id"],
                    filename=spec["filename"],
                    local_dir=GGUF_DIR,
                )
                changed = True
            else:
                print(f"Reusing cached GGUF: {gguf_path}")

            modelfile_path = _write_modelfile(
                model_name=spec["name"],
                gguf_path=gguf_path,
                temperature=spec["temperature"],
                num_ctx=spec["num_ctx"],
            )

            result = _run(["ollama", "create", spec["name"], "-f", modelfile_path])
            print(result.stdout)
            if result.stderr:
                print(result.stderr)

            changed = True

        if changed:
            volume.commit()

        result = _run(["ollama", "list"])
        print(result.stdout)

    @modal.exit()
    def teardown(self):
        if self.ollama_process and self.ollama_process.poll() is None:
            print("Stopping Ollama server...")
            self.ollama_process.terminate()
            try:
                self.ollama_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.ollama_process.kill()
                self.ollama_process.wait()

    @modal.web_server(port=OLLAMA_PORT, startup_timeout=10 * 60)
    def api(self):
        print(f"Ollama API is being served on port {OLLAMA_PORT}")


@app.local_entrypoint()
def main():
    print("Local entrypoint ready.")