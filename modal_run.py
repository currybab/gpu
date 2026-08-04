"""Modal 위에서 이 저장소의 커널 스크립트를 실행하는 러너.

사용 예:
    uv run modal run modal_run.py                                   # 기본: fused_softmax, H100
    uv run modal run modal_run.py --gpu B200                        # GPU 변경
    uv run modal run modal_run.py --script pmpp_v2/vectoradd_py/submission_triton.py

로컬 5090과 비교하려면 --gpu B200(같은 Blackwell 세대)이 가장 가깝다.
"""

import os
import pathlib

import modal

REPO = pathlib.Path(__file__).parent
REMOTE_REPO = "/root/repo"
REMOTE_OUT = "/root/out"

# nvcc가 들어있는 devel 이미지. torch의 load_inline(CUDA C++ 인라인 컴파일)까지 쓰려면 필요하다.
# CUDA 13.0.0 = Modal 호스트 CUDA(13.0, driver 580.95.05)와 정확히 일치 + torch 2.13 기본 빌드.
# Modal 권고: 컨테이너 CUDA는 호스트 CUDA를 넘지 않아야 함.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.13"
    )
    .pip_install(
        "torch==2.13.0",
        "triton==3.7.1",
        "matplotlib",
        "pandas",
    )
    .env({"MPLBACKEND": "Agg", "OUT_DIR": REMOTE_OUT})  # 헤드리스라 GUI 백엔드 사용 불가
    .add_local_dir(
        REPO,
        remote_path=REMOTE_REPO,
        ignore=["**/.git/**", "**/.venv/**", "**/__pycache__/**", "modal_out/**"],
    )
)

app = modal.App("gpu-lab", image=image)


@app.function(timeout=60 * 30)
def run_script(rel_path: str) -> dict[str, bytes]:
    """저장소 안의 스크립트 하나를 __main__ 으로 실행하고, 생성된 산출물을 돌려준다."""
    import runpy
    import subprocess
    import sys

    import torch
    import triton

    print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print(f"Torch CUDA: {torch.version.cuda}")

    script = pathlib.Path(REMOTE_REPO) / rel_path
    os.makedirs(REMOTE_OUT, exist_ok=True)
    # 스크립트가 같은 디렉토리의 모듈을 import 할 수 있게(예: task.py) 경로를 맞춘다.
    sys.path.insert(0, str(script.parent))
    os.chdir(REMOTE_OUT)

    runpy.run_path(str(script), run_name="__main__")

    return {
        p.name: p.read_bytes()
        for p in pathlib.Path(REMOTE_OUT).iterdir()
        if p.is_file()
    }


@app.local_entrypoint()
def main(
    script: str = "trition_tutorial/fused_softmax/fused_softmax.py",
    gpu: str = "B200",
):
    artifacts = run_script.with_options(gpu=gpu).remote(script)

    if artifacts:
        out_dir = REPO / "modal_out" / gpu
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, blob in artifacts.items():
            (out_dir / name).write_bytes(blob)
            print(f"saved: {out_dir / name}")
