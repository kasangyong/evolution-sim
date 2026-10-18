# 진화 시뮬레이션 (Evolution Sim)

지구를 닮은 환경만 주고 배속으로 지켜보는 인공생명 시뮬레이션이다.

- 무대는 2D 바다 단면이다.
- 심해 열수구의 최소 세포에서 출발한다.
- 각본은 없다. 산소 증가, 빙하기, 대멸종 같은 사건은 물리·화학 규칙에서 저절로 생겨야 한다.

- 설계: [docs/design/evolution-sim-arch.md](docs/design/evolution-sim-arch.md)
- 단계별 결과: [stage1-results.md](docs/design/stage1-results.md), [stage2-results.md](docs/design/stage2-results.md)
- 진행 상황과 다음 할 일: [claude-progress.txt](claude-progress.txt)

## 요구 사항

- NVIDIA GPU. CUDA 백엔드를 쓴다. RTX 4050 Laptop에서 개발했고, RTX 20xx 이상이면 동작할 것으로 본다.
- 최신 NVIDIA 드라이버.
- [uv](https://docs.astral.sh/uv/). uv가 Python 3.12를 알아서 받는다. taichi 1.7.4는 Python 3.13까지만 지원한다.

## 설치

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
git clone https://github.com/kasangyong/evolution-sim.git
```

```bash
cd evolution-sim && uv sync --python 3.12
```

## 실행

뷰어(배속, 레이어 선택, 생물 표시):

```bash
uv run python -m evo.main
```

UI 없이 N tick 실행하고 수지 오차와 개체 수를 출력:

```bash
uv run python -m evo.main --headless 20000
```

테스트(CPU 단일 스레드 결정성 테스트, CUDA 경합 일관성 검사 포함):

```bash
uv run pytest -q
```

## 검증 스크립트 (`PYTHONPATH=.` 필요)

| 스크립트 | 용도 |
|---|---|
| `scripts/stage1_check.py [--seeded]` | 생물 없는 세계로 10만 tick 수지·발산 검사 |
| `scripts/survival_run.py <seed> [ticks]` | 생존 1회 실행(S4) |
| `scripts/s4_gate.py` | 10 seed × 2회 생존 게이트. `runs/s4_gate.log`에 기록하고, 중단돼도 이어서 돈다 |
| `scripts/bench_life.py <warmup> [key=val ...] [--profile]` | 성능 측정(S2). 예: `world.vent_h=2.0 life.capacity=50000` |
| `scripts/check_cuda_consistency.py` | GPU 원자 연산 경합 상태에서 개체 풀과 수지 불변식 검사 |
