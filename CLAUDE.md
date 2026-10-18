# 에이전트 인수인계 — 진화 시뮬레이션

이 파일만 읽으면 작업을 이어갈 수 있게 썼다. 진행 이력과 다음 할 일은 `claude-progress.txt`에 있다. 작업을 마치면 그 파일 끝에 한 단락을 덧붙인다.

## 0. 시작하면 먼저 할 일 (순서대로)

1. `claude-progress.txt`를 읽는다. 지금 어느 단계인지, 다음에 무엇을 할지 적혀 있다.
2. 환경을 만든다. 처음 한 번만 하면 된다.

   ```bash
   uv sync --python 3.12
   ```

   - taichi 1.7.4는 Python 3.13까지만 지원한다.
   - 시스템 Python이 3.14여도 uv가 3.12를 따로 받는다.
3. 테스트를 돌린다. 31개 통과가 정상이고 약 3~5분 걸린다. NVIDIA GPU가 없으면 CUDA 테스트 1개는 skip된다.

   ```bash
   uv run pytest -q
   ```

4. GPU를 확인한다. 다른 프로세스가 GPU를 쓰는 중이면 성능 측정값이 흔들린다(§6).

   ```bash
   nvidia-smi
   ```

5. 설계 문서에서 필요한 절만 읽는다. `docs/design/evolution-sim-arch.md`의 §0(성공 기준), 결정 2(화학·에너지), 단계 표.

## 1. 무엇을 만드는가

- 사용자 요구는 하나다. **환경만 주고 배속을 걸어 진화를 지켜본다.**
- 시작한 뒤에는 개입하지 않는다. 산소 증가, 빙하기, 멸종 같은 사건을 각본(일정)으로 넣지 않는다. 물리·화학 규칙에서 저절로 생겨야 한다.
- 무대는 2D 바다 단면(1024 × 384)이다.
- 심해 열수구 근처의 수소 화학합성 세포 200마리에서 출발한다(B안).
- 진화는 유전체 → 형질, 돌연변이, 자연선택으로만 일어난다. 적합도 함수는 없다.

## 2. 사용자와 일하는 방식 (꼭 지킨다)

- **한국어로 답한다.** 코드와 식별자는 영어로 둔다. 산문 보고서 대신 짧은 표와 불릿을 쓴다.
- **단계별로 진행한다.** 단계가 끝나면 평가 에이전트(서브에이전트)가 냉정하게 **위험 / 보통 / 통과** 중 하나로 판정한다.
  - **통과일 때만** 다음 단계로 간다.
  - 보통이나 위험이면 지적 사항을 고치고 같은 에이전트에게 재평가를 요청한다.
- 코드를 바꾼 뒤에는 `uv run ruff check evo tests scripts`와 `uv run pytest -q`를 돌린다.
- 측정하지 않은 원인은 "추정"이라고 쓴다. 실패와 근소 통과는 그대로 보고한다.
- 커밋과 푸시는 사용자가 요청할 때만 한다. 원격 저장소는 `kasangyong/evolution-sim`(비공개)이다.
- 결정과 측정값은 `docs/design/stageN-results.md`에 기록한다(형식은 기존 파일과 같게).

## 3. 코드 지도

| 경로 | 내용 |
|---|---|
| `evo/config.py` | 모든 계수와 데이터클래스(`WorldConfig`, `ChemConfig`, `ClimateConfig`, `LifeConfig`), 화학종 인덱스 `H R X C O M N D` |
| `evo/world/fields.py` | `World`: 장 `chem[NS, w, d]`(f64), 지형, 유선 함수 `psi`, 수송, 열수구, 빛·온도, `acc` 장부 슬롯 |
| `evo/world/chemistry.py` | 비생물 반응(산화·분해·매몰·탈질). 열 단위 부분합으로 싱크를 장부에 기록한다 |
| `evo/world/climate.py` | 대기 1-박스: 기체 교환, 온실 온도, 결빙·알베도, 풍화, 메탄 광분해, H₂ 탈출 |
| `evo/world/reactions.py` | **반응 화학량의 단일 원천.** 전자 단위 `E_UNITS`, 화학 퍼텐셜 `MU`, `delta_g()` |
| `evo/world/ledger.py` | C·N·전자 수지 검사(S6). 몸 탄소도 포함한다 |
| `evo/life/genome.py` | 유전자 인덱스, 범위, 창시자 값(현재 10개) |
| `evo/life/agents.py` | 개체 풀(131,072 슬롯), free-list, `move`/`metabolize`/`die`/`reproduce` 커널, 원자 `take`/`give` |
| `evo/sim.py` | `Sim.step()`: 한 tick의 실행 순서 |
| `evo/ui/app.py`, `evo/ui/render.py` | Dear PyGui 뷰어(단일 스레드 루프, 레이어 선택, 생물 점 표시) |
| `evo/main.py` | 진입점. `--headless N`, `--arch cpu` |
| `tests/` | 반응 균형·ΔG, 수지, 결정성, 결빙 히스테리시스, free-list, ΔG 수율, CUDA 경합 일관성 |
| `scripts/` | 검증과 측정. 표는 README 참고 |

한 tick의 순서는 다음과 같다.
1. `emit_vents`
2. `transport`
3. `chemistry.step`
4. `gas_exchange`
5. `climate.step`
6. `update_light_temp`
7. `life.step`: `move` → `metabolize` → `die` → `reproduce`

## 4. 반드시 지킬 불변식

1. **원소 수지.** 물질이 바뀌는 곳은 다음 두 가지 중 하나여야 한다.
   - 시스템 안에서 이동만 한다(양쪽이 정확히 같다).
   - `world.acc`의 `SRC_*`나 `SINK_*`에 기록된다.

   새 저장고(장, 몸, 대기 슬롯)를 만들면 `ledger.totals()`에 추가한다. 회귀 기준은 수지 오차 < 1e-9(CPU 테스트)이다.
2. **반응은 `reactions.py`에 먼저 추가한다.** 그러면 `test_reactions.py`가 전자·탄소 균형을 검사한다. 에너지 수율은 `−delta_g()`로 계산하고 하드코딩하지 않는다.
3. **η < 1.** 효율 유전자의 상한은 0.95다. 1이 되면 역반응 쌍에서 공짜 에너지가 생긴다.
4. **화학장은 f64(`CHEM_DTYPE`)다.** f32로 바꾸면 정상상태의 반올림이 누적되어 수지가 깨진다(단계 1에서 실측).
5. **개체와 칸 사이의 물질 교환은 `Agents.take()`/`give()`로만 한다.** `take`는 원자 연산으로 빼고 부족분을 되돌린다. 그래서 칸이 음수가 되거나 물질이 복제되지 않는다.
6. **free-list:** push는 `die` 커널에서만, pop은 `reproduce` 커널에서만 한다. 한 커널에서 섞지 않는다. 슬롯은 이동하지 않으므로 참조는 `(slot, uid)`로 한다.
7. **화학종을 추가할 때** 함께 바꿀 곳:
   - `config.py`의 `NS`, `SPECIES_NAMES`
   - `ledger.SPECIES_KEY`
   - `reactions.E_UNITS`, `reactions.MU`
   - 뷰어 레이어

## 5. Taichi 함정 (전부 실제로 겪었다)

- 커널 안에서 `cfg = self.cfg`나 `w = self.world`처럼 로컬 변수로 받으면 **컴파일 오류**가 난다. `ti.static(dataclass)`도 실패한다. 항상 `self.cfg.x`, `self.world.chem[...]`로 직접 쓴다.
- **같은 주소 하나에 셀마다 원자 연산**을 하면 직렬화되어 10배 이상 느려진다(1.15ms 대 0.1ms). 열이나 블록 단위 부분합을 거쳐 합친다.
- 커널 안의 `x = 0.0`은 **f32**다. f64가 필요하면 `ti.cast(0.0, CHEM_DTYPE)`로 쓴다.
- `python -c "..."`로 커널을 정의하면 소스 검사 오류가 난다. 항상 파일로 실행한다.
- RNG는 런타임 전역이다. 결정성을 비교하려면 실행마다 `ti.init(arch=ti.cpu, cpu_max_num_threads=1, random_seed=0)`로 다시 초기화한다.
- 커널 호출은 비동기다. 시간을 재기 전에 `ti.sync()`를 호출한다.
- `uv run python scripts/x.py`를 쓸 때는 `PYTHONPATH=.`가 필요하다(pytest는 pyproject에 설정되어 있다).

## 6. 환경 함정

- **개발 PC(RTX 4050 Laptop)에서는 외부 Python 3.14 프로세스가 GPU를 간헐적으로 점유한다.** 사용자가 멈출 수 없다고 했다.
  - 벽시계 처리량은 32~455 tick/s로 크게 흔들린다.
  - 그래서 S2 성능은 `bench_life.py --profile`의 **커널 시간 합(평균, 최소)**으로 판정한다.
  - 새 PC(RTX 2060 예정)에서는 경합이 없다면 벽시계로 다시 잰다.
- **세션이 재시작되면 백그라운드 작업이 죽는다.** 1시간이 넘는 작업은 재개 가능한 스크립트로 만들고 분리 실행한다(`scripts/s4_gate.py`가 예시). 로그는 `runs/`에 남는다(git 제외).

  ```bash
  powershell -NoProfile -Command "Start-Process -FilePath 'uv' -ArgumentList 'run','python','scripts/s4_gate.py' -WindowStyle Hidden"
  ```

- Windows 화면 배율이 200%이면 Dear PyGui 창이 두 배로 커진다. 뷰어는 `VIEW_W = 900`(논리 픽셀)에 맞춰 두었다.
- 뷰어 스크린샷을 찍을 때는 DPI 인식(`SetProcessDPIAware`) 후 창 좌표로 캡처한다. 그러지 않으면 잘린다.

## 7. 성공 기준 요약 (설계 §0)

| 기준 | 내용 | 현재 |
|---|---|---|
| S1 | CPU 단일 스레드에서 같은 seed면 같은 상태 | 테스트 통과 |
| S2 | 5만 마리에서 200 tick/s 이상 | 경합 상태 GPU 시간 기준 약 220으로 근소 통과. 단계 3에서 다시 측정 |
| S3 | 24시간 연속 실행 무사고 | 단계 5 |
| S4 | 10 seed × 10만 tick 중 8개 이상 존속 | 20/20 (2회 반복) |
| S4' | 100만 tick 3회 중 2회 이상 존속 | 단계 6 |
| S5 | 화학합성이 아닌 에너지 전략이 각본 없이 출현 | 단계 3 이후 |
| S6 | 원소 수지 오차 < 0.1% | 약 1e-11 |

## 8. 다음 작업: 단계 2 재평가 → 단계 3

1. **단계 2 재평가**
   - 평가 에이전트(general-purpose)에게 다음을 넘긴다: `docs/design/stage2-results.md`, `evo/life/`, `tests/test_life.py`, 그리고 1차 평가의 지적 9개가 반영됐는지 검증해 달라는 요청.
   - 1차 지적 9개: S4 비결정성, S2 미검증, CUDA 경합 테스트, ΔG 수율 테스트, η 상한, 음수 표현, 분열 문턱 근거, 설계 대비 차이, 스크립트 정리.
   - 판정은 위험/보통/통과 중 하나로 받는다.
2. **단계 3 범위**(설계 단계 표 3행): 광합성, 호기 호흡, 황 산화·탄소고정, 산소 독성, 포식, 능동 이동, 신경망 뇌.
   - 유전체를 10에서 128로 늘린다. 형질 24개 + 뇌 8 → 8 → 4.
   - 공간 해시를 추가한다. 카운팅 정렬 + `ti.algorithms.PrefixSumExecutor`(CUDA). 포식과 감각에 필요하다.
   - 반응은 `reactions.py`에 이미 정의되어 있다: `photosynthesis`, `aerobic_respiration`, `sulfur_fixation`, `sulfur_oxidation`, `methane_oxidation`.
   - **O₂ 축적 메커니즘 테스트**: 테스트에서만 광합성 개체군을 인위로 심고, 매몰 → O₂ 축적 경로가 동작하는지 확인한다. 실제 실행에는 쓰지 않는다.
   - 통과 기준: 형질 분포가 시간에 따라 이동하고(선택이 작동한다는 증거), O₂ 메커니즘 테스트가 통과하고, S6·S4가 유지되고, S2를 다시 측정한다.
3. **성능을 먼저 확보한다.** S2 여유가 10%뿐이다. 수송 적용 패스(`transport` 두 번째 루프)를 화학 커널에 합치면 약 0.4ms/tick이 줄어든다. 비용의 약 70%는 f64 환경 커널이다.

## 9. 검증 명령

```bash
uv run pytest -q
```

```bash
uv run ruff check evo tests scripts
```

```bash
PYTHONPATH=. uv run python scripts/stage1_check.py --seeded
```

```bash
PYTHONPATH=. uv run python scripts/survival_run.py 3 20000
```

```bash
PYTHONPATH=. uv run python scripts/bench_life.py 3000 world.vent_h=2.0 life.capacity=50000 --profile
```

```bash
uv run python -m evo.main
```
