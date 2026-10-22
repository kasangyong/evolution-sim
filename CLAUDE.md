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
3. 테스트를 돌린다. 146개 통과가 정상이고 약 20분 걸린다(테스트마다 Taichi 런타임을 새로 연다). NVIDIA GPU가 없으면 CUDA 테스트 1개는 skip된다.

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
| `evo/life/genome.py` | 유전자 인덱스, 범위, 창시자 값(현재 330개 = 형질 20 + 뇌 136(12→8→4) + 부착 + 적응 시간 + 몸 설계 12(7b) + 신경 128(7c) + 가소성 32(7d). 새 유전자는 뒤에 덧붙인다. 형질 목록은 `TRAIT_GENES`) |
| `evo/life/organisms.py` | 단계 7b·7c 다세포 생물체: 마디(구슬-막대) 사슬, 조직 분화, 저항력 이론 헤엄(중점 규칙), 발생·생식 세포·조각남·전환, 신경계(발생 부호화 `wire`, CTRNN `neural`) |
| `evo/life/agents.py` | 개체 풀(131,072 슬롯, 슈퍼개체 `ncell`), free-list, 반응 실행 `run`(화학량은 `reactions.py`에서 만든 `INPUTS`/`OUTPUTS`/`BODY` 표만 읽는다), `move`/`census`/`think`(뇌)/`predate`/`metabolize`/`die`/`weigh_divisions`/`reproduce` 커널, 원자 `take`/`give` |
| `evo/sim.py` | `Sim.step()`: 한 tick의 실행 순서 |
| `evo/ui/app.py`, `evo/ui/render.py` | Dear PyGui 뷰어(단일 스레드 루프, 레이어 선택, 생물 점 표시) |
| `evo/ui/web.py`, `evo/ui/web.html` | 실시간 브라우저 뷰어(`--web PORT`). 자연색 화면, 확대 시 세포 단위 그리기, 시간 막대(되감기·다시 실행), 계통수. Taichi는 메인 스레드에만, HTTP 스레드는 PNG·통계·디스크 기록만 다룬다 |
| `evo/ui/render.py` | 자연색 렌더(`draw_natural`), 색소·종류, 확대 영역 세포 추출(uid 해시 고정 표본) |
| `evo/io/snapshot.py`, `evo/io/history.py` | 전체 상태 저장·복원, 시간 기계(프레임 기록·스냅샷·갈래·예산, 쓰기 스레드) |
| `evo/analysis/species.py` | 종 판별(리더-팔로워, 문턱 0.2, 이전 종 중심을 먼저 리더로)과 계통수 |
| `evo/analysis/events.py` | 사건 기록: 2,000 tick마다 광합성 번성·산소 증가·메탄 붕괴·결빙·해빙·대멸종·주요 종 출현과 사라짐·새 능력 확산을 감지하고 그 순간 화면을 `runs/history/<실행>/events/`에 남긴다 |
| `evo/main.py` | 진입점. `--headless N`, `--arch cpu` |
| `tests/` | 반응 균형·ΔG, 수지, 결정성, 결빙 히스테리시스, free-list, 반응 9개 수율, 빛 상한, O₂ 축적(매몰 대조), 슈퍼개체, 저서 부착, CUDA 경합 일관성 |
| `scripts/` | 검증과 측정. 표는 README 참고 |

한 tick의 순서는 다음과 같다.
1. `emit_vents`
2. `transport`
3. `chemistry.step`
4. `gas_exchange`
5. `climate.step`
6. `update_light_temp`
7. `life.step`: `move` → `census` → `think` → `predate` → `metabolize` → `die` → `weigh_divisions` → `reproduce`

## 4. 반드시 지킬 불변식

1. **원소 수지.** 물질이 바뀌는 곳은 다음 두 가지 중 하나여야 한다.
   - 시스템 안에서 이동만 한다(양쪽이 정확히 같다).
   - `world.acc`의 `SRC_*`나 `SINK_*`에 기록된다.

   새 저장고(장, 몸, 대기 슬롯)를 만들면 `ledger.totals()`에 추가한다. 회귀 기준은 수지 오차 < 1e-9(CPU 테스트)이다.
2. **반응은 `reactions.py`에 먼저 추가한다.** 그러면 `test_reactions.py`가 전자·탄소 균형을 검사한다. 에너지 수율은 `−delta_g()`로 계산하고 하드코딩하지 않는다.
3. **η < 1.** 효율 유전자의 상한은 0.95다. 1이 되면 역반응 쌍에서 공짜 에너지가 생긴다.
4. **화학장은 f64(`CHEM_DTYPE`)다.** f32로 바꾸면 정상상태의 반올림이 누적되어 수지가 깨진다(단계 1에서 실측).
5. **개체와 칸 사이의 물질 교환은 `Agents.take()`/`give()`로만 한다.** `take`는 원자 연산으로 빼고 부족분을 되돌린다. 그래서 칸이 음수가 되거나 물질이 복제되지 않는다.
6. **free-list:** 개체 풀은 push를 `die`에서만, pop을 `reproduce`와 `Organisms.develop`(마디 추가·생식 세포)에서만 한다. 생물체 표는 pop을 `transition`, push를 `integrity`에서 한다. 한 커널에서 push와 pop을 섞지 않는다. 슬롯은 이동하지 않으므로 참조는 `(slot, uid)`로 한다(생물체 표도 마디마다 uid를 들고 확인한다).
7. **화학종을 추가할 때** 함께 바꿀 곳:
   - `config.py`의 `NS`, `SPECIES_NAMES`
   - `ledger.SPECIES_KEY`
   - `reactions.E_UNITS`, `reactions.MU`
   - 뷰어 레이어
8. **슈퍼개체:** 슬롯 하나는 같은 유전체의 세포 `ncell`개다. 세포 크기는 `body / ncell`이고 분열 조건·성장 상한은 세포 크기 기준이다. 빈 슬롯이 부족하면 세포 수 비례 추첨으로 나누고, 못 받으면 `ncell × 2`다. 개체 수 통계는 Σ`ncell`이다(`stage3a-arch.md` 결정 7). 단계 7b부터 세포 여럿인 슬롯은 굶을 때 굶는 몫만 죽는다(빚 탕감 근사, `stage7b-results.md` §3).
9. **저서 부착:** 바닥 바로 위 물 칸의 세포는 수동 표류를 하지 않는다. 심해 군집이 풀 포화에서 살아남는 이유다.
10. **군체(단계 6):** 부착 유전자 확률로 딸세포가 묶음 안에 남으면 `colony = 1`이다. 풀 포화로 생긴 묶음(계산상)과 구분한다. 포식은 군체를 전체 크기로 비교한다. 스냅샷 스키마 2.
11. **다세포 생물체(단계 7b):** 마디 하나 = 개체 풀의 슬롯 하나(`org`, `seg`). 몸 탄소는 그대로 `Agents.body`에 있어 원소 장부가 바뀌지 않는다. 생물체 표(`evo/life/organisms.py`)가 마디 슬롯과 uid를 들고 운동(저항력 이론)·순환·발생·조각남·전환을 맡는다. 마디 안 세포는 같은 유전체(체세포 돌연변이 없음), 생식 세포만 돌연변이. 스냅샷 스키마 4(스키마 3은 `upgrade`로 읽는다).

## 5. Taichi 함정 (전부 실제로 겪었다)

- 커널 안에서 `cfg = self.cfg`나 `w = self.world`처럼 로컬 변수로 받으면 **컴파일 오류**가 난다. `ti.static(dataclass)`도 실패한다. 항상 `self.cfg.x`, `self.world.chem[...]`로 직접 쓴다.
- **같은 주소 하나에 셀마다 원자 연산**을 하면 직렬화되어 10배 이상 느려진다(1.15ms 대 0.1ms). 열이나 블록 단위 부분합을 거쳐 합친다.
- 커널 안의 `x = 0.0`은 **f32**다. f64가 필요하면 `ti.cast(0.0, CHEM_DTYPE)`로 쓴다.
- **f64 값과 곱해도 파이썬 실수 리터럴은 f32로 반올림된다.** 예: `x / 6.6`에서 6.6이 f32가 되어 1/6.6의 상대 오차가 1e-8이 된다. `ti.cast(k, f64)`로도 고쳐지지 않는다. `CHEM_DTYPE(k)`로 감싼다. 단계 3a에서 이 때문에 N 수지가 2.6e-9로 깨졌다.
- `python -c "..."`로 커널을 정의하면 소스 검사 오류가 난다. 항상 파일로 실행한다.
- RNG는 런타임 전역이다. 결정성을 비교하려면 실행마다 `ti.init(arch=ti.cpu, cpu_max_num_threads=1, random_seed=0)`로 다시 초기화한다.
- 커널 호출은 비동기다. 시간을 재기 전에 `ti.sync()`를 호출한다.
- `uv run python scripts/x.py`를 쓸 때는 `PYTHONPATH=.`가 필요하다(pytest는 pyproject에 설정되어 있다).

## 6. 환경 함정

- **이전 개발 PC(RTX 4050 Laptop)에서는 외부 Python 3.14 프로세스가 GPU를 간헐적으로 점유한다.** 사용자가 멈출 수 없다고 했다.
  - 벽시계 처리량은 32~455 tick/s로 크게 흔들린다.
  - 그래서 S2 성능은 `bench_life.py --profile`의 **커널 시간 합(평균, 최소)**으로 판정한다.
  - 현재 PC(RTX 2070 SUPER)에서는 경합이 없어서 벽시계로 판정한다(약 600 tick/s, `stage2-results.md` §5.1).
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
| S2 | 5만 마리에서 200 tick/s 이상 | 2070 SUPER 벽시계 412~426 (단계 3b, 유전체 124) |
| S3 | 24시간 연속 실행 무사고 | 단계 5 |
| S4 | 10 seed × 10만 tick 중 8개 이상 존속 | 20/20 (3b 게이트 v2, 최소 슬롯 1,886) |
| S4' | 100만 tick 3회 중 2회 이상 존속 | 단계 6. 참고로 뷰어 1회 115만 tick 존속 |
| S5 | 화학합성이 아닌 에너지 전략이 각본 없이 출현 | 광합성 10/20, 포식 출현(3b). 판정은 단계 6 |
| S6 | 원소 수지 오차 < 0.1% | 약 1e-11 |

## 8. 다음 작업

- 단계 1~4 평가 통과. 단계 6 통과(S3 2차 판정 대기), 7a 통과(S2를 S3 종료 뒤 13만 슬롯 번성 상태에서 벽시계로 재측정). 7a 측정: 주화성은 기능하지만 선택된 증거 없음, 7a 코드에서 광합성 지속 번성이 늦어짐(seed 3 약 188만 tick).
- ④ 설계: `docs/design/stage7b-arch.md`(7b 몸 → 7c 신경 → 7d 학습). 사용자 "끝까지 진행해"(10-22) → 열린 질문은 추천안. **7b 통과**(게이트 20/20, 생물체 각본 없이 3/20). **7c 통과**(신경계: 발생 부호화 + CTRNN, 유전자 298, 저항력 이론을 구슬-막대 + 중점 규칙으로 고침, 게이트 20/20, 신경 가진 생물체 5/20, 끝까지 생존 0). **7d 통과**(공유 뉴런 풀, 생물체 표 65,536, 포화 기록, 스키마 7, 3요소 학습 + 습관화 + 위상 조절 신호, 덧붙이기 발생, 디스크 여유 확인, 유전자 330). ④는 기작으로 완료, 진화로 남는 증거는 아직 없음. 다음: 생물체가 남지 못하는 이유 측정(수명·사망 원인·에너지 수지). 디스크 여유가 적다(예전 기록 폴더 삭제는 사용자 결정 대기).
- 사용자 계획(2026-10-20): ① 미생물 화학 감각(7a, 완료) → README에 사진 넣어 푸시 → **④ 중추 뇌와 학습**. ④는 생물을 "점"에서 "여러 세포의 몸"으로 바꾸는 큰 구조 변경이 먼저 필요하다(②세포 분업, ③신경 그물). GPU 8 GB로는 선충급 뇌(시냅스 수천) 수만 마리가 현실적이라고 사용자에게 설명했다.
- 사용자 요청: 오래 지켜보며 변화가 생기면 캡처하고 기록 → `evo/analysis/events.py`로 구현됨.
- S3: 1차 24시간 무사고 통과, 메모리 +7.6%(기준 5% 미달) → 종 기록 상한으로 수정 후 재측정 중(`runs/s3_monitor.log`).
- 커밋: `stage3-6-live-observation` 브랜치에 1커밋 푸시됨. 그 뒤 변경은 미커밋.

### 사용자와 확정한 방향 (2026-10-18)
- **관찰만 한다.** 시작한 뒤에는 개입하지 않는다. 개발 중 계수 튜닝은 예외다.
- **"지구와 95% 일치"는 목표가 아니다.** 잴 기준이 없고, 진화는 다시 돌려도 같지 않으며, 결과를 맞추면 각본이 된다. 대신 지구에서 작동한 원리와 사건 순서(화학합성 → 광합성 → 산소 증가 → 메탄 온실 붕괴·결빙 → 호기성 번성)가 각본 없이 재현되는지를 본다.
- **화면은 완전히 현실적인 다큐멘터리 느낌**으로 만든다(단계 4). 화면에 보이는 것은 모두 실제 시뮬레이션 값에서 나와야 한다. 데이터에 없는 것을 꾸며 그리지 않는다.

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
