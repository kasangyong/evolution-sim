"""Event journal (stage 7a, user request 2026-10-20): watch the world and log notable changes —
a photosynthesis bloom, oxygen rising, a snowball Earth, a mass extinction, a new dominant species,
a new ability spreading — each with a description, the numbers behind it and where to look.

Pure Python on summaries the viewer already computes; no Taichi calls. Every detector has
hysteresis (it re-arms only after the condition has clearly gone away), so one event is logged
once, not every check."""
from dataclasses import dataclass, field


@dataclass
class Event:
    tick: int
    kind: str
    title: str  # short Korean headline
    detail: str  # what changed and the situation, in Korean
    numbers: dict = field(default_factory=dict)
    where: tuple | None = None  # (x, y) to zoom on, if the event has a place


@dataclass
class Summary:
    """What the detectors look at, taken every few thousand ticks."""
    tick: int
    cells: float
    slots: int
    lit_slots: int
    o2: float
    ch4: float
    co2: float
    ts: float
    ice: float
    # fractions of living slots that carry an ability
    pred_frac: float
    colony_frac: float
    swim_frac: float
    sense_frac: float  # food-sense tumble bias minus the largest control-sense bias
    # species: id -> (cells, (x, y) centre of its slots, main metabolism in Korean)
    species: dict = field(default_factory=dict)
    # where to look for an event kind (densest block of the cells involved), if known
    places: dict = field(default_factory=dict)
    # stage 7b: multicellular organisms, the longest body (segments), organisms that have swum
    orgs: int = 0
    org_seg_max: int = 0
    org_swimmers: int = 0
    # stage 7c: organisms with neurons; with a central brain (≥ 8 neurons, most in the head)
    org_neural: int = 0
    org_central: int = 0
    org_refused: int = 0  # colonies refused so far because the organism table was full


ABILITIES = (  # (summary field, kind, threshold, re-arm below, Korean name, what it means)
    ("pred_frac", "predation", 0.10, 0.03, "포식",
     "실제로 사냥하는(포식 효율 > 0.2이면서 사냥 중인) 세포가 생물의 10%를 넘었습니다"),
    ("colony_frac", "colony", 0.05, 0.01, "군체",
     "딸세포가 떨어지지 않고 붙어 사는 군체가 생물의 5%를 넘었습니다"),
    ("swim_frac", "swimming", 0.10, 0.03, "능동 헤엄",
     "스스로 헤엄치는 계통이 생물의 10%를 넘었습니다"),
    ("sense_frac", "chemotaxis", 0.10, 0.03, "화학 감각(주화성)",
     ("자기 먹이(화학합성은 수소, 광합성은 빛)가 줄면 방향을 바꾸는 헤엄 세포의 쏠림이, 먹이와 "
      "상관없는 다른 감각의 쏠림보다 생물의 10%p 이상 커졌습니다. 한 계통이 퍼지면 모든 감각의 "
      "배선이 함께 따라오므로 한 번의 쏠림만으로 선택이라고 단정할 수는 없습니다. 선택의 후보이고, "
      "여러 실행에서 먹이 감각만 일관되게 쏠리는지로 확인합니다")),
)


class EventDetector:
    def __init__(self):
        self.armed: dict[str, bool] = {}
        self.ch4_peak = 0.0
        self.cells_peak = 0.0
        self.major: set[int] | None = None  # set from the first summary: the start is no event
        # species ids flicker as clusters regroup: announce only what persists two checks
        self.pending: set[int] = set()  # major once, not yet announced
        self.announced: set[int] = set()
        self.missing: set[int] = set()  # announced, absent once

    def _fire(self, key: str, cond: bool, rearm: bool) -> bool:
        """True once when cond becomes true; re-arms when `rearm` holds."""
        armed = self.armed.get(key, True)
        if armed and cond:
            self.armed[key] = False
            return True
        if not armed and rearm:
            self.armed[key] = True
        return False

    def update(self, s: Summary) -> list[Event]:
        ev: list[Event] = []
        nums = {"tick": s.tick, "세포": round(s.cells), "빛 받는 슬롯": s.lit_slots,
                "대기 O₂": f"{s.o2:.3g}", "대기 CH₄": f"{s.ch4:.3g}", "수면 온도": round(s.ts, 1)}

        if self._fire("bloom", s.lit_slots > 1000, s.lit_slots < 100):
            ev.append(Event(s.tick, "bloom", "광합성 번성",
                            f"빛이 닿는 곳에서 사는 슬롯이 {s.lit_slots:,}개가 됐습니다. "
                            "광합성 계통이 표층으로 퍼졌습니다.", nums, s.places.get("bloom")))
        if self._fire("oxygen", s.o2 > 1e3, s.o2 < 10):
            ev.append(Event(s.tick, "oxygen", "산소 증가",
                            f"대기 산소가 {s.o2:.3g}에 이르렀습니다. 광합성이 내놓은 산소가 "
                            "환원제를 다 쓰고 남기 시작한 것입니다(대산화 사건과 같은 원리).", nums))
        self.ch4_peak = max(self.ch4_peak, s.ch4)
        if self._fire("methane", self.ch4_peak > 100 and s.ch4 < 0.1 * self.ch4_peak,
                      s.ch4 > 0.5 * self.ch4_peak):
            ev.append(Event(s.tick, "methane", "메탄 온실 붕괴",
                            f"대기 메탄이 최고 {self.ch4_peak:.3g}에서 {s.ch4:.3g}로 줄었습니다. "
                            "온실효과가 약해져 기후가 식습니다.", nums))
            self.ch4_peak = s.ch4
        # re-armed by the thaw below, so the thaw can see that a snowball happened
        if self._fire("snowball", s.ice >= 0.99, False):
            ev.append(Event(s.tick, "snowball", "전 해양 결빙(눈덩이 지구)",
                            f"수면 온도 {s.ts:.1f}°C, 바다 전체가 얼었습니다. 빛이 막혀 광합성 "
                            "생물이 사라지고 심해 열수구 생물만 남습니다.", nums))
        if not self.armed.get("snowball", True) and s.ice <= 0.01:  # only after a snowball
            self.armed["snowball"] = True
            ev.append(Event(s.tick, "thaw", "해빙",
                            f"얼음이 녹았습니다(수면 {s.ts:.1f}°C). 얼음 아래 쌓인 화산 CO₂ "
                            f"({s.co2:.3g})가 기후를 데웠습니다.", nums))

        self.cells_peak = max(self.cells_peak * 0.999, s.cells)  # slowly forgetting peak
        if self._fire("extinction", s.cells < 0.5 * self.cells_peak and self.cells_peak > 1000,
                      s.cells > 0.8 * self.cells_peak):
            ev.append(Event(s.tick, "extinction", "대멸종",
                            f"세포 수가 최근 최고 {self.cells_peak:,.0f}의 절반 아래"
                            f"({s.cells:,.0f})로 줄었습니다.", nums))

        for attr, kind, on, off, name, what in ABILITIES:
            frac = getattr(s, attr)
            if self._fire(kind, frac >= on, frac < off):
                ev.append(Event(s.tick, kind, f"새 능력 확산: {name}",
                                f"{what}(지금 {frac:.0%}).", nums, s.places.get(kind)))

        if self._fire("organism", s.orgs > 0, s.orgs == 0):
            ev.append(Event(s.tick, "organism", "다세포 생물체 출현",
                            f"분화하는 군체가 몸을 가진 생물체가 됐습니다(지금 {s.orgs}마리). 세포가 "
                            "대사·근육·섬모·생식 조직으로 나뉘고, 생식 조직이 내보낸 세포 하나에서 "
                            "다음 세대가 자랍니다.", nums, s.places.get("organism")))
        if self._fire("body", s.org_seg_max >= 4, s.org_seg_max < 2):
            ev.append(Event(s.tick, "body", "마디 4개 이상의 몸",
                            f"가장 긴 생물체가 {s.org_seg_max}마디가 됐습니다.", nums,
                            s.places.get("organism")))
        if self._fire("org_swim", s.org_swimmers > 0, s.orgs == 0):
            ev.append(Event(s.tick, "org_swim", "생물체가 헤엄친다",
                            f"근육이나 섬모로 스스로 몸길이의 몇 배 이상 나아간 생물체가 생겼습니다"
                            f"({s.org_swimmers}마리, 저항력 이론으로 계산한 실제 이동).", nums,
                            s.places.get("org_swim")))

        if self._fire("neurons", s.org_neural > 0, s.org_neural == 0):
            ev.append(Event(s.tick, "neurons", "첫 신경 세포",
                            f"신경계를 가진 생물체가 생겼습니다({s.org_neural}마리). 뉴런 유형 유전자가 "
                            "몸에 나타나 회로로 자랐습니다. 뉴런은 감각을 받아 마디를 굽힙니다.", nums,
                            s.places.get("neurons")))
        if self._fire("central", s.org_central > 0, s.org_central == 0):
            ev.append(Event(s.tick, "central", "중추 신경계(뇌)",
                            f"뉴런 8개 이상 중 절반 넘게가 머리에 모인 생물체가 생겼습니다"
                            f"({s.org_central}마리). 신경이 한곳에 모이는 중추화입니다.", nums,
                            s.places.get("central")))

        if self._fire("org_full", s.org_refused > 0, False):
            ev.append(Event(s.tick, "org_full", "생물체 표 가득 참(계산 상한)",
                            f"생물체 표가 가득 차 군체 {s.org_refused:,}개가 생물체가 되지 못했습니다. "
                            "자연선택이 아니라 계산 상한에 걸린 것입니다(개체 풀 상한과 같은 종류).",
                            nums))

        total = max(sum(c for c, _, _ in s.species.values()), 1.0)
        now_major = {sid for sid, (c, _, _) in s.species.items() if c >= 0.05 * total}
        if self.major is None:
            self.major = now_major
            self.announced = set(now_major)  # the founders: known, not news
            return ev
        for sid in sorted(now_major & self.pending - self.announced):  # major twice in a row
            c, where, meta = s.species[sid]
            self.announced.add(sid)
            ev.append(Event(s.tick, "species", f"새 주요 종 #{sid}",
                            f"종 #{sid}({meta})이 두 번 연속 판별에서 전체 세포의 {c / total:.0%}를 "
                            "차지했습니다.", nums, where))
        self.pending = now_major - self.announced
        alive = set(s.species)
        for sid in sorted(self.announced & self.missing - alive):  # absent twice in a row
            self.announced.discard(sid)
            ev.append(Event(s.tick, "species_gone", f"주요 종 #{sid} 사라짐",
                            f"주요 종이던 #{sid}이 두 번 연속 판별에서 보이지 않습니다(멸종했거나 "
                            "유전자가 바뀌어 다른 종으로 이어졌습니다).", nums))
        self.missing = self.announced - alive
        self.major = now_major
        return ev
