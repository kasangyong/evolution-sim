import time

import dearpygui.dearpygui as dpg
import numpy as np
import taichi as ti

from evo.config import ATM_NAMES, SPECIES_NAMES
from evo.life.genome import GENE_NAMES
from evo.sim import Sim
from evo.ui.render import LIGHT_LAYER, TEMP_LAYER, Renderer
from evo.world.fields import CL_ICE, CL_TS

LAYERS = SPECIES_NAMES + ["Light", "Temperature"]
FRAME_BUDGET_S = 0.028
LEDGER_EVERY = 1000
LIFE_STATS_EVERY_FRAMES = 30
# logical size; Windows display scaling (e.g. 200%) enlarges the window further
VIEW_W = 900
VERTICAL_EXAGGERATION = 1.3


def layer_vmax(sim: Sim, layer: int) -> float:
    w = sim.world
    if layer == LIGHT_LAYER:
        a = w.light.to_numpy()
    elif layer == TEMP_LAYER:
        a = w.temp.to_numpy()
    else:
        a = w.chem.to_numpy()[layer]
    a = a[w.water_np]
    return float(max(np.percentile(a, 99.5), 1e-6))


def life_stats(sim: Sim) -> str:
    snap = sim.life.snapshot()
    n = len(snap["body"])
    if n == 0:
        return "life: extinct"
    means = snap["genome"].mean(axis=0)
    genes = "  ".join(f"{g} {m:.3g}" for g, m in zip(GENE_NAMES[:7], means[:7]))
    return f"life: {n:,} cells  biomass {snap['body'].sum():.3g}\nmean genes  {genes}"


def run(sim: Sim, gpu_name: str):
    w = sim.world
    renderer = Renderer(w, sim.life)
    state = {"paused": False, "speed": 10, "layer": 0, "vmax": 1.0, "err": None, "tps": 0.0,
             "show_life": True, "life": ""}

    dpg.create_context()
    with dpg.texture_registry():
        dpg.add_raw_texture(w.w, w.d, np.zeros(w.w * w.d * 4, np.float32), tag="world_tex",
                            format=dpg.mvFormat_Float_rgba)

    def on_layer(_, value):
        state["layer"] = LAYERS.index(value)
        state["vmax"] = layer_vmax(sim, state["layer"])

    view_h = int(VIEW_W * w.d / w.w * VERTICAL_EXAGGERATION)
    with dpg.window(tag="main"):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Pause/Run", callback=lambda: state.update(paused=not state["paused"]))
            dpg.add_slider_int(label="ticks/frame (0 = max)", default_value=10, min_value=0,
                               max_value=500, width=250,
                               callback=lambda _, v: state.update(speed=v))
            dpg.add_combo(LAYERS, default_value=LAYERS[0], width=140, callback=on_layer)
            dpg.add_checkbox(label="life", default_value=True,
                             callback=lambda _, v: state.update(show_life=v))
            dpg.add_button(label="Rescale", callback=lambda: on_layer(None, LAYERS[state["layer"]]))
        dpg.add_text("", tag="stats")
        dpg.add_text(f"GPU: {gpu_name}")
        dpg.add_image("world_tex", width=VIEW_W, height=view_h)

    dpg.create_viewport(title="Evolution Sim - stage 2", width=VIEW_W + 40,
                        height=view_h + 200, vsync=False)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main", True)
    state["vmax"] = layer_vmax(sim, state["layer"])

    last_ledger = sim.tick
    frame = 0
    while dpg.is_dearpygui_running():
        t_start = time.perf_counter()
        done = 0
        if not state["paused"]:
            if state["speed"] > 0:
                sim.step(state["speed"])
                done = state["speed"]
            else:
                while time.perf_counter() - t_start < FRAME_BUDGET_S:
                    sim.step(5)
                    done += 5
                    ti.sync()
            ti.sync()
        dt = time.perf_counter() - t_start
        if done:
            state["tps"] = 0.9 * state["tps"] + 0.1 * done / max(dt, 1e-6)
        if sim.tick - last_ledger >= LEDGER_EVERY:
            state["err"] = sim.budget_error()
            last_ledger = sim.tick

        renderer.draw(state["layer"], state["vmax"])
        if state["show_life"]:
            renderer.draw_life()
        frame += 1
        if frame % LIFE_STATS_EVERY_FRAMES == 1:
            state["life"] = life_stats(sim)
            state["vmax"] = layer_vmax(sim, state["layer"])
        dpg.set_value("world_tex", renderer.rgba.to_numpy().ravel())

        atm = w.atm.to_numpy()
        clim = w.clim.to_numpy()
        err = state["err"]
        dpg.set_value("stats", "\n".join([
            (f"tick {sim.tick:,}   (geologic ~ {sim.tick * 1000 / 1e6:,.2f} Myr at 1:1000, "
             f"not real Earth time)   {state['tps']:,.0f} tick/s"),
            f"surface T {clim[CL_TS]:.1f} C   ice {clim[CL_ICE] * 100:.0f}%   "
            + "   ".join(f"atm {n} {atm[i]:.3g}" for i, n in enumerate(ATM_NAMES)),
            "budget error  " + (f"C {err.carbon:.1e}  N {err.nitrogen:.1e}  e {err.electrons:.1e}"
                                if err else "(pending)"),
            f"layer {LAYERS[state['layer']]}  color scale max {state['vmax']:.3g}",
            state["life"],
        ]))
        dpg.render_dearpygui_frame()
    dpg.destroy_context()

