# got_driver.py
from typing import List, Tuple, Dict, Any
from vlm import ModelHandler
from utils import extract_driving_action, integrate_driving_commands

# -------- Prompts (short & strict) --------

class GoTPrompts:
    def scene(self, hint: str) -> str:
        return (
            "You label driving scenes.\n"
            f"Image: {hint}\n"
            "Describe ONLY what is visible (vehicles, people, lanes, lights/signs, road, hazards).\n"
            "One short paragraph. No lists. No actions."
        )

    def intent(self, scene: str, v: List[float], k: List[float]) -> str:
        return (
            "Propose 3 brief intent hypotheses based on the scene and history.\n"
            "Each: speed (accelerate/decelerate/maintain + amount) and steering (left/right/follow + amount).\n"
            f"Scene: {scene}\n"
            f"Past speeds (0.5s): {v}\n"
            f"Past curvatures (0.5s): {k}\n"
            "Output:\n"
            "1) ...\n2) ...\n3) ..."
        )

    def actions(self, scene: str, intent: str, v: List[float], k: List[float]) -> str:
        return (
            "Predict EXACTLY 6 (speed, curvature) pairs at 0.5s for the next 3s.\n"
            "Follow the intent; keep values smooth and feasible.\n"
            f"Scene: {scene}\n"
            f"Intent: {intent}\n"
            f"Past speeds: {v}\n"
            f"Past curvatures: {k}\n"
            "Return ONLY a Python list like: [(v1,c1),(v2,c2),(v3,c3),(v4,c4),(v5,c5),(v6,c6)]"
        )

    def refine(self, scene: str, intent: str, actions: str, critique: str) -> str:
        return (
            "Revise the actions if needed for smoothness and feasibility. Return ONLY 6 tuples.\n"
            f"Scene: {scene}\nIntent: {intent}\nProposed: {actions}\nCritique: {critique}\n"
            "Revised:"
        )

# -------- Light parsing --------

class GoTParser:
    def scene(self, text: str) -> str:
        return text.strip()

    def intents(self, text: str) -> List[str]:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        out = []
        for l in lines:
            if l[:2].isdigit() or l[:1] in "-•*":
                l = l.split(")", 1)[-1].split(".", 1)[-1].strip()
            out.append(l)
        return out[:3] or []

    def actions(self, text: str) -> List[Tuple[float, float]]:
        return extract_driving_action(text, error_handling=True) or []

# -------- Heuristic scoring (GT-free) --------

def traj_cost(traj: List[Tuple[float, float]], acts: List[Tuple[float, float]]) -> float:
    if not traj or not acts or len(acts) != 6:
        return 1e9
    x_end = traj[-1][0]
    v = [a for a, _ in acts]
    k = [b for _, b in acts]

    V_MIN, V_MAX, K_MAX = 0.0, 20.0, 0.30
    accel = sum(abs(v[i]-v[i-1]) for i in range(1, len(v)))
    jerk  = sum(abs(k[i]-k[i-1]) for i in range(1, len(k)))
    bounds = sum((V_MIN - x)*5.0 for x in v if x < V_MIN) \
           + sum((x - V_MAX)*2.0 for x in v if x > V_MAX) \
           + sum((abs(x) - K_MAX)*5.0 for x in k if abs(x) > K_MAX)
    return -x_end + 0.3*accel + 0.6*jerk + bounds

# -------- Orchestrator --------

class GoTRunner:
    def __init__(self, model_handler: ModelHandler, beam_k: int = 4, top_m: int = 2, refinements: int = 1):
        self.mh = model_handler
        self.px = GoTPrompts()
        self.ps = GoTParser()
        self.beam_k, self.top_m, self.refinements = beam_k, top_m, refinements

    def _n(self, prompt: str, image: str, n: int) -> List[Dict[str, Any]]:
        outs = []
        for i in range(n):
            p = f"{prompt}\n\n(Diverse candidate #{i+1}.)"
            text, tokens, t = self.mh.get_response(prompt=p, image_path=image)
            outs.append({"text": text, "tokens": tokens, "time": t})
        return outs

    def run_frame(self, image_path: str, prev_speed: List[float], prev_curv: List[float]) -> Dict[str, Any]:
        tok = {"scene_prompt": {"input": 0, "output": 0},
               "intent_prompt": {"input": 0, "output": 0},
               "waypoint_prompt": {"input": 0, "output": 0}}
        tim = {"scene_prompt": 0.0, "intent_prompt": 0.0, "waypoint_prompt": 0.0}
        branches = []

        # 1) scenes
        s_prompt = self.px.scene("[CAM_FRONT]")
        s_raw = self._n(s_prompt, image_path, self.beam_k)
        for r in s_raw:
            tok["scene_prompt"]["input"] += r["tokens"]["input"]
            tok["scene_prompt"]["output"] += r["tokens"]["output"]
            tim["scene_prompt"] += r["time"]
        scenes = [self.ps.scene(r["text"]) for r in s_raw]

        # 2) intents
        si = []
        for sc in scenes:
            ip = self.px.intent(sc, prev_speed, prev_curv)
            r = self._n(ip, image_path, 1)[0]
            tok["intent_prompt"]["input"] += r["tokens"]["input"]
            tok["intent_prompt"]["output"] += r["tokens"]["output"]
            tim["intent_prompt"] += r["time"]
            si.append((sc, self.ps.intents(r["text"]) or [r["text"].strip()]))

        # 3) actions
        for sc, intents in si:
            for it in intents[:self.beam_k]:
                ap = self.px.actions(sc, it, prev_speed, prev_curv)
                r = self._n(ap, image_path, 1)[0]
                tok["waypoint_prompt"]["input"] += r["tokens"]["input"]
                tok["waypoint_prompt"]["output"] += r["tokens"]["output"]
                tim["waypoint_prompt"] += r["time"]

                acts = self.ps.actions(r["text"])
                if not acts: 
                    continue
                traj = integrate_driving_commands(acts, dt=0.5)
                branches.append({
                    "scene_desc": sc,
                    "intent": it,
                    "actions_str": r["text"],
                    "actions": acts,
                    "trajectory": traj,
                    "cost": traj_cost(traj, acts),
                })

        # Fallback
        if not branches:
            ap = self.px.actions(scenes[0] if scenes else "Unparsed scene",
                                 "Follow lane; maintain/slight adjust speed; avoid conflicts.",
                                 prev_speed, prev_curv)
            r = self._n(ap, image_path, 1)[0]
            acts = self.ps.actions(r["text"])
            traj = integrate_driving_commands(acts, dt=0.5) if acts else []
            branches = [{
                "scene_desc": scenes[0] if scenes else "",
                "intent": "fallback",
                "actions_str": r["text"],
                "actions": acts,
                "trajectory": traj,
                "cost": traj_cost(traj, acts) if acts else 1e9,
            }]

        # 4) select & refine
        branches.sort(key=lambda b: b["cost"])
        survivors = branches[:max(1, self.top_m)]
        for _ in range(self.refinements):
            refd = []
            for b in survivors:
                rp = self.px.refine(
                    b["scene_desc"], b["intent"], b["actions_str"],
                    "Reduce speed spikes and curvature jerk; keep forward progress within bounds."
                )
                r = self._n(rp, image_path, 1)[0]
                tok["waypoint_prompt"]["input"] += r["tokens"]["input"]
                tok["waypoint_prompt"]["output"] += r["tokens"]["output"]
                tim["waypoint_prompt"] += r["time"]
                ra = self.ps.actions(r["text"])
                if not ra:
                    refd.append(b); continue
                rt = integrate_driving_commands(ra, dt=0.5)
                rc = traj_cost(rt, ra)
                refd.append(min(b, {
                    "scene_desc": b["scene_desc"],
                    "intent": b["intent"],
                    "actions_str": r["text"],
                    "actions": ra,
                    "trajectory": rt,
                    "cost": rc,
                }, key=lambda x: x["cost"]))
            survivors = sorted(refd, key=lambda x: x["cost"])[:max(1, self.top_m)]

        win = min(survivors, key=lambda b: b["cost"])
        return {
            "best_actions_str": win["actions_str"],
            "best_actions": win["actions"],
            "branches": branches,
            "token_usage": tok,
            "time_usage": {
                "scene_prompt": tim["scene_prompt"],
                "intent_prompt": tim["intent_prompt"],
                "waypoint_prompt": tim["waypoint_prompt"],
                "total": tim["scene_prompt"] + tim["intent_prompt"] + tim["waypoint_prompt"],
            },
        }
