# predict.py (minimal, GoT-enabled)
import os, argparse, datetime
from nuscenes import NuScenes
from utils import *
from vlm import ModelHandler
from got_driver import GoTRunner

# ---- CLI ----
def parse_args():
    p = argparse.ArgumentParser("LightEMMA inference")
    p.add_argument("--model", type=str, default="chatgpt-4o-latest")
    p.add_argument("--continue_dir", type=str, default=None)
    p.add_argument("--scene", type=str, default=None)
    p.add_argument("--reasoning", type=str, default="got", choices=["got","cot"])
    p.add_argument("--beam_k", type=int, default=4)
    p.add_argument("--top_m", type=int, default=2)
    p.add_argument("--refinements", type=int, default=1)
    return p.parse_args()

# ---- Short CoT prompts (baseline mode) ----
def build_scene_prompt():
    return ("Describe ONLY what is visible in one short paragraph "
            "(vehicles, people, lanes, lights/signs, road, hazards). No actions.")

def build_intent_prompt(scene, v, k):
    return (
        f"Scene: {scene}\n"
        f"Past speeds (0.5s): {v}\nPast curvatures (0.5s): {k}\n"
        "What was the ego's prior intent (acc/dec/maint + amount; left/right/follow + amount)? "
        "And what should it do for the next 3s? One paragraph."
    )

def build_waypoint_prompt(scene, v, k, intent):
    return (
        f"Scene: {scene}\nIntent: {intent}\n"
        f"Past speeds: {v}\nPast curvatures: {k}\n"
        "Return EXACTLY 6 (speed, curvature) tuples at 0.5s for 3s. "
        "Return ONLY a Python list like: [(v1,c1),(v2,c2),(v3,c3),(v4,c4),(v5,c5),(v6,c6)]"
    )

# ---- Main ----
def run_prediction():
    args = parse_args()
    cfg = load_config("config.yaml")

    OBS_LEN = cfg["prediction"]["obs_len"]
    FUT_LEN = cfg["prediction"]["fut_len"]
    EXT_LEN = cfg["prediction"]["ext_len"]
    TTL_LEN = OBS_LEN + FUT_LEN + EXT_LEN

    mh = ModelHandler(args.model, cfg); mh.initialize_model()
    print(f"Model: {args.model}")

    got = GoTRunner(mh, args.beam_k, args.top_m, args.refinements) if args.reasoning=="got" else None
    if got: print(f"GoT: beam_k={args.beam_k}, top_m={args.top_m}, refinements={args.refinements}")

    nusc = NuScenes(version=cfg["data"]["version"], dataroot=cfg["data"]["root"], verbose=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.continue_dir:
        results_dir = args.continue_dir; print(f"Continue: {results_dir}")
    else:
        results_dir = f"{cfg['data']['results']}/{args.model}_{ts}"
        os.makedirs(os.path.join(results_dir, "output"), exist_ok=True)
        print(f"Results: {results_dir}")

    scenes = [s for s in nusc.scene if s["name"] == args.scene] if args.scene else nusc.scene
    if args.scene and not scenes:
        print(f"Scene '{args.scene}' not found."); return
    if not args.scene: print(f"Processing {len(scenes)} scenes")

    for sc in scenes:
        name = sc["name"]
        out_path = os.path.join(results_dir, "output", f"{name}.json")
        if os.path.exists(out_path):
            print(f"Skip (exists): {name}"); continue

        first, last = sc["first_sample_token"], sc["last_sample_token"]
        desc = sc["description"]
        print(f"\nScene '{name}': {desc}")

        data = {
            "scene_info": {
                "name": name, "description": desc,
                "first_sample_token": first, "last_sample_token": last
            },
            "frames": [],
            "metadata": {"model": args.model, "timestamp": ts, "total_frames": 0}
        }

        cams, imgs, pos, hdg, tms, tokens = [], [], [], [], [], []
        cur = first
        while cur:
            smp = nusc.get("sample", cur); tokens.append(cur)
            cam = nusc.get("sample_data", smp["data"]["CAM_FRONT"])
            imgs.append(os.path.join(nusc.dataroot, cam["filename"]))
            cams.append(nusc.get("calibrated_sensor", cam["calibrated_sensor_token"]))
            ego = nusc.get("ego_pose", cam["ego_pose_token"])
            pos.append(tuple(ego["translation"][:2])); hdg.append(quaternion_to_yaw(ego["rotation"]))
            tms.append(ego["timestamp"])
            cur = smp["next"] if cur != last else None

        n = len(imgs)
        if n < TTL_LEN:
            print(f"Skip '{name}': frames {n} < {TTL_LEN}."); continue

        for i in range(0, n - TTL_LEN, 1):
            print(f"Processing frame {i+1}/{n - TTL_LEN} …")
            try:
                ci = i + OBS_LEN + 1
                image = imgs[ci]
                sample_tok = tokens[ci]
                cam_param = cams[ci]
                cur_pos, cur_hdg = pos[ci], hdg[ci]

                obs_pos = pos[ci - OBS_LEN - 1 : ci + 1]
                obs_pos = global_to_ego_frame(cur_pos, cur_hdg, obs_pos)
                obs_t = tms[ci - OBS_LEN - 1 : ci + 1]

                prev_v = compute_speed(obs_pos, obs_t)
                prev_k = compute_curvature(obs_pos)
                prev_actions = list(zip(prev_v, prev_k))

                fut = pos[ci - 1 : ci + FUT_LEN + 1]
                fut = global_to_ego_frame(cur_pos, cur_hdg, fut)[2:]

                # ---- reasoning ----
                if got:
                    g = got.run_frame(image_path=image, prev_speed=prev_v, prev_curv=prev_k)
                    pred_actions_str = g["best_actions_str"]
                    scene_desc = "Graph-of-Thought"
                    driving_intent = "Selected via scoring"
                    tok_scene = {"input": g["token_usage"]["scene_prompt"]["input"], "output": g["token_usage"]["scene_prompt"]["output"]}
                    tok_intent = {"input": g["token_usage"]["intent_prompt"]["input"], "output": g["token_usage"]["intent_prompt"]["output"]}
                    tok_waypt = {"input": g["token_usage"]["waypoint_prompt"]["input"], "output": g["token_usage"]["waypoint_prompt"]["output"]}
                    t_scene, t_intent, t_waypt = g["time_usage"]["scene_prompt"], g["time_usage"]["intent_prompt"], g["time_usage"]["waypoint_prompt"]
                else:
                    sp = build_scene_prompt()
                    scene_desc, sc_tok, sc_t = mh.get_response(prompt=sp, image_path=image)
                    ip = build_intent_prompt(scene_desc, prev_v, prev_k)
                    driving_intent, in_tok, in_t = mh.get_response(prompt=ip, image_path=image)
                    wp = build_waypoint_prompt(scene_desc, prev_v, prev_k, driving_intent)
                    pred_actions_str, wp_tok, wp_t = mh.get_response(prompt=wp, image_path=image)
                    tok_scene, tok_intent, tok_waypt = sc_tok, in_tok, wp_tok
                    t_scene, t_intent, t_waypt = sc_t, in_t, wp_t

                # ---- frame JSON (compat preserved) ----
                frame = {
                    "frame_index": i,
                    "sample_token": sample_tok,
                    "image_name": os.path.basename(image),
                    "timestamp": tms[ci],
                    "camera_params": {
                        "rotation": cam_param["rotation"],
                        "translation": cam_param["translation"],
                        "camera_intrinsic": cam_param["camera_intrinsic"]
                    },
                    "ego_info": {
                        "position": cur_pos,
                        "heading": cur_hdg,
                        "obs_positions": obs_pos,
                        "obs_actions": prev_actions,
                        "gt_positions": fut,
                    },
                    "inference": {
                        "scene_prompt": format_long_text("GoT" if got else sp),
                        "scene_description": format_long_text(scene_desc if not got else "Graph-of-Thought"),
                        "intent_prompt": format_long_text("GoT" if got else ip),
                        "driving_intent": format_long_text(driving_intent if not got else "Selected via GoT scoring"),
                        "waypoint_prompt": format_long_text("GoT" if got else wp),
                        "pred_actions_str": pred_actions_str
                    },
                    "token_usage": {
                        "scene_prompt": tok_scene,
                        "intent_prompt": tok_intent,
                        "waypoint_prompt": tok_waypt
                    },
                    "time_usage": {
                        "scene_prompt": t_scene,
                        "intent_prompt": t_intent,
                        "waypoint_prompt": t_waypt
                    }
                }
                data["frames"].append(frame)

            except Exception as e:
                print(f"Error frame {i} in {name}: {e}")
                continue

        data["metadata"]["total_frames"] = len(data["frames"])
        save_dict_to_json(data, out_path)
        print(f"Saved {out_path} with {data['metadata']['total_frames']} frames")

if __name__ == "__main__":
    run_prediction()
