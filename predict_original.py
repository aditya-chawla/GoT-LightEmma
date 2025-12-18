import os
import argparse
import datetime
from nuscenes import NuScenes

from utils import *
from vlm import ModelHandler


def parse_args():
    parser = argparse.ArgumentParser(description="LightEMMA: End-to-End Autonomous Driving")
    parser.add_argument("--model", type=str, default="chatgpt-4o-latest", 
                        help="Options: gpt-series, claude-series, gemini-series, "
                        "qwen2.5-7b, qwen2.5-72b, llama-3.2-11b, llama-3.2-90b")
    parser.add_argument("--continue_dir", type=str, default=None,
                        help="Path to the directory with previously processed scene JSON files to resume processing")
    parser.add_argument("--scene", type=str, default=None,
                        help="Optional: Specific scene name to process.")
    return parser.parse_args()

def run_prediction():
    # Parse arguments and load configuration
    args = parse_args()
    config = load_config("config.yaml")

    # Load NuScenes parameters from config
    OBS_LEN = config["prediction"]["obs_len"]
    FUT_LEN = config["prediction"]["fut_len"]
    EXT_LEN = config["prediction"]["ext_len"]
    TTL_LEN = OBS_LEN + FUT_LEN + EXT_LEN 
    
    # Initialize model
    model_handler = ModelHandler(args.model, config)
    model_handler.initialize_model()
    print(f"Using model: {args.model}")
    
    # Initialize NuScenes dataset
    nusc = NuScenes(version=config["data"]["version"], dataroot=config["data"]["root"], verbose=True)
    
    # Configure output paths
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Use the provided directory for continuation, or create a new one
    if args.continue_dir:
        results_dir = args.continue_dir
        print(f"Continuing from existing directory: {results_dir}")
    else:
        results_dir = f"{config['data']['results']}/{args.model}_{timestamp}/output"
        os.makedirs(results_dir, exist_ok=True)
        print(f"Created new results directory: {results_dir}")
    
    # Select scenes to process
    if args.scene:
        # Find the specific scene by name
        selected_scenes = [scene for scene in nusc.scene if scene["name"] == args.scene]
        if not selected_scenes:
            print(f"Scene '{args.scene}' not found in dataset")
            return
    else:
        # Process all scenes if no specific scene is specified
        selected_scenes = nusc.scene
        print(f"Processing all {len(selected_scenes)} scenes")
    
    # Process each selected scene
    for scene in selected_scenes:
        scene_name = scene["name"]

        # Skip if already processed in continuation mode
        output_path = os.path.join(results_dir, "output", f"{scene_name}.json")
        if os.path.exists(output_path):
            print(f"Skipping already processed scene: {scene_name}")
            continue

        first_sample_token = scene["first_sample_token"]
        last_sample_token = scene["last_sample_token"]
        description = scene["description"]
        
        print(f"\nProcessing scene '{scene_name}': {description}")
        
        # Create scene data structure
        scene_data = {
            "scene_info": {
                "name": scene_name,
                "description": description,
                "first_sample_token": first_sample_token,
                "last_sample_token": last_sample_token
            },
            "frames": [],
            "metadata": {
                "model": args.model,
                "timestamp": timestamp,
                "total_frames": 0
            }
        }
        
        # Collect scene data
        camera_params = []
        front_camera_images = []
        ego_positions = []
        ego_headings = []
        timestamps = []
        sample_tokens = []
        
        curr_sample_token = first_sample_token
        
        # Retrieve all frames in the scene
        while curr_sample_token:
            sample = nusc.get("sample", curr_sample_token)
            sample_tokens.append(curr_sample_token)
            
            cam_front_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            front_camera_images.append(
                os.path.join(nusc.dataroot, cam_front_data["filename"])
            )
            
            # Get the camera parameters
            camera_params.append(
                nusc.get("calibrated_sensor", cam_front_data["calibrated_sensor_token"])
            )
            
            # Get ego vehicle state
            ego_state = nusc.get("ego_pose", cam_front_data["ego_pose_token"])
            ego_positions.append(tuple(ego_state["translation"][0:2]))
            ego_headings.append(quaternion_to_yaw(ego_state["rotation"]))
            timestamps.append(ego_state["timestamp"])
            
            # Move to next sample or exit loop if at the end
            curr_sample_token = (
                sample["next"] if curr_sample_token != last_sample_token else None
            )
        
        num_frames = len(front_camera_images)
        
        # Check if we have enough frames
        if num_frames < TTL_LEN:
            print(f"Skipping '{scene_name}', insufficient frames ({num_frames} < {TTL_LEN}).")
            continue
        
        # Process each frame in the scene
        for i in range(0, num_frames - TTL_LEN, 1):
            try:
                cur_index = i + OBS_LEN + 1
                frame_index = i  # The relative index in the processed subset
                
                image_path = front_camera_images[cur_index]
                print(f"Processing frame {i} from {scene_name}, image: {image_path}")
                
                sample_token = sample_tokens[cur_index]
                camera_param = camera_params[cur_index]
                
                # Get current position and heading
                cur_pos = ego_positions[cur_index]
                cur_heading = ego_headings[cur_index]
                
                # Get observation data (past positions and timestamps)
                obs_pos = ego_positions[cur_index - OBS_LEN - 1 : cur_index + 1]
                obs_pos = global_to_ego_frame(cur_pos, cur_heading, obs_pos)
                obs_time = timestamps[cur_index - OBS_LEN - 1 : cur_index + 1]
                
                # Calculate past speeds and curvatures
                prev_speed = compute_speed(obs_pos, obs_time)
                prev_curvatures = compute_curvature(obs_pos)
                prev_actions = list(zip(prev_speed, prev_curvatures))
                
                # Get future positions and timestamps (ground truth)
                fut_pos = ego_positions[cur_index - 1 : cur_index + FUT_LEN + 1]
                fut_pos = global_to_ego_frame(cur_pos, cur_heading, fut_pos)
                
                # Remove extra indices used for speed and curvature calculation
                fut_pos = fut_pos[2:]
                
                # Define prompts for LLM inference
                scene_prompt = (
                    f"You are an autonomous driving labeller. "
                    "You have access to the front-view camera image. "
                    "You must observe and analyze the movements of vehicles and pedestrians, "
                    "lane markings, traffic lights, and any relevant objects in the scene. "
                    "describe what you observe, but do not infer the ego's action. "
                    "generate your response in plain text in one paragraph without any formating. "
                )
                
                # Run scene description inference
                scene_description, scene_tokens, scene_time = model_handler.get_response(
                    prompt=scene_prompt,
                    image_path=image_path
                )
                print("Scene description:", scene_description)
                
                # Generate intent prompt based on scene description
                intent_prompt = (
                    f"You are an autonomous driving labeller. "
                    "You have access to the front-view camera image. "
                    "The scene is described as follows: "
                    f"{scene_description} "
                    "The ego vehicle's speed for the past 3 seconds with 0.5 sec resolution is"
                    f"{prev_speed} m/s (last index is the most recent) "
                    "The ego vehicle's curvature for the past 3 seconds with 0.5 sec resolution is"
                    f"{prev_curvatures} (last index is the most recent) "
                    "A positive curvature indicates the ego is turning left."
                    "A negative curvature indicates the ego is turning right. "
                    "What was the ego's previous intent? "
                    "Was it accelerating (by how much), decelerating (by how much), or maintaining speed? "
                    "Was it turning left (by how much), turning right (by how much), or following the lane? "
                    "Taking into account the ego's previous intent, how should it drive in the next 3 seconds? "
                    "Should the ego accelerate (by how much), decelerate (by how much), or maintain speed? "
                    "Should the ego turn left (by how much), turn right (by how much), or follow the lane?  "
                    "Generate your response in plain text in one paragraph without any formating. "
                )
                
                # Run driving intent inference
                driving_intent, intent_tokens, intent_time = model_handler.get_response(
                    prompt=intent_prompt,
                    image_path=image_path
                )
                print("Driving intent:", driving_intent)
                
                # Generate waypoint prompt based on scene and intent
                waypoint_prompt = (
                    f"You are an autonomous driving labeller. "
                    "You have access to the front-view camera image. "
                    "The scene is described as follows: "
                    f"{scene_description} "
                    "The ego vehicle's speed for the past 3 seconds with 0.5 sec resolution is"
                    f"{prev_speed} m/s (last index is the most recent) "
                    "The ego vehicle's curvature for the past 3 seconds with 0.5 sec resolution is"
                    f"{prev_curvatures} (last index is the most recent) "
                    "A positive curvature indicates the ego is turning left."
                    "A negative curvature indicates the ego is turning right. "
                    "The high-level driving instructions are as follows: "
                    f"{driving_intent} "
                    "Predict the speed and curvature for the next 6 waypoints, with 0.5-second resolution. "
                    "The predicted speed and curvature changes must obey the physical constraints of the vehicle. "
                    "Predict Exactly 6 pairs of speed and curvature, in the format:"
                    "[(v1, c1), (v2, c2), (v3, c3), (v4, c4), (v5, c5), (v6, c6)]. "
                    "ONLY return the answers in the required format, do not include punctuation or text."
                )
                
                # Run waypoint prediction inference
                pred_actions_str, waypoint_tokens, waypoint_time = model_handler.get_response(
                    prompt=waypoint_prompt,
                    image_path=image_path
                )
                print("Predicted actions:", pred_actions_str)
                
                # Prepare frame data structure
                frame_data = {
                    "frame_index": frame_index,
                    "sample_token": sample_token,
                    "image_name": os.path.basename(image_path),
                    "timestamp": timestamps[cur_index],
                    "camera_params": {
                        "rotation": camera_param["rotation"],
                        "translation": camera_param["translation"],
                        "camera_intrinsic": camera_param["camera_intrinsic"]
                    },
                    "ego_info": {
                        "position": cur_pos,
                        "heading": cur_heading,
                        "obs_positions": obs_pos,
                        "obs_actions": prev_actions,
                        "gt_positions": fut_pos,
                    },
                    "inference": {
                        "scene_prompt": format_long_text(scene_prompt),
                        "scene_description": format_long_text(scene_description),
                        "intent_prompt": format_long_text(intent_prompt),
                        "driving_intent": format_long_text(driving_intent),
                        "waypoint_prompt": format_long_text(waypoint_prompt),
                        "pred_actions_str": pred_actions_str
                    },
                    "token_usage": {
                        "scene_prompt": scene_tokens,
                        "intent_prompt": intent_tokens,
                        "waypoint_prompt": waypoint_tokens
                    },
                    "time_usage": {
                        "scene_prompt": scene_time,
                        "intent_prompt": intent_time,
                        "waypoint_prompt": waypoint_time
                    }
                }
                
                # Add frame data to scene
                scene_data["frames"].append(frame_data)
                
            except Exception as e:
                print(f"Error processing frame {i} in {scene_name}: {e}")
                continue
        
        # Update total frames count
        scene_data["metadata"]["total_frames"] = len(scene_data["frames"])
        
        # Save scene data
        scene_file_path = f"{results_dir}/{scene_name}.json"
        save_dict_to_json(scene_data, scene_file_path)
        print(f"Scene data saved to {scene_file_path} with {len(scene_data['frames'])} frames")

if __name__ == "__main__":
    run_prediction()