"""Run every real-flight model in sim on the circle track and dump per-step CSVs
matched to the rosbag folders, so analyze_real_traj.py can overlay sim vs real.

Isaac Sim is launched only once; the five checkpoints are loaded sequentially.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

local_rsl_path = os.path.abspath("src/third_parties/rsl_rl_local")
if os.path.exists(local_rsl_path):
    sys.path.insert(0, local_rsl_path)

import argparse

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Race-v0")
parser.add_argument("--model_root", type=str,
                    default=f"{project_root}/roslog/sim2real_2/sim2real_2")
parser.add_argument("--bag_root", type=str,
                    default=f"{project_root}/roslog/sim2real_2/sim2real_2/rosbags")
parser.add_argument("--episode_s", type=float, default=40.0,
                    help="How long to run each model (seconds).")
parser.add_argument("--track", type=str, default="circle",
                    help="Track to use (circle/powerloop/...).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--models", type=str, nargs="*", default=None,
                    help="Specific model names; default is all folders with best_model.pt.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
import glob
import shutil

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import src.isaac_quad_sim2real.tasks  # noqa: F401


def discover_models(model_root, wanted):
    found = []
    for entry in sorted(os.listdir(model_root)):
        full = os.path.join(model_root, entry)
        ckpt = os.path.join(full, "best_model.pt")
        if os.path.isdir(full) and os.path.isfile(ckpt):
            if wanted and entry not in wanted:
                continue
            found.append((entry, ckpt))
    return found


def run_one_episode(env, base, policy, dt_step, max_steps, csv_path):
    obs = env.get_observations()
    if hasattr(obs, "get"):
        obs = obs["policy"]
    rows = []
    for step in range(max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)
            if hasattr(obs, "get"):
                obs = obs["policy"]
        pos = base._robot.data.root_link_pos_w[0].cpu()
        vel = base._robot.data.root_com_lin_vel_w[0].cpu()
        spd = float(torch.linalg.norm(vel))
        gidx = int(base._idx_wp[0].item())
        ng = int(base._n_gates_passed[0].item())
        term = bool(base.reset_terminated[0].item())
        trunc = bool(base.reset_time_outs[0].item())
        rows.append([
            step, f"{step * dt_step:.4f}",
            f"{pos[0].item():.4f}", f"{pos[1].item():.4f}", f"{pos[2].item():.4f}",
            f"{vel[0].item():.4f}", f"{vel[1].item():.4f}", f"{vel[2].item():.4f}",
            f"{spd:.4f}", gidx, ng, 0.0, 0.0, int(term), int(trunc), 0.0, "best_model.pt",
        ])
        if bool(dones[0].item()):
            break
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step", "time_s", "pos_x", "pos_y", "pos_z",
            "vel_x", "vel_y", "vel_z", "speed",
            "gate_idx", "n_gates_passed", "x_gate_frame",
            "reward", "terminated", "truncated", "abs_pitch_rate", "checkpoint",
        ])
        w.writerows(rows)
    return len(rows)


def main():
    models = discover_models(args_cli.model_root, set(args_cli.models or []))
    if not models:
        print(f"[ERROR] No models found under {args_cli.model_root}")
        return
    print(f"[INFO] Models to replay: {[m[0] for m in models]}")
    print(f"[INFO] Track: {args_cli.track}")

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True,
    )
    env_cfg.is_train = False
    env_cfg.seed = args_cli.seed
    env_cfg.track_name = args_cli.track
    env_cfg.max_n_laps = 99  # disable completion timeout; run full episode_s

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    base = env.unwrapped
    dt_step = base.cfg.sim.dt * base.cfg.decimation
    max_steps = int(args_cli.episode_s / dt_step)

    # build runner once; we'll just swap checkpoint state dicts
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    for name, ckpt in models:
        print(f"\n[RUN] model={name}  ckpt={ckpt}")
        try:
            runner.load(ckpt)
        except Exception as e:
            print(f"   [ERROR] load failed: {e}")
            continue
        policy = runner.get_inference_policy(device=base.device)

        # force a fresh episode for this model
        base.episode_length_buf[:] = base.max_episode_length  # trigger done on next step
        # take a single no-op step to let the env auto-reset all envs
        with torch.inference_mode():
            env.step(torch.zeros((base.num_envs, base.cfg.action_space), device=base.device))

        bag_dir = os.path.join(args_cli.bag_root, f"group36_{name}")
        os.makedirs(bag_dir, exist_ok=True)
        out_csv = os.path.join(bag_dir, "sim_traj.csv")
        n = run_one_episode(env, base, policy, dt_step, max_steps, out_csv)
        print(f"   wrote {n} rows -> {out_csv}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
