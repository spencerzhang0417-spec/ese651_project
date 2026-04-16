"""Decode a circle-track flight bag, detect gate passes + collisions, and plot.

Usage:
    python scripts/analyze_real_flight.py <bag_dir> <namespace> \\
        [--t0 T0] [--tf TF] [--out OUT]

Reads the single .mcap inside <bag_dir>, streams /odom, /ctbr_cmd, /observations
for the given namespace, and writes <out>/<bag_name>_diagnostics.png +
<out>/<bag_name>.json.

Gate-pass rule mirrors quadcopter_strategies.py:73-92
(prev_x_gate > 0 & curr_x_gate <= 0 & yz_dist < 0.75). Waypoints from
quadcopter_env.py:433-438 (circle track). Collision = plane crossing on any
gate while drone is outside the 1.0 m x 1.0 m opening.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mcap_ros2.reader import read_ros2_messages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from scipy.spatial.transform import Rotation as R

# --- Track constants (circle track, quadcopter_env.py:433-438) --------------
CIRCLE_WAYPOINTS = np.array(
    [
        [0.0, 3.0, 0.75, 0.0, 0.0, 0.00],
        [-1.5, 4.5, 0.75, 0.0, 0.0, -1.57],
        [0.0, 6.0, 1.75, 0.0, 0.0, 3.14],
        [1.5, 4.5, 0.75, 0.0, 0.0, 1.57],
    ]
)
GATE_SIDE = 1.0  # quadcopter_env.py:125
GATE_HALF = GATE_SIDE / 2.0
PASS_RADIUS = 0.75  # quadcopter_strategies.py:78

# --- Collision detection constants ------------------------------------------
# A "frame hit" means the drone crossed a gate's plane while it was outside the
# 1.0 m opening but still *close* to the frame. Anything far outside (e.g. the
# drone flying past a different gate on the opposite side of the track) is
# ignored.
COLLISION_OPENING_HALF = GATE_HALF  # 0.5 m (inside this = clean pass, not a hit)
COLLISION_MARGIN = 0.4  # extra metres past the edge still counted as a hit
COLLISION_DEBOUNCE_S = 0.3


def build_gate_frames(waypoints):
    """Return (N,3) positions, (N,) yaws, list of scipy Rotations (world<-gate)."""
    pos = waypoints[:, 0:3]
    eul = waypoints[:, 3:6]
    rots = [R.from_euler("xyz", e) for e in eul]
    return pos, eul[:, 2], rots


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def parse_bag(bag_path, ns):
    odom_t, odom_pos, odom_quat, odom_vel_b = [], [], [], []
    ctbr_t, ctbr_thrust, ctbr_body_rates = [], [], []
    obs_t, obs_payload = [], []

    odom_topic = f"/{ns}/odom"
    obs_topic = f"/{ns}/observations"
    ctbr_topic = "/ctbr_cmd"

    for m in read_ros2_messages(bag_path):
        topic = m.channel.topic
        msg = m.ros_msg
        if topic == odom_topic:
            t = stamp_to_sec(msg.header.stamp)
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            v = msg.twist.twist.linear
            odom_t.append(t)
            odom_pos.append([p.x, p.y, p.z])
            odom_quat.append([q.x, q.y, q.z, q.w])  # scipy convention
            odom_vel_b.append([v.x, v.y, v.z])
        elif topic == ctbr_topic:
            # CommandCTBR has no explicit stamp; use log time.
            t = m.log_time_ns * 1e-9
            ctbr_t.append(t)
            ctbr_thrust.append(msg.thrust_n)
            ctbr_body_rates.append([msg.roll_rate, msg.pitch_rate, msg.yaw_rate])
        elif topic == obs_topic:
            t = m.log_time_ns * 1e-9
            obs_t.append(t)
            obs_payload.append(
                {
                    "lin_vel": list(msg.lin_vel),
                    "rot": list(msg.rot),
                    "corners_pos_b_curr": list(msg.corners_pos_b_curr),
                    "corners_pos_b_next": list(msg.corners_pos_b_next),
                }
            )

    return dict(
        odom_t=np.array(odom_t),
        odom_pos=np.array(odom_pos),
        odom_quat=np.array(odom_quat),
        odom_vel_b=np.array(odom_vel_b),
        ctbr_t=np.array(ctbr_t),
        ctbr_thrust=np.array(ctbr_thrust),
        ctbr_body_rates=np.array(ctbr_body_rates),
        obs_t=np.array(obs_t),
        obs_payload=obs_payload,
    )


def detect_events(odom_t, odom_pos, wp_pos, wp_rots):
    """Return (gate_passes, collisions). t_rel in seconds from first odom sample."""
    t0 = odom_t[0]
    t_rel = odom_t - t0
    n_wp = len(wp_pos)

    # Pre-compute drone pos in every gate frame for every sample: shape (N, n_wp, 3)
    pos_in_g = np.stack(
        [wp_rots[i].inv().apply(odom_pos - wp_pos[i]) for i in range(n_wp)], axis=1
    )
    x_g = pos_in_g[:, :, 0]
    yz_dist = np.linalg.norm(pos_in_g[:, :, 1:], axis=2)

    # --- Gate pass: sequential, like the strategy code -----------------------
    gate_passes = []
    idx = 0
    prev_x = x_g[0, idx]
    for k in range(1, len(odom_t)):
        cx = x_g[k, idx]
        if prev_x > 0 and cx <= 0 and yz_dist[k, idx] < PASS_RADIUS:
            gate_passes.append(
                dict(t_rel=float(t_rel[k]), idx_before=int(idx), idx_after=int((idx + 1) % n_wp))
            )
            idx = (idx + 1) % n_wp
            prev_x = x_g[k, idx]
        else:
            prev_x = cx

    # --- Collision: per-sample plane crossings outside the opening, all gates -
    collisions = []
    y_g = pos_in_g[:, :, 1]
    z_g = pos_in_g[:, :, 2]
    last_collision_t = {i: -1e9 for i in range(n_wp)}
    pass_times = {round(gp["t_rel"], 3) for gp in gate_passes}
    for i in range(n_wp):
        for k in range(1, len(odom_t)):
            crossed = (x_g[k - 1, i] > 0 and x_g[k, i] <= 0) or (
                x_g[k - 1, i] < 0 and x_g[k, i] >= 0
            )
            if not crossed:
                continue
            ay, az = abs(y_g[k, i]), abs(z_g[k, i])
            if ay <= COLLISION_OPENING_HALF and az <= COLLISION_OPENING_HALF:
                continue  # inside the opening -> pass-through, not a hit
            outer = COLLISION_OPENING_HALF + COLLISION_MARGIN
            if ay > outer or az > outer:
                continue  # too far from the frame to be a hit
            tk = float(t_rel[k])
            if tk - last_collision_t[i] < COLLISION_DEBOUNCE_S:
                continue
            if round(tk, 3) in pass_times:
                continue
            last_collision_t[i] = tk
            collisions.append(
                dict(t_rel=tk, gate_idx=int(i), y_g=float(y_g[k, i]), z_g=float(z_g[k, i]))
            )

    collisions.sort(key=lambda c: c["t_rel"])
    return gate_passes, collisions


def rotate_body_to_world(vel_b, quat_xyzw):
    """quat_xyzw: (N,4) scipy convention."""
    rots = R.from_quat(quat_xyzw)
    return rots.apply(vel_b)


def gate_square_3d(wp_pos, rot, side=GATE_SIDE):
    """Return the 5 corners (closing polygon) of the gate frame in world coords."""
    d = side / 2.0
    local = np.array([[0, d, d], [0, -d, d], [0, -d, -d], [0, d, -d], [0, d, d]])
    return rot.apply(local) + wp_pos


def gate_circle_3d(wp_pos, rot, radius=PASS_RADIUS, n=40):
    th = np.linspace(0, 2 * np.pi, n)
    local = np.stack([np.zeros_like(th), radius * np.cos(th), radius * np.sin(th)], axis=1)
    return rot.apply(local) + wp_pos


def make_plots(data, passes, collisions, wp_pos, wp_rots, out_png, title):
    t0 = data["odom_t"][0]
    t_odom = data["odom_t"] - t0
    vel_w = rotate_body_to_world(data["odom_vel_b"], data["odom_quat"])
    speed = np.linalg.norm(vel_w, axis=1)

    fig = plt.figure(figsize=(16, 16))
    fig.suptitle(title, fontsize=14)
    gs = fig.add_gridspec(4, 2)

    # --- 3D trajectory ------------------------------------------------------
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax3d.plot(data["odom_pos"][:, 0], data["odom_pos"][:, 1], data["odom_pos"][:, 2],
              color="C0", lw=1.2)
    for i, (p, r) in enumerate(zip(wp_pos, wp_rots)):
        sq = gate_square_3d(p, r)
        ax3d.plot(sq[:, 0], sq[:, 1], sq[:, 2], color="0.3")
        ax3d.text(p[0], p[1], p[2] + 0.6, f"G{i}", color="0.3")
    for gp in passes:
        k = np.searchsorted(t_odom, gp["t_rel"])
        k = min(k, len(t_odom) - 1)
        ax3d.scatter(*data["odom_pos"][k], color="green", s=30, marker="o")
    for cl in collisions:
        k = np.searchsorted(t_odom, cl["t_rel"])
        k = min(k, len(t_odom) - 1)
        ax3d.scatter(*data["odom_pos"][k], color="red", s=50, marker="x")
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    ax3d.set_title("3D trajectory")

    # --- XY top-down --------------------------------------------------------
    axxy = fig.add_subplot(gs[0, 1])
    axxy.plot(data["odom_pos"][:, 0], data["odom_pos"][:, 1], color="C0", lw=1.2,
              label="trajectory")
    for i, (p, r) in enumerate(zip(wp_pos, wp_rots)):
        sq = gate_square_3d(p, r)
        axxy.plot(sq[:, 0], sq[:, 1], color="0.3")
        cr = gate_circle_3d(p, r)
        axxy.plot(cr[:, 0], cr[:, 1], color="0.7", lw=0.7, linestyle="--")
        axxy.text(p[0], p[1] + 0.15, f"G{i}", color="0.3")
    # heading ticks every 0.5 s
    dt = np.median(np.diff(t_odom)) if len(t_odom) > 1 else 0.02
    stride = max(1, int(round(0.5 / dt)))
    for k in range(0, len(t_odom), stride):
        yaw = R.from_quat(data["odom_quat"][k]).as_euler("xyz")[2]
        dx, dy = 0.12 * np.cos(yaw), 0.12 * np.sin(yaw)
        axxy.arrow(data["odom_pos"][k, 0], data["odom_pos"][k, 1], dx, dy,
                   head_width=0.04, color="0.5", length_includes_head=True, alpha=0.5)
    for gp in passes:
        k = min(np.searchsorted(t_odom, gp["t_rel"]), len(t_odom) - 1)
        axxy.scatter(data["odom_pos"][k, 0], data["odom_pos"][k, 1],
                     color="green", s=40, marker="o", zorder=4)
    for cl in collisions:
        k = min(np.searchsorted(t_odom, cl["t_rel"]), len(t_odom) - 1)
        axxy.scatter(data["odom_pos"][k, 0], data["odom_pos"][k, 1],
                     color="red", s=60, marker="x", zorder=4)
    axxy.set_aspect("equal")
    axxy.set_xlabel("x [m]"); axxy.set_ylabel("y [m]")
    axxy.set_title(f"XY top-down  (passes: {len(passes)}, collisions: {len(collisions)})")
    axxy.grid(True, alpha=0.3)

    # --- z(t) ----------------------------------------------------------------
    axz = fig.add_subplot(gs[1, 0])
    axz.plot(t_odom, data["odom_pos"][:, 2], color="C0")
    for i, p in enumerate(wp_pos):
        axz.axhline(p[2], color="0.7", lw=0.6, linestyle="--")
        axz.text(t_odom[-1], p[2], f" G{i}={p[2]:.2f}", va="center", color="0.4", fontsize=8)
    for gp in passes:
        axz.axvline(gp["t_rel"], color="green", lw=0.5, alpha=0.6)
    for cl in collisions:
        axz.axvline(cl["t_rel"], color="red", lw=0.5, alpha=0.6)
    axz.set_xlabel("t [s]"); axz.set_ylabel("z [m]"); axz.set_title("z(t)")
    axz.grid(True, alpha=0.3)

    # --- speed + vxyz -------------------------------------------------------
    axs = fig.add_subplot(gs[1, 1])
    axs.plot(t_odom, speed, color="k", label="|v|")
    axs.plot(t_odom, vel_w[:, 0], color="C0", lw=0.6, label="vx")
    axs.plot(t_odom, vel_w[:, 1], color="C1", lw=0.6, label="vy")
    axs.plot(t_odom, vel_w[:, 2], color="C2", lw=0.6, label="vz")
    axs.set_xlabel("t [s]"); axs.set_ylabel("m/s"); axs.set_title("speed & velocity (world)")
    axs.grid(True, alpha=0.3); axs.legend(loc="upper right", fontsize=8)

    # --- thrust --------------------------------------------------------------
    axt = fig.add_subplot(gs[2, 0])
    if len(data["ctbr_t"]):
        axt.plot(data["ctbr_t"] - t0, data["ctbr_thrust"], color="C3")
    axt.set_xlabel("t [s]"); axt.set_ylabel("thrust [N]"); axt.set_title("commanded thrust")
    axt.grid(True, alpha=0.3)

    # --- body rates ---------------------------------------------------------
    axr = fig.add_subplot(gs[2, 1])
    if len(data["ctbr_t"]):
        tr = data["ctbr_t"] - t0
        axr.plot(tr, data["ctbr_body_rates"][:, 0], color="C0", label="roll")
        axr.plot(tr, data["ctbr_body_rates"][:, 1], color="C1", label="pitch")
        axr.plot(tr, data["ctbr_body_rates"][:, 2], color="C2", label="yaw")
    axr.set_xlabel("t [s]"); axr.set_ylabel("rate [deg/s]")
    axr.set_title("commanded body rates")
    axr.grid(True, alpha=0.3); axr.legend(loc="upper right", fontsize=8)

    # --- gate idx + cumulative passes ---------------------------------------
    axg = fig.add_subplot(gs[3, :])
    idx_seq_t = [0.0]
    idx_seq_v = [0]
    for gp in passes:
        idx_seq_t.append(gp["t_rel"])
        idx_seq_v.append(gp["idx_before"])
        idx_seq_t.append(gp["t_rel"])
        idx_seq_v.append(gp["idx_after"])
    idx_seq_t.append(t_odom[-1])
    idx_seq_v.append(idx_seq_v[-1])
    axg.step(idx_seq_t, idx_seq_v, where="post", color="C0", label="target gate idx")
    axg.set_ylabel("gate idx", color="C0")
    axg.tick_params(axis="y", labelcolor="C0")
    axg2 = axg.twinx()
    cum_t = [0.0] + [gp["t_rel"] for gp in passes] + [t_odom[-1]]
    cum_v = [0] + list(range(1, len(passes) + 1)) + [len(passes)]
    axg2.step(cum_t, cum_v, where="post", color="C2", label="cumulative passes")
    axg2.set_ylabel("cum. passes", color="C2")
    axg2.tick_params(axis="y", labelcolor="C2")
    for cl in collisions:
        axg.axvline(cl["t_rel"], color="red", lw=0.7, alpha=0.7)
    axg.set_xlabel("t [s]"); axg.set_title("gate progression")
    axg.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bag_dir")
    p.add_argument("namespace")
    p.add_argument("--t0", type=float, default=None)
    p.add_argument("--tf", type=float, default=None)
    p.add_argument("--out", default="outputs/real_analysis")
    args = p.parse_args()

    bag_dir = Path(args.bag_dir)
    mcaps = sorted(glob.glob(str(bag_dir / "*.mcap")))
    if not mcaps:
        raise SystemExit(f"no .mcap in {bag_dir}")
    if len(mcaps) > 1:
        print(f"[warn] multiple .mcap files, using {mcaps[0]}")
    bag_path = mcaps[0]
    bag_name = bag_dir.name.replace(" ", "_")

    print(f"[info] reading {bag_path}")
    data = parse_bag(bag_path, args.namespace)
    if len(data["odom_t"]) == 0:
        raise SystemExit(f"no /{args.namespace}/odom messages found")

    # Clip by t0/tf relative to first odom sample
    t_start = data["odom_t"][0]
    if args.t0 is not None or args.tf is not None:
        lo = t_start + (args.t0 or 0.0)
        hi = t_start + (args.tf if args.tf is not None else 1e18)
        for key_t, key_arrs in [
            ("odom_t", ["odom_pos", "odom_quat", "odom_vel_b"]),
            ("ctbr_t", ["ctbr_thrust", "ctbr_body_rates"]),
            ("obs_t", ["obs_payload"]),
        ]:
            t = data[key_t]
            if len(t) == 0:
                continue
            mask = (t >= lo) & (t <= hi)
            data[key_t] = t[mask]
            for k in key_arrs:
                if k == "obs_payload":
                    data[k] = [x for x, m in zip(data[k], mask) if m]
                else:
                    data[k] = data[k][mask]

    wp_pos, _, wp_rots = build_gate_frames(CIRCLE_WAYPOINTS)
    passes, collisions = detect_events(data["odom_t"], data["odom_pos"], wp_pos, wp_rots)

    topic_counts = {
        "odom": int(len(data["odom_t"])),
        "ctbr_cmd": int(len(data["ctbr_t"])),
        "observations": int(len(data["obs_t"])),
    }
    print(
        f"[info] counts: {topic_counts}  passes={len(passes)}  collisions={len(collisions)}"
    )

    os.makedirs(args.out, exist_ok=True)
    png_path = Path(args.out) / f"{bag_name}_diagnostics.png"
    json_path = Path(args.out) / f"{bag_name}.json"

    t0_rel = data["odom_t"][0]
    export = dict(
        bag=str(bag_path),
        namespace=args.namespace,
        t0_epoch=float(t0_rel),
        topic_counts=topic_counts,
        odom=dict(
            t=(data["odom_t"] - t0_rel).tolist(),
            pos=data["odom_pos"].tolist(),
            quat_xyzw=data["odom_quat"].tolist(),
            vel_body=data["odom_vel_b"].tolist(),
        ),
        ctbr=dict(
            t=(data["ctbr_t"] - t0_rel).tolist() if len(data["ctbr_t"]) else [],
            thrust_n=data["ctbr_thrust"].tolist() if len(data["ctbr_t"]) else [],
            body_rates_deg_s=data["ctbr_body_rates"].tolist() if len(data["ctbr_t"]) else [],
        ),
        observations=dict(
            t=(data["obs_t"] - t0_rel).tolist() if len(data["obs_t"]) else [],
            payload=data["obs_payload"],
        ),
        gate_passes=passes,
        collisions=collisions,
    )
    with open(json_path, "w") as f:
        json.dump(export, f)
    print(f"[info] wrote {json_path}")

    make_plots(data, passes, collisions, wp_pos, wp_rots, png_path, title=bag_name)
    print(f"[info] wrote {png_path}")


if __name__ == "__main__":
    main()
