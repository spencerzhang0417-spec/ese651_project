"""Overlay sim vs. real for one or more circle-track flights.

For each (bag, pt) pair, md5 the .pt and locate the matching
logs/rsl_rl/sim2real_part1/<ts>/best_model.pt, then read its
videos/play/play_metrics.csv as the sim rollout. Overlay against the real
trajectory exported by scripts/analyze_real_flight.py.

Usage:
    # explicit pairing: bag_dir=pt_path
    python scripts/compare_sim_real.py \\
        --pair rosbags/group36_faster=rosbags/faster/best_model.pt \\
        --pair rosbags/group36_1=rosbags/first\\ run/best_model.pt \\
        --real-out outputs/real_analysis \\
        --out outputs/real_analysis/sim_vs_real.png

    # auto: uses PAIRS dict below
    python scripts/compare_sim_real.py --auto \\
        --out outputs/real_analysis/sim_vs_real.png [--show-3d]
"""

import argparse
import csv
import glob
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation as R

# --- shared track constants -------------------------------------------------
CIRCLE_WAYPOINTS = np.array(
    [
        [0.0, 3.0, 0.75, 0.0, 0.0, 0.00],
        [-1.5, 4.5, 0.75, 0.0, 0.0, -1.57],
        [0.0, 6.0, 1.75, 0.0, 0.0, 3.14],
        [1.5, 4.5, 0.75, 0.0, 0.0, 1.57],
    ]
)
GATE_SIDE = 1.0
PASS_RADIUS = 0.75

# auto-mode bag_dir_name -> pt_dir_name (siblings under rosbags/)
PAIRS = {"group36_1": "first run", "group36_faster": "faster"}

# real bag_dir_name -> namespace (inferred from topic scan, see analyze script)
REAL_NAMESPACES = {"group36_1": "crazy_jirl_b2", "group36_faster": "crazy_jirl_b3"}


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_sim_run_by_md5(pt_md5, logs_root):
    matches = []
    for p in sorted(glob.glob(str(Path(logs_root) / "*" / "best_model.pt"))):
        if md5_file(p) == pt_md5:
            matches.append(Path(p).parent)
    return matches


def read_play_metrics(run_dir):
    csv_path = run_dir / "videos" / "play" / "play_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    rows = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    t = np.array([float(r["time_s"]) for r in rows])
    pos = np.array([[float(r["pos_x"]), float(r["pos_y"]), float(r["pos_z"])] for r in rows])
    vel = np.array([[float(r["vel_x"]), float(r["vel_y"]), float(r["vel_z"])] for r in rows])
    speed = np.array([float(r["speed"]) for r in rows])
    n_passed = np.array([int(r["n_gates_passed"]) for r in rows])
    terminated = np.array([int(r["terminated"]) for r in rows])
    return dict(t=t, pos=pos, vel=vel, speed=speed, n_passed=n_passed, terminated=terminated)


def sim_first_pass_time(sim):
    idx = np.where(np.diff(sim["n_passed"]) >= 1)[0]
    if len(idx) == 0:
        return None
    return float(sim["t"][idx[0] + 1])


def sim_collisions(sim):
    """Return list of t indices where terminated flips 0->1."""
    term = sim["terminated"]
    flips = np.where((term[1:] == 1) & (term[:-1] == 0))[0] + 1
    return [float(sim["t"][i]) for i in flips]


def build_gate_frames(waypoints):
    pos = waypoints[:, 0:3]
    rots = [R.from_euler("xyz", e) for e in waypoints[:, 3:6]]
    return pos, rots


def gate_square_3d(wp_pos, rot, side=GATE_SIDE):
    d = side / 2.0
    local = np.array([[0, d, d], [0, -d, d], [0, -d, -d], [0, d, -d], [0, d, d]])
    return rot.apply(local) + wp_pos


def gate_circle_3d(wp_pos, rot, radius=PASS_RADIUS, n=40):
    th = np.linspace(0, 2 * np.pi, n)
    local = np.stack([np.zeros_like(th), radius * np.cos(th), radius * np.sin(th)], axis=1)
    return rot.apply(local) + wp_pos


def rotate_body_to_world(vel_b, quat_xyzw):
    return R.from_quat(quat_xyzw).apply(vel_b)


def load_real(real_out_dir, bag_name):
    p = Path(real_out_dir) / f"{bag_name}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p}. Run scripts/analyze_real_flight.py on rosbags/{bag_name} first."
        )
    with open(p) as f:
        return json.load(f)


def prepare_pair(bag_dir, pt_path, real_out_dir, logs_root, analyze_on_miss=True):
    bag_name = Path(bag_dir).name.replace(" ", "_")
    real = load_real(real_out_dir, bag_name)
    pt_md5 = md5_file(pt_path)
    sim_matches = find_sim_run_by_md5(pt_md5, logs_root)
    if not sim_matches:
        raise SystemExit(
            f"no sim run with md5 {pt_md5} for {pt_path} under {logs_root}. "
            f"Re-run scripts/rsl_rl/play_race.py --video against that checkpoint."
        )
    sim = read_play_metrics(sim_matches[0])
    sim_t0 = sim_first_pass_time(sim)
    if sim_t0 is None:
        sim_t0 = sim["t"][0]
    if real["gate_passes"]:
        real_t0 = real["gate_passes"][0]["t_rel"]
    else:
        real_t0 = 0.0
    return dict(
        bag_name=bag_name,
        pt_md5=pt_md5,
        sim_run=sim_matches[0],
        sim=sim,
        sim_t0=sim_t0,
        real=real,
        real_t0=real_t0,
    )


def summarize(pair):
    sim = pair["sim"]
    real = pair["real"]
    sim_speed_max = float(np.max(sim["speed"]))
    real_quat = np.array(real["odom"]["quat_xyzw"])
    real_vel_b = np.array(real["odom"]["vel_body"])
    real_vel_w = rotate_body_to_world(real_vel_b, real_quat)
    real_speed_max = float(np.max(np.linalg.norm(real_vel_w, axis=1)))
    sim_gates = int(sim["n_passed"][-1])
    real_gates = len(real["gate_passes"])
    sim_colls = len(sim_collisions(sim))
    real_colls = len(real["collisions"])

    def lap_time_from_passes(pass_times):
        if len(pass_times) < 5:
            return None
        return float(pass_times[4] - pass_times[0])

    sim_pass_idx = np.where(np.diff(sim["n_passed"]) >= 1)[0] + 1
    sim_pass_t = sim["t"][sim_pass_idx]
    sim_lap = lap_time_from_passes(sim_pass_t)
    real_pass_t = np.array([gp["t_rel"] for gp in real["gate_passes"]])
    real_lap = lap_time_from_passes(real_pass_t)
    wp_z = CIRCLE_WAYPOINTS[:, 2]
    real_pos = np.array(real["odom"]["pos"])
    sim_z_err = float(np.max([abs(sim["pos"][i, 2] - wp_z[i % 4]) for i in range(len(sim["pos"]))]))
    real_z_err = float(np.max([abs(p[2] - min(wp_z.tolist(), key=lambda z: abs(p[2] - z)))
                               for p in real_pos]))
    return dict(
        sim_speed_max=sim_speed_max, real_speed_max=real_speed_max,
        sim_gates=sim_gates, real_gates=real_gates,
        sim_collisions=sim_colls, real_collisions=real_colls,
        sim_lap_time=sim_lap, real_lap_time=real_lap,
        sim_max_dz=sim_z_err, real_max_dz=real_z_err,
    )


def render_overlay(pairs, wp_pos, wp_rots, out_png):
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.6, 1, 1])
    axxy = fig.add_subplot(gs[0])
    axz = fig.add_subplot(gs[1])
    axs = fig.add_subplot(gs[2])

    for i, (p, r) in enumerate(zip(wp_pos, wp_rots)):
        sq = gate_square_3d(p, r)
        cr = gate_circle_3d(p, r)
        axxy.plot(sq[:, 0], sq[:, 1], color="0.3")
        axxy.plot(cr[:, 0], cr[:, 1], color="0.7", lw=0.6, linestyle="--")
        axxy.text(p[0], p[1] + 0.2, f"G{i}", color="0.3")
        axz.axhline(p[2], color="0.8", lw=0.5, linestyle="--")

    colors = plt.get_cmap("tab10")
    summary_lines = []
    for ci, pair in enumerate(pairs):
        c = colors(ci)
        name = pair["bag_name"]
        sim, real = pair["sim"], pair["real"]
        sim_t = sim["t"] - pair["sim_t0"]
        real_t_odom = np.array(real["odom"]["t"]) - pair["real_t0"]
        real_pos = np.array(real["odom"]["pos"])
        real_quat = np.array(real["odom"]["quat_xyzw"])
        real_vel_w = rotate_body_to_world(np.array(real["odom"]["vel_body"]), real_quat)
        real_speed = np.linalg.norm(real_vel_w, axis=1)

        axxy.plot(sim["pos"][:, 0], sim["pos"][:, 1], color=c, lw=1.5,
                  label=f"{name} SIM")
        axxy.plot(real_pos[:, 0], real_pos[:, 1], color=c, lw=1.0, linestyle="--",
                  label=f"{name} REAL")

        axz.plot(sim_t, sim["pos"][:, 2], color=c, lw=1.5)
        axz.plot(real_t_odom, real_pos[:, 2], color=c, lw=1.0, linestyle="--")

        axs.plot(sim_t, sim["speed"], color=c, lw=1.5)
        axs.plot(real_t_odom, real_speed, color=c, lw=1.0, linestyle="--")

        # gate-pass markers
        sim_pass_idx = np.where(np.diff(sim["n_passed"]) >= 1)[0] + 1
        for k in sim_pass_idx:
            axxy.scatter(sim["pos"][k, 0], sim["pos"][k, 1], color="green", s=25, marker="o", zorder=4)
        for gp in real["gate_passes"]:
            kk = int(np.searchsorted(np.array(real["odom"]["t"]), gp["t_rel"] + pair["real_t0"]))
            kk = min(kk, len(real_pos) - 1)
            axxy.scatter(real_pos[kk, 0], real_pos[kk, 1], color="green", s=25, marker="^", zorder=4)

        # collision markers
        for ct in sim_collisions(sim):
            k = int(np.searchsorted(sim["t"], ct))
            k = min(k, len(sim["pos"]) - 1)
            axxy.scatter(sim["pos"][k, 0], sim["pos"][k, 1], color="red", s=60, marker="x", zorder=5)
        for cl in real["collisions"]:
            kk = int(np.searchsorted(np.array(real["odom"]["t"]), cl["t_rel"] + pair["real_t0"]))
            kk = min(kk, len(real_pos) - 1)
            axxy.scatter(real_pos[kk, 0], real_pos[kk, 1], color="red", s=60, marker="x", zorder=5)

        s = summarize(pair)
        summary_lines.append(
            f"{name}: sim gates={s['sim_gates']} (coll {s['sim_collisions']}), "
            f"real gates={s['real_gates']} (coll {s['real_collisions']}); "
            f"lap sim={s['sim_lap_time']} real={s['real_lap_time']}; "
            f"max|Δz| sim={s['sim_max_dz']:.2f} real={s['real_max_dz']:.2f}; "
            f"vmax sim={s['sim_speed_max']:.2f} real={s['real_speed_max']:.2f}"
        )

    axxy.set_aspect("equal")
    axxy.set_xlabel("x [m]"); axxy.set_ylabel("y [m]"); axxy.set_title("XY overlay")
    axxy.grid(True, alpha=0.3); axxy.legend(loc="upper right", fontsize=8)
    axz.set_xlabel("t − t_first_pass [s]"); axz.set_ylabel("z [m]"); axz.set_title("z(t)")
    axz.grid(True, alpha=0.3)
    axs.set_xlabel("t − t_first_pass [s]"); axs.set_ylabel("speed [m/s]"); axs.set_title("speed(t)")
    axs.grid(True, alpha=0.3)

    fig.suptitle("Sim vs. Real — Circle track (solid=sim, dashed=real, ^=real pass)", fontsize=13)
    fig.text(
        0.01, 0.005, "\n".join(summary_lines),
        fontsize=8, family="monospace", va="bottom",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    fig.savefig(out_png, dpi=120)
    print(f"[info] wrote {out_png}")
    return fig


def render_3d(pairs, wp_pos, wp_rots):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    for i, (p, r) in enumerate(zip(wp_pos, wp_rots)):
        sq = gate_square_3d(p, r)
        cr = gate_circle_3d(p, r)
        ax.plot(sq[:, 0], sq[:, 1], sq[:, 2], color="0.3")
        ax.plot(cr[:, 0], cr[:, 1], cr[:, 2], color="0.7", lw=0.5, linestyle="--")
        ax.text(p[0], p[1], p[2] + 0.6, f"G{i}", color="0.3")
    colors = plt.get_cmap("tab10")
    for ci, pair in enumerate(pairs):
        c = colors(ci)
        sim, real = pair["sim"], pair["real"]
        real_pos = np.array(real["odom"]["pos"])
        ax.plot(sim["pos"][:, 0], sim["pos"][:, 1], sim["pos"][:, 2],
                color=c, lw=1.5, label=f"{pair['bag_name']} SIM")
        ax.plot(real_pos[:, 0], real_pos[:, 1], real_pos[:, 2],
                color=c, lw=1.0, linestyle="--", label=f"{pair['bag_name']} REAL")
        sim_pass_idx = np.where(np.diff(sim["n_passed"]) >= 1)[0] + 1
        for k in sim_pass_idx:
            ax.scatter(*sim["pos"][k], color="green", s=30, marker="o")
        for gp in real["gate_passes"]:
            kk = int(np.searchsorted(np.array(real["odom"]["t"]), gp["t_rel"] + pair["real_t0"]))
            kk = min(kk, len(real_pos) - 1)
            ax.scatter(*real_pos[kk], color="green", s=30, marker="^")
        for ct in sim_collisions(sim):
            k = int(np.searchsorted(sim["t"], ct))
            k = min(k, len(sim["pos"]) - 1)
            ax.scatter(*sim["pos"][k], color="red", s=60, marker="x")
        for cl in real["collisions"]:
            kk = int(np.searchsorted(np.array(real["odom"]["t"]), cl["t_rel"] + pair["real_t0"]))
            kk = min(kk, len(real_pos) - 1)
            ax.scatter(*real_pos[kk], color="red", s=60, marker="x")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Sim vs Real (3D, interactive)")
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", action="append", default=[],
                   help="bag_dir=pt_path. May be given multiple times.")
    p.add_argument("--auto", action="store_true",
                   help="Use PAIRS dict at top of file: group36_1/group36_faster.")
    p.add_argument("--real-out", default="outputs/real_analysis",
                   help="Directory where analyze_real_flight.py wrote <bag>.json")
    p.add_argument("--logs-root", default="logs/rsl_rl/sim2real_part1")
    p.add_argument("--out", default="outputs/real_analysis/sim_vs_real.png")
    p.add_argument("--show-3d", action="store_true")
    args = p.parse_args()

    pair_specs = []
    if args.auto:
        for bag_name, pt_dir in PAIRS.items():
            bag_dir = Path("rosbags") / bag_name
            pt_path = Path("rosbags") / pt_dir / "best_model.pt"
            if not pt_path.exists():
                print(f"[skip] {bag_name}: no pt at {pt_path}")
                continue
            pair_specs.append((str(bag_dir), str(pt_path)))
    for s in args.pair:
        if "=" not in s:
            raise SystemExit(f"--pair expects bag_dir=pt_path, got: {s}")
        a, b = s.split("=", 1)
        pair_specs.append((a, b))
    if not pair_specs:
        raise SystemExit("no pairs; pass --auto or --pair bag_dir=pt_path")

    pairs = []
    for bag_dir, pt_path in pair_specs:
        print(f"[info] pairing {bag_dir}  +  {pt_path}")
        pair = prepare_pair(bag_dir, pt_path, args.real_out, args.logs_root)
        print(
            f"       md5={pair['pt_md5'][:12]}  sim_run={pair['sim_run'].name}  "
            f"sim_t0={pair['sim_t0']:.2f}s  real_t0={pair['real_t0']:.2f}s"
        )
        pairs.append(pair)

    wp_pos, wp_rots = build_gate_frames(CIRCLE_WAYPOINTS)
    os.makedirs(Path(args.out).parent, exist_ok=True)
    render_overlay(pairs, wp_pos, wp_rots, args.out)

    if args.show_3d:
        render_3d(pairs, wp_pos, wp_rots)


if __name__ == "__main__":
    main()
