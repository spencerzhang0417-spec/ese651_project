"""Plot real-flight trajectories (from rosbags) in 3D, overlaying sim trajectories when available.

For each model the script:
  - reads /crazy_jirl_b3/pose from the matching rosbag (real flight)
  - looks for a sim trajectory CSV next to the bag folder (play_metrics.csv style schema)
  - draws both in one 3D plot + an altitude-over-time subplot

Sim CSV path convention (first match wins):
  1. <bag_dir>/sim_traj.csv
  2. <repo>/roslog/sim2real_2/sim2real_2/<model_name>/sim_traj.csv
  3. --sim_dir <dir>/<model_name>.csv  (if --sim_dir given on CLI)
"""
import os
import glob
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

ROSBAG_ROOT = "/home/boyan/ese651_project/roslog/sim2real_2/sim2real_2/rosbags"
MODEL_ROOT = "/home/boyan/ese651_project/roslog/sim2real_2/sim2real_2"
OUT_DIR = "/home/boyan/ese651_project/roslog/sim2real_2/plots"
os.makedirs(OUT_DIR, exist_ok=True)

# circle track waypoints (from quadcopter_env.py `circle` track)
# [x, y, z, yaw]
TRACK = np.array([
    [ 0.0, 3.0, 0.75,  0.00],  # G0
    [-1.5, 4.5, 0.75, -1.57],  # G1
    [ 0.0, 6.0, 1.75,  3.14],  # G2 HIGH
    [ 1.5, 4.5, 0.75,  1.57],  # G3
])
GATE_SIDE = 1.0  # matches GateModelCfg


def extract_real(bag_mcap):
    t, x, y, z = [], [], [], []
    with open(bag_mcap, "rb") as f:
        r = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, ros_msg in r.iter_decoded_messages(
            topics=["/crazy_jirl_b3/pose"]
        ):
            t.append(message.log_time / 1e9)
            x.append(ros_msg.pose.position.x)
            y.append(ros_msg.pose.position.y)
            z.append(ros_msg.pose.position.z)
    if not t:
        return None
    t = np.array(t); t -= t[0]
    return t, np.array(x), np.array(y), np.array(z)


def extract_sim(csv_path):
    t, x, y, z = [], [], [], []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            t.append(float(row["time_s"]))
            x.append(float(row["pos_x"]))
            y.append(float(row["pos_y"]))
            z.append(float(row["pos_z"]))
    if not t:
        return None
    return np.array(t), np.array(x), np.array(y), np.array(z)


def find_sim_csv(model_name, bag_dir, sim_dir_arg):
    for p in [
        os.path.join(bag_dir, "sim_traj.csv"),
        os.path.join(MODEL_ROOT, model_name, "sim_traj.csv"),
    ]:
        if os.path.isfile(p):
            return p
    if sim_dir_arg:
        p = os.path.join(sim_dir_arg, f"{model_name}.csv")
        if os.path.isfile(p):
            return p
    return None


def draw_gate_3d(ax, wp, side=GATE_SIDE, color="k", alpha=0.5):
    """Draw a square gate oriented by yaw around Z."""
    x0, y0, z0, yaw = wp
    d = side / 2
    local = np.array([
        [0,  d,  d],
        [0, -d,  d],
        [0, -d, -d],
        [0,  d, -d],
        [0,  d,  d],  # close loop
    ])
    cz, sz = np.cos(yaw), np.sin(yaw)
    R = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    world = local @ R.T + np.array([x0, y0, z0])
    ax.plot(world[:, 0], world[:, 1], world[:, 2], "-", color=color, lw=2.0, alpha=alpha)


def plot_track_3d(ax):
    for i, wp in enumerate(TRACK):
        draw_gate_3d(ax, wp)
        ax.text(wp[0], wp[1], wp[2] + 0.15, f"G{i}", fontsize=9)


def set_axes_equal(ax, xyz_points):
    xs = np.concatenate([p[:, 0] for p in xyz_points])
    ys = np.concatenate([p[:, 1] for p in xyz_points])
    zs = np.concatenate([p[:, 2] for p in xyz_points])
    rng = np.array([np.ptp(xs), np.ptp(ys), np.ptp(zs)])
    mid = np.array([xs.mean(), ys.mean(), zs.mean()])
    r = rng.max() / 2 + 0.3
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(max(0, mid[2] - r), mid[2] + r)


def plot_single(name, real, sim, out_path):
    fig = plt.figure(figsize=(13, 6))
    # 3D traj
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    plot_track_3d(ax1)
    xyz_list = []
    if real is not None:
        tr, xr, yr, zr = real
        xyz = np.stack([xr, yr, zr], 1); xyz_list.append(xyz)
        ax1.plot(xr, yr, zr, "-", color="tab:blue", lw=1.3, alpha=0.85, label="real")
        ax1.scatter(xr[0], yr[0], zr[0], c="tab:green", s=40, label="real start")
        ax1.scatter(xr[-1], yr[-1], zr[-1], c="tab:red", marker="x", s=60, label="real end")
    if sim is not None:
        ts, xs, ys, zs = sim
        xyz = np.stack([xs, ys, zs], 1); xyz_list.append(xyz)
        ax1.plot(xs, ys, zs, "-", color="tab:orange", lw=1.3, alpha=0.85, label="sim")
        ax1.scatter(xs[0], ys[0], zs[0], c="darkgreen", s=30, marker="^")
    # include track points in bounds
    xyz_list.append(np.concatenate([TRACK[:, :3], TRACK[:, :3] + [0, 0, 0.1]], axis=0).reshape(-1, 3))
    if xyz_list:
        set_axes_equal(ax1, xyz_list)
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)"); ax1.set_zlabel("z (m)")
    ax1.set_title(f"{name} — 3D trajectory")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.view_init(elev=22, azim=-60)

    # altitude vs time
    ax2 = fig.add_subplot(1, 2, 2)
    if real is not None:
        tr, xr, yr, zr = real
        ax2.plot(tr, zr, color="tab:blue", lw=1.0, label="real z")
    if sim is not None:
        ts, xs, ys, zs = sim
        ax2.plot(ts, zs, color="tab:orange", lw=1.0, label="sim z")
    ax2.axhline(0.75, color="k", ls="--", alpha=0.3, label="low gates")
    ax2.axhline(1.75, color="k", ls=":", alpha=0.3, label="high gate")
    ax2.set_xlabel("t (s)"); ax2.set_ylabel("z (m)")
    ax2.set_title("altitude vs time")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Sim2real: model {name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_all_combined(runs, out_path):
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    plot_track_3d(ax1)
    cmap = plt.get_cmap("tab10")
    xyz_list = [np.concatenate([TRACK[:, :3], TRACK[:, :3] + [0, 0, 0.1]], axis=0).reshape(-1, 3)]
    for i, (name, (real, sim)) in enumerate(runs.items()):
        c = cmap(i)
        if real is not None:
            tr, xr, yr, zr = real
            ax1.plot(xr, yr, zr, "-", color=c, lw=1.1, alpha=0.8, label=f"{name} (real)")
            xyz_list.append(np.stack([xr, yr, zr], 1))
        if sim is not None:
            ts, xs, ys, zs = sim
            ax1.plot(xs, ys, zs, "--", color=c, lw=1.1, alpha=0.5, label=f"{name} (sim)")
            xyz_list.append(np.stack([xs, ys, zs], 1))
    set_axes_equal(ax1, xyz_list)
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)"); ax1.set_zlabel("z (m)")
    ax1.set_title("Trajectories: all runs (solid=real, dashed=sim)")
    ax1.legend(fontsize=7, loc="upper left", ncol=2)
    ax1.view_init(elev=22, azim=-60)

    ax2 = fig.add_subplot(1, 2, 2)
    for i, (name, (real, sim)) in enumerate(runs.items()):
        c = cmap(i)
        if real is not None:
            ax2.plot(real[0], real[3], "-", color=c, lw=0.9, alpha=0.9, label=f"{name} real")
        if sim is not None:
            ax2.plot(sim[0], sim[3], "--", color=c, lw=0.9, alpha=0.6, label=f"{name} sim")
    ax2.axhline(0.75, color="k", ls="--", alpha=0.3)
    ax2.axhline(1.75, color="k", ls=":", alpha=0.3)
    ax2.set_xlabel("t (s)"); ax2.set_ylabel("z (m)")
    ax2.set_title("altitude vs time")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_dir", type=str, default=None,
                        help="Dir containing sim_traj CSVs named <model>.csv")
    args = parser.parse_args()

    bags = sorted(glob.glob(f"{ROSBAG_ROOT}/group36_*"))
    runs = {}
    for bag_dir in bags:
        name = os.path.basename(bag_dir).replace("group36_", "")
        mcap_files = sorted(glob.glob(f"{bag_dir}/*.mcap"))
        if not mcap_files:
            continue
        print(f"[READ] {name}")
        real = extract_real(mcap_files[0])
        if real is None:
            print(f"   no pose msgs")
            continue
        sim_csv = find_sim_csv(name, bag_dir, args.sim_dir)
        sim = extract_sim(sim_csv) if sim_csv else None
        if sim_csv:
            print(f"   sim overlay from {sim_csv}")
        runs[name] = (real, sim)
        plot_single(name, real, sim, f"{OUT_DIR}/traj_{name}.png")

    if runs:
        plot_all_combined(runs, f"{OUT_DIR}/traj_all.png")
        print(f"\n[INFO] Plots saved under {OUT_DIR}")

    print("\nSummary:")
    print(f"{'name':<12} {'dur(s)':>7} {'zmax':>6} {'sim?':>6}")
    for name, (real, sim) in runs.items():
        has_sim = "yes" if sim is not None else "no"
        print(f"{name:<12} {real[0][-1]:7.2f} {real[3].max():6.2f} {has_sim:>6}")

    missing = [n for n, (_, s) in runs.items() if s is None]
    if missing:
        print("\n[INFO] No sim trajectory found for:", ", ".join(missing))
        print("       Drop a CSV (play_metrics.csv schema) at one of:")
        print("         <bag_dir>/sim_traj.csv")
        print(f"         {MODEL_ROOT}/<model>/sim_traj.csv")
        print("       or pass --sim_dir <dir> with <model>.csv files.")


if __name__ == "__main__":
    main()
