"""Emit a Foxglove-friendly sidecar .mcap with gates + sim trajectory.

Loads the matching sim run for a given (bag_dir, pt_path) pair, builds a
sidecar mcap that contains:
  - /gates            foxglove.SceneUpdate (static, LINE_LOOP per gate)
  - /sim/pose         foxglove.PoseInFrame per sim sample (animated)
  - /sim/path         foxglove.SceneUpdate (static LINE_STRIP of full sim traj)
  - /sim/tf           foxglove.FrameTransform per sim sample (parent=mocap)

The sim messages are published with timestamps in the *real bag's* epoch so
Foxglove plays both back on one time cursor. Alignment = first gate pass in
the real bag matches first gate pass in sim.

Usage:
    python scripts/export_foxglove_overlay.py \\
        --pair rosbags/group36_faster=rosbags/faster/best_model.pt \\
        --real-out outputs/real_analysis \\
        --out outputs/real_analysis/group36_faster_overlay.mcap

    python scripts/export_foxglove_overlay.py --auto \\
        --out-dir outputs/real_analysis
"""

import argparse
import csv
import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from mcap.well_known import SchemaEncoding, MessageEncoding
from mcap_protobuf.writer import Writer as PbWriter
from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
from foxglove_schemas_protobuf.PoseInFrame_pb2 import PoseInFrame
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.LinePrimitive_pb2 import LinePrimitive
from scipy.spatial.transform import Rotation as R

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
FRAME_ID = "mocap"

PAIRS = {"group36_1": "first run", "group36_faster": "faster"}


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_sim_run_by_md5(pt_md5, logs_root):
    for p in sorted(glob.glob(str(Path(logs_root) / "*" / "best_model.pt"))):
        if md5_file(p) == pt_md5:
            return Path(p).parent
    return None


def read_play_metrics(run_dir):
    csv_path = run_dir / "videos" / "play" / "play_metrics.csv"
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    t = np.array([float(r["time_s"]) for r in rows])
    pos = np.array([[float(r["pos_x"]), float(r["pos_y"]), float(r["pos_z"])] for r in rows])
    n_passed = np.array([int(r["n_gates_passed"]) for r in rows])
    terminated = np.array([int(r["terminated"]) for r in rows])
    return dict(t=t, pos=pos, n_passed=n_passed, terminated=terminated)


def sim_first_pass_time(sim):
    idx = np.where(np.diff(sim["n_passed"]) >= 1)[0]
    return float(sim["t"][idx[0] + 1]) if len(idx) else 0.0


def load_real(real_out_dir, bag_name):
    with open(Path(real_out_dir) / f"{bag_name}.json") as f:
        return json.load(f)


def _ns_to_ts(ns):
    """ns (int) -> google.protobuf.Timestamp components."""
    return int(ns // 1_000_000_000), int(ns % 1_000_000_000)


def _set_ts(ts_field, ns):
    s, n = _ns_to_ts(ns)
    ts_field.seconds = s
    ts_field.nanos = n


def _set_color(c, r, g, b, a):
    c.r, c.g, c.b, c.a = r, g, b, a


def _gate_square_world(wp_pos, rot, side=GATE_SIDE):
    d = side / 2.0
    local = np.array([[0, d, d], [0, -d, d], [0, -d, -d], [0, d, -d]])
    return rot.apply(local) + wp_pos


def _gate_circle_world(wp_pos, rot, radius=PASS_RADIUS, n=40):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    local = np.stack([np.zeros_like(th), radius * np.cos(th), radius * np.sin(th)], axis=1)
    return rot.apply(local) + wp_pos


def build_gates_scene(publish_ns):
    """Static SceneUpdate with 4 gate frames + detection circles."""
    scene = SceneUpdate()
    for i, wp in enumerate(CIRCLE_WAYPOINTS):
        pos = wp[0:3]
        rot = R.from_euler("xyz", wp[3:6])
        e = scene.entities.add()
        _set_ts(e.timestamp, publish_ns)
        e.frame_id = FRAME_ID
        e.id = f"gate_{i}"
        # frame (LINE_LOOP)
        frame_line = e.lines.add()
        frame_line.type = LinePrimitive.LINE_LOOP
        frame_line.thickness = 0.04
        frame_line.scale_invariant = False
        _set_color(frame_line.color, 0.2, 0.2, 0.2, 1.0)
        for p in _gate_square_world(pos, rot):
            pt = frame_line.points.add()
            pt.x, pt.y, pt.z = float(p[0]), float(p[1]), float(p[2])
        # detection circle (LINE_LOOP, lighter)
        circ_line = e.lines.add()
        circ_line.type = LinePrimitive.LINE_LOOP
        circ_line.thickness = 0.015
        circ_line.scale_invariant = False
        _set_color(circ_line.color, 0.6, 0.6, 0.6, 0.8)
        for p in _gate_circle_world(pos, rot):
            pt = circ_line.points.add()
            pt.x, pt.y, pt.z = float(p[0]), float(p[1]), float(p[2])
    return scene


def build_path_scene(pos_xyz, entity_id, rgba, thickness, publish_ns):
    scene = SceneUpdate()
    e = scene.entities.add()
    _set_ts(e.timestamp, publish_ns)
    e.frame_id = FRAME_ID
    e.id = entity_id
    line = e.lines.add()
    line.type = LinePrimitive.LINE_STRIP
    line.thickness = thickness
    line.scale_invariant = False
    _set_color(line.color, *rgba)
    for p in pos_xyz:
        pt = line.points.add()
        pt.x, pt.y, pt.z = float(p[0]), float(p[1]), float(p[2])
    return scene


def emit_sidecar(bag_dir, pt_path, real_out_dir, logs_root, out_path):
    bag_name = Path(bag_dir).name.replace(" ", "_")
    real = load_real(real_out_dir, bag_name)
    real_t0_epoch = float(real["t0_epoch"])
    real_first_pass = real["gate_passes"][0]["t_rel"] if real["gate_passes"] else 0.0

    pt_md5 = md5_file(pt_path)
    sim_run = find_sim_run_by_md5(pt_md5, logs_root)
    if sim_run is None:
        raise SystemExit(f"no sim run with md5 {pt_md5} under {logs_root}")
    sim = read_play_metrics(sim_run)
    sim_t0 = sim_first_pass_time(sim)

    # Align: publish_epoch[i] = real_start_epoch + real_first_pass + (sim_t[i] - sim_t0)
    # Static topics (gates, path) published at real_start_epoch.
    real_start_ns = int(real_t0_epoch * 1e9)
    align_offset_s = real_first_pass - sim_t0

    print(
        f"[info] {bag_name}: sim_run={sim_run.name} "
        f"real_first_pass={real_first_pass:.2f}s sim_first_pass={sim_t0:.2f}s "
        f"align_offset={align_offset_s:+.2f}s"
    )

    os.makedirs(Path(out_path).parent, exist_ok=True)
    with open(out_path, "wb") as f:
        w = PbWriter(f)

        # --- static: gates ----------------------------------------------------
        gates = build_gates_scene(publish_ns=real_start_ns)
        w.write_message(
            topic="/gates",
            message=gates,
            log_time=real_start_ns,
            publish_time=real_start_ns,
        )

        # --- static: full sim path -------------------------------------------
        sim_path = build_path_scene(
            sim["pos"], entity_id="sim_path",
            rgba=(1.0, 0.3, 0.1, 0.9), thickness=0.03,
            publish_ns=real_start_ns,
        )
        w.write_message(
            topic="/sim/path",
            message=sim_path,
            log_time=real_start_ns,
            publish_time=real_start_ns,
        )

        # --- static: full real path ------------------------------------------
        real_pos = np.array(real["odom"]["pos"])
        real_path = build_path_scene(
            real_pos, entity_id="real_path",
            rgba=(0.1, 0.4, 1.0, 0.9), thickness=0.03,
            publish_ns=real_start_ns,
        )
        w.write_message(
            topic="/real/path",
            message=real_path,
            log_time=real_start_ns,
            publish_time=real_start_ns,
        )

        # --- animated: sim pose + tf per sample ------------------------------
        for i in range(len(sim["t"])):
            t_epoch_s = real_t0_epoch + align_offset_s + float(sim["t"][i])
            t_ns = int(t_epoch_s * 1e9)
            if t_ns < 0:
                continue

            pose = PoseInFrame()
            _set_ts(pose.timestamp, t_ns)
            pose.frame_id = FRAME_ID
            p = sim["pos"][i]
            pose.pose.position.x = float(p[0])
            pose.pose.position.y = float(p[1])
            pose.pose.position.z = float(p[2])
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            w.write_message(topic="/sim/pose", message=pose, log_time=t_ns, publish_time=t_ns)

            tf = FrameTransform()
            _set_ts(tf.timestamp, t_ns)
            tf.parent_frame_id = FRAME_ID
            tf.child_frame_id = "sim_drone"
            tf.translation.x = float(p[0])
            tf.translation.y = float(p[1])
            tf.translation.z = float(p[2])
            tf.rotation.x = 0.0
            tf.rotation.y = 0.0
            tf.rotation.z = 0.0
            tf.rotation.w = 1.0
            w.write_message(topic="/sim/tf", message=tf, log_time=t_ns, publish_time=t_ns)

        w.finish()

    print(f"[info] wrote {out_path} ({len(sim['t'])} sim samples)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", action="append", default=[],
                   help="bag_dir=pt_path. Can repeat.")
    p.add_argument("--auto", action="store_true")
    p.add_argument("--real-out", default="outputs/real_analysis")
    p.add_argument("--logs-root", default="logs/rsl_rl/sim2real_part1")
    p.add_argument("--out", help="output mcap path (only used with single --pair)")
    p.add_argument("--out-dir", default="outputs/real_analysis",
                   help="used with --auto; writes <bag_name>_overlay.mcap per pair")
    args = p.parse_args()

    pair_specs = []
    if args.auto:
        for bag_name, pt_dir in PAIRS.items():
            pt_path = Path("rosbags") / pt_dir / "best_model.pt"
            if not pt_path.exists():
                print(f"[skip] {bag_name}: no pt at {pt_path}")
                continue
            pair_specs.append((f"rosbags/{bag_name}", str(pt_path)))
    for s in args.pair:
        if "=" not in s:
            raise SystemExit(f"--pair expects bag_dir=pt_path, got: {s}")
        a, b = s.split("=", 1)
        pair_specs.append((a, b))
    if not pair_specs:
        raise SystemExit("no pairs; pass --auto or --pair")

    for bag_dir, pt_path in pair_specs:
        bag_name = Path(bag_dir).name.replace(" ", "_")
        if args.out and len(pair_specs) == 1:
            out_path = args.out
        else:
            out_path = str(Path(args.out_dir) / f"{bag_name}_overlay.mcap")
        emit_sidecar(bag_dir, pt_path, args.real_out, args.logs_root, out_path)


if __name__ == "__main__":
    main()
