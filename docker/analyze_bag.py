#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import sys
import glob
import re

try:
    import gnupg
except ImportError:
    import types
    gnupg = types.ModuleType("gnupg")
    sys.modules["gnupg"] = gnupg

import rosbag
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from snapstack_msgs.msg import Goal
from geometry_msgs.msg import PoseStamped


def compute_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def extract_number(filename):
    match = re.search(r'faster_num_(\d+)', filename)
    if match:
        return int(match.group(1))
    return float('inf')


def _trapz_safe(y, t):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    n = min(y.size, t.size)
    if n < 2:
        return 0.0
    if y.size != t.size:
        y = y[:n]
        t = t[:n]

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    mask = np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 2:
        return 0.0

    return float(np.trapz(y[mask], t[mask]))


def _ensure_strictly_increasing(t):
    t = np.asarray(t, dtype=float).copy()
    if t.size == 0:
        return t
    eps = 1e-9
    for k in range(1, t.size):
        if t[k] <= t[k - 1]:
            t[k] = t[k - 1] + eps
    return t


def compute_Jsmooth_and_Seff(times, jerk_norms, jerk_x=None, jerk_y=None, jerk_z=None):
    t = np.asarray(times, dtype=float)
    j = np.asarray(jerk_norms, dtype=float)

    n = min(t.size, j.size)
    if n < 2:
        return {"J_smooth": 0.0, "S_eff": 0.0, "snaps": np.array([])}

    t = t[:n]
    j = j[:n]
    t = _ensure_strictly_increasing(t)

    j2 = j * j
    J2 = _trapz_safe(j2, t)
    T = max(float(t[-1] - t[0]), 1e-12)
    J_smooth = float(np.sqrt(J2 / T))

    if jerk_x is not None and jerk_y is not None and jerk_z is not None:
        jx = np.asarray(jerk_x, dtype=float)[:t.size]
        jy = np.asarray(jerk_y, dtype=float)[:t.size]
        jz = np.asarray(jerk_z, dtype=float)[:t.size]
        try:
            edge = 2 if t.size >= 3 else 1
            sx = np.gradient(jx, t, edge_order=edge)
            sy = np.gradient(jy, t, edge_order=edge)
            sz = np.gradient(jz, t, edge_order=edge)
        except TypeError:
            sx = np.gradient(jx, t)
            sy = np.gradient(jy, t)
            sz = np.gradient(jz, t)
        s2 = sx*sx + sy*sy + sz*sz
        S2 = _trapz_safe(s2, t)
        S_eff = float(np.sqrt(S2 / T))
        snaps = np.sqrt(s2)
    else:
        try:
            edge = 2 if t.size >= 3 else 1
            djdt = np.gradient(j, t, edge_order=edge)
        except TypeError:
            djdt = np.gradient(j, t)
        s2 = djdt * djdt
        S2 = _trapz_safe(s2, t)
        S_eff = float(np.sqrt(S2 / T))
        snaps = np.sqrt(s2)

    return {"J_smooth": J_smooth, "S_eff": S_eff, "snaps": snaps}


def process_bag(bag_file, tol=0.5, v_constraint=10.0, a_constraint=20.0, j_constraint=30.0):
    bag = rosbag.Bag(bag_file)

    goal_time = None
    goal_position = None
    start_time = None
    end_time = None

    pos_cmd_times = []
    positions = []

    # Per-axis logs (for plotting + axis-constraint checking)
    vx_list, vy_list, vz_list = [], [], []
    ax_list, ay_list, az_list = [], [], []
    jx_list, jy_list, jz_list = [], [], []

    # Norm logs (optional; keep for smoothness metrics convenience)
    jerk_norms = []

    initial_pos_ref = None

    vel_violations = 0
    acc_violations = 0
    jerk_violations = 0
    total_cmds = 0

    perct = 0.01
    v_thresh = v_constraint * (1.0 + perct)
    a_thresh = a_constraint * (1.0 + perct)
    j_thresh = j_constraint * (1.0 + perct)

    print("Processing bag: {0}".format(bag_file))

    for topic, msg, t in bag.read_messages(topics=["/SQ01s/faster/term_goal", "/SQ01s/goal"]):
        if topic == "/SQ01s/faster/term_goal" and goal_time is None:
            goal_time = t.to_sec()
            goal_position = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
            print("  Found /SQ01s/faster/term_goal at time {0:.3f}, goal_position = {1}".format(goal_time, goal_position))
            continue

        if topic != "/SQ01s/goal" or goal_time is None:
            continue

        pos_time = t.to_sec()
        if pos_time < goal_time:
            continue

        pos = (msg.p.x, msg.p.y, msg.p.z)

        if start_time is None:
            if initial_pos_ref is None:
                initial_pos_ref = pos
                continue
            if compute_distance(pos, initial_pos_ref) <= tol:
                continue
            start_time = pos_time
            print("  Start of travel detected at time {0:.3f}".format(start_time))

        total_cmds += 1
        pos_cmd_times.append(pos_time)
        positions.append(pos)

        vx, vy, vz = float(msg.v.x), float(msg.v.y), float(msg.v.z)
        ax, ay, az = float(msg.a.x), float(msg.a.y), float(msg.a.z)
        jx, jy, jz = float(msg.j.x), float(msg.j.y), float(msg.j.z)

        vx_list.append(vx); vy_list.append(vy); vz_list.append(vz)
        ax_list.append(ax); ay_list.append(ay); az_list.append(az)
        jx_list.append(jx); jy_list.append(jy); jz_list.append(jz)

        jnorm = float(np.linalg.norm([jx, jy, jz]))
        jerk_norms.append(jnorm)

        # PER-AXIS violations (any axis exceeds)
        if (abs(vx) > v_thresh) or (abs(vy) > v_thresh) or (abs(vz) > v_thresh):
            vel_violations += 1
        if (abs(ax) > a_thresh) or (abs(ay) > a_thresh) or (abs(az) > a_thresh):
            acc_violations += 1
        if (abs(jx) > j_thresh) or (abs(jy) > j_thresh) or (abs(jz) > j_thresh):
            jerk_violations += 1

        if compute_distance(pos, goal_position) <= tol:
            end_time = pos_time
            print("  Goal reached at time {0:.3f}".format(end_time))
            break

    bag.close()

    if goal_time is None or goal_position is None:
        print("  Missing /SQ01s/faster/term_goal.")
        return None
    if start_time is None or end_time is None:
        print("  Could not compute travel time (no start or no finish).")
        return None
    if end_time <= start_time:
        print("  Invalid travel interval.")
        return None

    travel_time = float(end_time - start_time)

    path_length = 0.0
    if len(positions) >= 2:
        for i in range(len(positions) - 1):
            path_length += compute_distance(positions[i], positions[i + 1])

    t_arr = np.asarray(pos_cmd_times, dtype=float)
    j_arr = np.asarray(jerk_norms, dtype=float)

    smoothness_l1 = _trapz_safe(j_arr, t_arr)
    metrics = compute_Jsmooth_and_Seff(
        times=t_arr,
        jerk_norms=j_arr,
        jerk_x=np.asarray(jx_list, dtype=float),
        jerk_y=np.asarray(jy_list, dtype=float),
        jerk_z=np.asarray(jz_list, dtype=float)
    )

    if total_cmds > 0:
        v_pct = (float(vel_violations) / float(total_cmds)) * 100.0
        a_pct = (float(acc_violations) / float(total_cmds)) * 100.0
        j_pct = (float(jerk_violations) / float(total_cmds)) * 100.0
    else:
        v_pct = a_pct = j_pct = 0.0

    return {
        "travel_time": travel_time,
        "path_length": float(path_length),
        "pos_cmd_times": t_arr,

        # per-axis arrays
        "vx": np.asarray(vx_list, dtype=float),
        "vy": np.asarray(vy_list, dtype=float),
        "vz": np.asarray(vz_list, dtype=float),
        "ax": np.asarray(ax_list, dtype=float),
        "ay": np.asarray(ay_list, dtype=float),
        "az": np.asarray(az_list, dtype=float),
        "jx": np.asarray(jx_list, dtype=float),
        "jy": np.asarray(jy_list, dtype=float),
        "jz": np.asarray(jz_list, dtype=float),

        # smoothness / snap
        "jerk_norms": j_arr,
        "snaps": metrics["snaps"],
        "smoothness_l1": float(smoothness_l1),
        "J_smooth": float(metrics["J_smooth"]),
        "S_eff": float(metrics["S_eff"]),

        # violations
        "vel_violations": int(vel_violations),
        "acc_violations": int(acc_violations),
        "jerk_violations": int(jerk_violations),
        "total_cmds": int(total_cmds),
        "vel_violate_pct": float(v_pct),
        "acc_violate_pct": float(a_pct),
        "jerk_violate_pct": float(j_pct),
    }


def save_plots(bag_file, results, v_constraint, a_constraint, j_constraint, show_snap=True):
    base_name = os.path.splitext(os.path.basename(bag_file))[0]
    folder = os.path.dirname(bag_file)

    # Histogram of per-axis magnitudes for velocity
    plt.figure()
    plt.hist(np.abs(results["vx"]), bins=20, alpha=0.6, label="|vx|")
    plt.hist(np.abs(results["vy"]), bins=20, alpha=0.6, label="|vy|")
    plt.hist(np.abs(results["vz"]), bins=20, alpha=0.6, label="|vz|")
    plt.axvline(x=v_constraint, color="red", linestyle="--",
                label="v_limit (axis) = {0} m/s".format(v_constraint))
    plt.xlabel("Axis speed magnitude (m/s)")
    plt.ylabel("Frequency")
    plt.title("Velocity Per-Axis Histogram")
    plt.legend()
    plt.grid(True)
    hist_path = os.path.join(folder, "{0}_velocity_axis_hist.pdf".format(base_name))
    plt.savefig(hist_path)
    plt.close()

    t = results["pos_cmd_times"]

    vx, vy, vz = results["vx"], results["vy"], results["vz"]
    ax, ay, az = results["ax"], results["ay"], results["az"]
    jx, jy, jz = results["jx"], results["jy"], results["jz"]
    s = results.get("snaps", None)

    use_snap = bool(show_snap and s is not None and getattr(s, "size", 0) == getattr(t, "size", 0))
    rows = 4 if use_snap else 3
    plt.figure(figsize=(12, 12 if use_snap else 10))

    # Velocity per-axis
    plt.subplot(rows, 1, 1)
    plt.plot(t, vx, label="vx")
    plt.plot(t, vy, label="vy")
    plt.plot(t, vz, label="vz")
    plt.axhline(y=v_constraint, color="red", linestyle="--", label="+v_limit")
    plt.axhline(y=-v_constraint, color="red", linestyle="--", label="-v_limit")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid(True)

    # Acceleration per-axis
    plt.subplot(rows, 1, 2)
    plt.plot(t, ax, label="ax")
    plt.plot(t, ay, label="ay")
    plt.plot(t, az, label="az")
    plt.axhline(y=a_constraint, color="red", linestyle="--", label="+a_limit")
    plt.axhline(y=-a_constraint, color="red", linestyle="--", label="-a_limit")
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()
    plt.grid(True)

    # Jerk per-axis
    plt.subplot(rows, 1, 3)
    plt.plot(t, jx, label="jx")
    plt.plot(t, jy, label="jy")
    plt.plot(t, jz, label="jz")
    plt.axhline(y=j_constraint, color="red", linestyle="--", label="+j_limit")
    plt.axhline(y=-j_constraint, color="red", linestyle="--", label="-j_limit")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (m/s^3)")
    plt.legend()
    plt.grid(True)

    if use_snap:
        plt.subplot(rows, 1, 4)
        plt.plot(t, s, label="Snap (m/s^4)")
        plt.xlabel("Time (s)")
        plt.ylabel("Snap (m/s^4)")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    suffix = "_vel_accel_jerk_axes_snap.pdf" if use_snap else "_vel_accel_jerk_axes.pdf"
    time_history_path = os.path.join(folder, "{0}{1}".format(base_name, suffix))
    plt.savefig(time_history_path)
    plt.close()


def main():
    if len(sys.argv) < 6:
        print("Usage: {0} <bag_folder> <tolerance (m)> <v_max_axis> <a_max_axis> <j_max_axis>".format(sys.argv[0]))
        sys.exit(1)

    bag_folder = sys.argv[1]
    tol = float(sys.argv[2])
    v_constraint = float(sys.argv[3])
    a_constraint = float(sys.argv[4])
    j_constraint = float(sys.argv[5])

    bag_files = glob.glob(os.path.join(bag_folder, "*.bag"))
    if not bag_files:
        print("No bag files found in folder: {0}".format(bag_folder))
        sys.exit(1)

    bag_files = sorted(bag_files, key=lambda f: extract_number(os.path.basename(f)))

    processed_count = 0
    stats_lines = []
    stats_lines.append("Bag File Statistics:\n\n")
    stats_lines.append("Note: constraint violations are PER-AXIS with 1% slack.\n\n")

    for bag_file in bag_files:
        result = process_bag(bag_file, tol, v_constraint, a_constraint, j_constraint)
        if result is None:
            stats_lines.append("{0}: Could not compute travel time (no start/finish)\n\n".format(
                os.path.basename(bag_file)))
            continue

        processed_count += 1

        stats_lines.append("{0}:\n".format(os.path.basename(bag_file)))
        stats_lines.append("  Travel time: {0:.3f} s\n".format(result["travel_time"]))
        stats_lines.append("  Path length: {0:.3f} m\n".format(result["path_length"]))
        stats_lines.append("  Smoothness (∫||jerk|| dt): {0:.6f} m/s^2\n".format(result["smoothness_l1"]))
        stats_lines.append("  J_smooth (RMS jerk): {0:.6f} m/s^3\n".format(result["J_smooth"]))
        stats_lines.append("  S_eff (RMS snap): {0:.6f} m/s^4\n".format(result["S_eff"]))
        stats_lines.append("  Velocity violations (|v_axis|>{0}, +1% slack): {1} ({2:.2f}%)\n".format(
            v_constraint, result["vel_violations"], result["vel_violate_pct"]))
        stats_lines.append("  Acceleration violations (|a_axis|>{0}, +1% slack): {1} ({2:.2f}%)\n".format(
            a_constraint, result["acc_violations"], result["acc_violate_pct"]))
        stats_lines.append("  Jerk violations (|j_axis|>{0}, +1% slack): {1} ({2:.2f}%)\n\n".format(
            j_constraint, result["jerk_violations"], result["jerk_violate_pct"]))

        save_plots(bag_file, result, v_constraint, a_constraint, j_constraint, show_snap=True)

    stats_file = os.path.join(bag_folder, "faster_statistics.txt")
    f = open(stats_file, "w")
    try:
        f.writelines(stats_lines)
    finally:
        f.close()

    print("Saved overall statistics to: {0}".format(stats_file))


if __name__ == "__main__":
    main()
