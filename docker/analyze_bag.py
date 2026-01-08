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


# ------------------------- J_smooth helpers (Py2) -------------------------
def _trapz_safe(y, t):
    """Trapezoidal integral of y wrt t with basic safety checks."""
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
    """Nudge non-increasing timestamps forward by tiny epsilon (copy)."""
    t = np.asarray(t, dtype=float).copy()
    if t.size == 0:
        return t
    eps = 1e-9
    for k in range(1, t.size):
        if t[k] <= t[k - 1]:
            t[k] = t[k - 1] + eps
    return t


def compute_Jsmooth_and_Seff(times, jerk_norms, jerk_x=None, jerk_y=None, jerk_z=None):
    """
    Compute:
      - J_smooth = sqrt((1/T) * ∫ ||jerk||^2 dt)  [m/s^3]
      - S_eff    = sqrt((1/T) * ∫ ||snap||^2 dt)  [m/s^4], snap = d(jerk)/dt
    If jerk components are unavailable, we compute snap from ||jerk|| only (approx),
    but best is to pass jerk_x/y/z when you can.
    """
    t = np.asarray(times, dtype=float)
    j = np.asarray(jerk_norms, dtype=float)

    n = min(t.size, j.size)
    if n < 2:
        return {"J_smooth": 0.0, "S_eff": 0.0, "snaps": np.array([])}

    t = t[:n]
    j = j[:n]
    t = _ensure_strictly_increasing(t)

    # RMS jerk
    j2 = j * j
    J2 = _trapz_safe(j2, t)
    T = max(float(t[-1] - t[0]), 1e-12)
    J_smooth = float(np.sqrt(J2 / T))

    # RMS snap: prefer vector jerk if provided; else use d/dt(||j||) as an approximation
    if jerk_x is not None and jerk_y is not None and jerk_z is not None:
        jx = np.asarray(jerk_x, dtype=float)[:t.size]
        jy = np.asarray(jerk_y, dtype=float)[:t.size]
        jz = np.asarray(jerk_z, dtype=float)[:t.size]

        # np.gradient supports edge_order in newer numpy; guard for older versions
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
        # Fallback: snap approx from ||j|| only
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
    """
    Reads:
      /SQ01s/faster/term_goal  (PoseStamped-like with msg.pose.position)
      /SQ01s/goal              (Goal-like with msg.p, msg.v, msg.a, msg.j)

    Computes:
      travel_time, path_length,
      smoothness_l1 = ∫||jerk|| dt,
      J_smooth (RMS jerk), S_eff (RMS snap),
      plus violation counts.
    """
    bag = rosbag.Bag(bag_file)
    goal_time = None
    goal_position = None
    travel_end_time = None

    pos_cmd_times = []
    velocities = []
    accelerations = []
    jerks = []
    positions = []

    # If you want snap from vector jerk, keep components too
    jerk_x, jerk_y, jerk_z = [], [], []

    vel_violations = 0
    acc_violations = 0
    jerk_violations = 0

    print("Processing bag: {0}".format(bag_file))

    for topic, msg, t in bag.read_messages(topics=["/SQ01s/faster/term_goal", "/SQ01s/goal"]):
        if topic == "/SQ01s/faster/term_goal" and goal_time is None:
            goal_time = t.to_sec()
            goal_position = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
            print("  Found /SQ01s/faster/term_goal at time {0:.3f}, goal_position = {1}".format(goal_time, goal_position))

        elif topic == "/SQ01s/goal" and goal_time is not None:
            pos_time = t.to_sec()
            if pos_time < goal_time:
                continue

            pos_cmd_times.append(pos_time)

            pos = (msg.p.x, msg.p.y, msg.p.z)
            positions.append(pos)

            vel = float(np.linalg.norm([msg.v.x, msg.v.y, msg.v.z]))
            acc = float(np.linalg.norm([msg.a.x, msg.a.y, msg.a.z]))

            # jerk vector + norm
            jx = float(msg.j.x)
            jy = float(msg.j.y)
            jz = float(msg.j.z)
            jrk = float(np.linalg.norm([jx, jy, jz]))

            velocities.append(vel)
            accelerations.append(acc)
            jerks.append(jrk)
            jerk_x.append(jx); jerk_y.append(jy); jerk_z.append(jz)

            if vel > v_constraint:
                vel_violations += 1
            if acc > a_constraint:
                acc_violations += 1
            if jrk > j_constraint:
                jerk_violations += 1

            if compute_distance(pos, goal_position) <= tol:
                travel_end_time = pos_time
                print("  Goal reached at time {0:.3f}".format(travel_end_time))
                break

    bag.close()

    if goal_time is None or travel_end_time is None:
        print("  Could not compute travel time for this bag.")
        return None

    travel_time = float(travel_end_time - goal_time)

    # Path length
    path_length = 0.0
    if len(positions) >= 2:
        for i in range(len(positions) - 1):
            path_length += compute_distance(positions[i], positions[i + 1])

    # --- Smoothness metrics ---
    t_arr = np.asarray(pos_cmd_times, dtype=float)
    j_arr = np.asarray(jerks, dtype=float)

    # Legacy L1 jerk: ∫ ||j|| dt
    smoothness_l1 = _trapz_safe(j_arr, t_arr)

    # RMS jerk and RMS snap
    metrics = compute_Jsmooth_and_Seff(
        times=t_arr,
        jerk_norms=j_arr,
        jerk_x=np.asarray(jerk_x, dtype=float),
        jerk_y=np.asarray(jerk_y, dtype=float),
        jerk_z=np.asarray(jerk_z, dtype=float)
    )
    J_smooth = metrics["J_smooth"]
    S_eff = metrics["S_eff"]
    snaps = metrics["snaps"]

    return {
        "travel_time": travel_time,
        "path_length": float(path_length),
        "pos_cmd_times": t_arr,
        "velocities": np.asarray(velocities, dtype=float),
        "accelerations": np.asarray(accelerations, dtype=float),
        "jerks": j_arr,
        "snaps": snaps,
        "smoothness_l1": float(smoothness_l1),
        "J_smooth": float(J_smooth),
        "S_eff": float(S_eff),
        "vel_violations": int(vel_violations),
        "acc_violations": int(acc_violations),
        "jerk_violations": int(jerk_violations),
    }


def save_plots(bag_file, results, v_constraint, a_constraint, j_constraint, show_snap=True):
    base_name = os.path.splitext(os.path.basename(bag_file))[0]
    folder = os.path.dirname(bag_file)

    # Histogram
    plt.figure()
    plt.hist(results["velocities"], bins=20, edgecolor="black")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Frequency")
    plt.title("Velocity Profile Histogram")
    plt.axvline(x=v_constraint, color="red", linestyle="--",
                label="v constraint = {0} m/s".format(v_constraint))
    plt.legend()
    plt.grid(True)
    hist_path = os.path.join(folder, "{0}_velocity_profile.pdf".format(base_name))
    plt.savefig(hist_path)
    plt.close()
    print("  Saved velocity histogram to: {0}".format(hist_path))

    # Time histories
    t = results["pos_cmd_times"]
    v = results["velocities"]
    a = results["accelerations"]
    j = results["jerks"]
    s = results.get("snaps", None)

    use_snap = bool(show_snap and s is not None and getattr(s, "size", 0) == getattr(t, "size", 0))
    rows = 4 if use_snap else 3
    plt.figure(figsize=(12, 12 if use_snap else 10))

    plt.subplot(rows, 1, 1)
    plt.plot(t, v, label="Velocity (m/s)")
    plt.axhline(y=v_constraint, color="red", linestyle="--",
                label="v constraint = {0} m/s".format(v_constraint))
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid(True)

    plt.subplot(rows, 1, 2)
    plt.plot(t, a, label="Acceleration (m/s^2)")
    plt.axhline(y=a_constraint, color="red", linestyle="--",
                label="a constraint = {0} m/s^2".format(a_constraint))
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()
    plt.grid(True)

    plt.subplot(rows, 1, 3)
    plt.plot(t, j, label="Jerk (m/s^3)")
    plt.axhline(y=j_constraint, color="red", linestyle="--",
                label="j constraint = {0} m/s^3".format(j_constraint))
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
    suffix = "_vel_accel_jerk_snap.pdf" if use_snap else "_vel_accel_jerk.pdf"
    time_history_path = os.path.join(folder, "{0}{1}".format(base_name, suffix))
    plt.savefig(time_history_path)
    plt.close()
    print("  Saved time history plot to: {0}".format(time_history_path))


def main():
    # script bag_folder tol v_max a_max j_max
    if len(sys.argv) < 6:
        print("Usage: {0} <bag_folder> <tolerance (m)> <v_max> <a_max> <j_max>".format(sys.argv[0]))
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

    overall_travel_times = []
    overall_path_lengths = []
    overall_smoothness_l1 = []
    overall_J_smooth = []
    overall_S_eff = []

    overall_vel_violations = 0
    overall_acc_violations = 0
    overall_jerk_violations = 0
    processed_count = 0

    stats_lines = []
    stats_lines.append("Bag File Statistics:\n\n")

    for bag_file in bag_files:
        result = process_bag(bag_file, tol, v_constraint, a_constraint, j_constraint)
        if result is None:
            stats_lines.append("{0}: Could not compute travel time (missing /goal or goal reached)\n\n".format(
                os.path.basename(bag_file)))
            continue

        processed_count += 1
        overall_travel_times.append(result["travel_time"])
        overall_path_lengths.append(result["path_length"])
        overall_smoothness_l1.append(result["smoothness_l1"])
        overall_J_smooth.append(result["J_smooth"])
        overall_S_eff.append(result["S_eff"])

        overall_vel_violations += result["vel_violations"]
        overall_acc_violations += result["acc_violations"]
        overall_jerk_violations += result["jerk_violations"]

        stats_lines.append("{0}:\n".format(os.path.basename(bag_file)))
        stats_lines.append("  Travel time: {0:.3f} s\n".format(result["travel_time"]))
        stats_lines.append("  Path length: {0:.3f} m\n".format(result["path_length"]))
        stats_lines.append("  Smoothness (∫||jerk|| dt): {0:.6f} m/s^2\n".format(result["smoothness_l1"]))
        stats_lines.append("  J_smooth (RMS jerk): {0:.6f} m/s^3\n".format(result["J_smooth"]))
        stats_lines.append("  S_eff (RMS snap): {0:.6f} m/s^4\n".format(result["S_eff"]))
        stats_lines.append("  Velocity violations (>{0} m/s): {1}\n".format(v_constraint, result["vel_violations"]))
        stats_lines.append("  Acceleration violations (>{0} m/s^2): {1}\n".format(a_constraint, result["acc_violations"]))
        stats_lines.append("  Jerk violations (>{0} m/s^3): {1}\n\n".format(j_constraint, result["jerk_violations"]))

        save_plots(bag_file, result, v_constraint, a_constraint, j_constraint, show_snap=True)

    if processed_count > 0:
        avg_travel_time = float(np.mean(overall_travel_times))
        avg_path_length = float(np.mean(overall_path_lengths))
        avg_smoothness_l1 = float(np.mean(overall_smoothness_l1)) if overall_smoothness_l1 else 0.0
        avg_J_smooth = float(np.mean(overall_J_smooth)) if overall_J_smooth else 0.0
        avg_S_eff = float(np.mean(overall_S_eff)) if overall_S_eff else 0.0

        stats_lines.append("Overall Statistics:\n")
        stats_lines.append("  Processed bag files: {0}\n".format(processed_count))
        stats_lines.append("  Average travel time: {0:.3f} s\n".format(avg_travel_time))
        stats_lines.append("  Average path length: {0:.3f} m\n".format(avg_path_length))
        stats_lines.append("  Average smoothness (∫||jerk|| dt): {0:.6f} m/s^2\n".format(avg_smoothness_l1))
        stats_lines.append("  Average J_smooth (RMS jerk): {0:.6f} m/s^3\n".format(avg_J_smooth))
        stats_lines.append("  Average S_eff (RMS snap): {0:.6f} m/s^4\n".format(avg_S_eff))
        stats_lines.append("  Total velocity violations: {0}\n".format(overall_vel_violations))
        stats_lines.append("  Total acceleration violations: {0}\n".format(overall_acc_violations))
        stats_lines.append("  Total jerk violations: {0}\n".format(overall_jerk_violations))
    else:
        stats_lines.append("No valid travel times computed from the bag files.\n")

    stats_file = os.path.join(bag_folder, "faster_statistics.txt")
    f = open(stats_file, "w")
    try:
        f.writelines(stats_lines)
    finally:
        f.close()

    print("Saved overall statistics to: {0}".format(stats_file))


if __name__ == "__main__":
    main()
