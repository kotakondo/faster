#!/usr/bin/env python3
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
import matplotlib.pyplot as plt

# Import custom PositionCommand message.
from snapstack_msgs.msg import Goal
from geometry_msgs.msg import PoseStamped

def compute_distance(p1, p2):
    """Compute Euclidean distance between two 3D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def extract_number(filename):
    """Extract numeric part from filename using regex.
       Assumes filename contains 'faster_num_<number>'. Returns the integer number.
    """
    match = re.search(r'faster_num_(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return float('inf')  # Place files with no number at the end

def process_bag(bag_file, tol=0.5, v_constraint=10.0, a_constraint=20.0, j_constraint=30.0):
    """
    Process a single bag file.
    
    It reads /SQ01s/faster/term_goal and /SQ01s/goal topics.
    Returns a dictionary with:
      - travel_time (float)
      - path_length (float) : total path length (m) computed from consecutive positions
      - pos_cmd_times (np.array of times)
      - velocities (np.array of float)
      - accelerations (np.array of float)
      - jerks (np.array of float)
      - vel_violations, acc_violations, jerk_violations (int counts)
    Returns None if travel time could not be computed.
    """
    bag = rosbag.Bag(bag_file)
    goal_time = None
    goal_position = None
    travel_end_time = None

    pos_cmd_times = []
    velocities = []
    accelerations = []
    jerks = []
    positions = []  # List to record positions for path length computation

    # Violation counts
    vel_violations = 0
    acc_violations = 0
    jerk_violations = 0

    print("Processing bag: {}".format(bag_file))
    for topic, msg, t in bag.read_messages(topics=["/SQ01s/faster/term_goal", "/SQ01s/goal"]):
        if topic == "/SQ01s/faster/term_goal" and goal_time is None:
            goal_time = t.to_sec()
            goal_position = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
            print("  Found /SQ01s/faster/term_goal at time {:.3f}, goal_position = {}".format(goal_time, goal_position))
        elif topic == "/SQ01s/goal" and goal_time is not None:
            pos_time = t.to_sec()
            if pos_time < goal_time:
                continue

            pos_cmd_times.append(pos_time)
            # Record the current commanded position.
            pos = (msg.p.x, msg.p.y, msg.p.z)
            positions.append(pos)

            # Compute norms of the command vectors.
            vel = np.linalg.norm([msg.v.x, msg.v.y, msg.v.z])
            acc = np.linalg.norm([msg.a.x, msg.a.y, msg.a.z])
            jrk = np.linalg.norm([msg.j.x, msg.j.y, msg.j.z])
            velocities.append(vel)
            accelerations.append(acc)
            jerks.append(jrk)

            if vel > v_constraint:
                vel_violations += 1
            if acc > a_constraint:
                acc_violations += 1
            if jrk > j_constraint:
                jerk_violations += 1

            # Check if goal is reached.
            if compute_distance(pos, goal_position) <= tol:
                travel_end_time = pos_time
                print("  Goal reached at time {:.3f}".format(travel_end_time))
                break

    bag.close()

    if goal_time is None or travel_end_time is None:
        print("  Could not compute travel time for this bag.")
        return None

    travel_time = travel_end_time - goal_time

    # Compute total path length (sum of distances between consecutive positions).
    path_length = 0.0
    if len(positions) >= 2:
        for i in range(len(positions) - 1):
            path_length += compute_distance(positions[i], positions[i+1])

    result = {
        "travel_time": travel_time,
        "path_length": path_length,
        "pos_cmd_times": np.array(pos_cmd_times),
        "velocities": np.array(velocities),
        "accelerations": np.array(accelerations),
        "jerks": np.array(jerks),
        "vel_violations": vel_violations,
        "acc_violations": acc_violations,
        "jerk_violations": jerk_violations,
    }
    return result

def save_plots(bag_file, results):
    """
    Generate and save two plots:
      1. Velocity profile histogram saved as <bag_name>_velocity_profile.pdf
      2. Time history plot (velocity, acceleration, jerk) saved as <bag_name>_vel_accel_jerk.pdf
    """
    base_name = os.path.splitext(os.path.basename(bag_file))[0]
    folder = os.path.dirname(bag_file)
    v_constraint = 10.0
    a_constraint = 20.0
    j_constraint = 30.0

    # Plot 1: Velocity histogram.
    plt.figure()
    plt.hist(results["velocities"], bins=20, edgecolor="black")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Frequency")
    plt.title("Velocity Profile Histogram")
    plt.axvline(x=v_constraint, color="red", linestyle="--", label=f"v constraint = {v_constraint} m/s")
    plt.legend()
    plt.grid(True)
    hist_path = os.path.join(folder, f"{base_name}_velocity_profile.pdf")
    plt.savefig(hist_path)
    plt.close()
    print("  Saved velocity histogram to:", hist_path)

    # Plot 2: Time history for velocity, acceleration, and jerk.
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(results["pos_cmd_times"], results["velocities"], label="Velocity (m/s)")
    plt.axhline(y=v_constraint, color="red", linestyle="--", label=f"v constraint = {v_constraint} m/s")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(results["pos_cmd_times"], results["accelerations"], label="Acceleration (m/s²)")
    plt.axhline(y=a_constraint, color="red", linestyle="--", label=f"a constraint = {a_constraint} m/s²")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(results["pos_cmd_times"], results["jerks"], label="Jerk (m/s³)")
    plt.axhline(y=j_constraint, color="red", linestyle="--", label=f"j constraint = {j_constraint} m/s³")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (m/s³)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    time_history_path = os.path.join(folder, f"{base_name}_vel_accel_jerk.pdf")
    plt.savefig(time_history_path)
    plt.close()
    print("  Saved vel/accel/jerk time history plot to:", time_history_path)

def main():
    if len(sys.argv) < 5:
        print("Usage: {} <bag_folder> [tolerance (m)] <v_max> <a_max> <j_max>".format(sys.argv[0]))
        sys.exit(1)

    bag_folder = sys.argv[1]
    tol = 0.5
    if len(sys.argv) > 2:
        tol = float(sys.argv[2])
    v_constraint = float(sys.argv[3])
    a_constraint = float(sys.argv[4])
    j_constraint = float(sys.argv[5])

    bag_files = glob.glob(os.path.join(bag_folder, "*.bag"))
    if not bag_files:
        print("No bag files found in folder:", bag_folder)
        sys.exit(1)

    # Sort the bag files based on the numeric value in their names.
    bag_files = sorted(bag_files, key=lambda f: extract_number(os.path.basename(f)))

    overall_travel_times = []
    overall_path_lengths = []
    overall_vel_violations = 0
    overall_acc_violations = 0
    overall_jerk_violations = 0
    processed_count = 0

    stats_lines = []
    stats_lines.append("Bag File Statistics:\n\n")

    for bag_file in bag_files:
        result = process_bag(bag_file, tol, v_constraint, a_constraint, j_constraint)
        if result is None:
            stats_lines.append(f"{os.path.basename(bag_file)}: Could not compute travel time (missing /goal or goal reached)\n\n")
            continue

        processed_count += 1
        overall_travel_times.append(result["travel_time"])
        overall_path_lengths.append(result["path_length"])
        overall_vel_violations += result["vel_violations"]
        overall_acc_violations += result["acc_violations"]
        overall_jerk_violations += result["jerk_violations"]

        stats_lines.append(f"{os.path.basename(bag_file)}:\n")
        stats_lines.append(f"  Travel time: {result['travel_time']:.3f} s\n")
        stats_lines.append(f"  Path length: {result['path_length']:.3f} m\n")
        stats_lines.append(f"  Velocity violations (>10 m/s): {result['vel_violations']}\n")
        stats_lines.append(f"  Acceleration violations (>20 m/s²): {result['acc_violations']}\n")
        stats_lines.append(f"  Jerk violations (>30 m/s³): {result['jerk_violations']}\n\n")

        # Save the plots for this bag.
        save_plots(bag_file, result)

    if processed_count > 0:
        avg_travel_time = np.mean(overall_travel_times)
        avg_path_length = np.mean(overall_path_lengths)
        stats_lines.append("Overall Statistics:\n")
        stats_lines.append(f"  Processed bag files: {processed_count}\n")
        stats_lines.append(f"  Average travel time: {avg_travel_time:.3f} s\n")
        stats_lines.append(f"  Average path length: {avg_path_length:.3f} m\n")
        stats_lines.append(f"  Total velocity violations: {overall_vel_violations}\n")
        stats_lines.append(f"  Total acceleration violations: {overall_acc_violations}\n")
        stats_lines.append(f"  Total jerk violations: {overall_jerk_violations}\n")
    else:
        stats_lines.append("No valid travel times computed from the bag files.\n")

    stats_file = os.path.join(bag_folder, "faster_statistics.txt")
    with open(stats_file, "w") as f:
        f.writelines(stats_lines)
    print("Saved overall statistics to:", stats_file)

if __name__ == "__main__":
    main()
