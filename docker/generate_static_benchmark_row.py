#!/usr/bin/env python3
"""
Generate/update the FASTER row in the static_benchmark.tex LaTeX table.

Reads benchmark CSV outputs and updates ONLY the FASTER row, leaving all
other rows (EGO-Swarm2, SUPER, DYNUS) untouched.

Usage:
    python3 generate_static_benchmark_row.py \
        --data-dir ../data \
        --latex-table /home/kkondo/paper_writing/DYNUS_v3/tables/static_benchmark.tex
"""

import argparse
import csv
import os
import sys


def read_overall_summary(data_dir):
    """Read faster_overall_summary.csv and return dict of metrics."""
    csv_path = os.path.join(data_dir, "faster_overall_summary.csv")
    if not os.path.exists(csv_path):
        print("ERROR: {} not found".format(csv_path))
        sys.exit(1)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) != 1:
        print("ERROR: Expected 1 row in overall summary, got {}".format(len(rows)))
        sys.exit(1)

    row = rows[0]
    return {
        "travel_time_mean_s": float(row["travel_time_mean_s"]),
        "path_length_mean_m": float(row["path_length_mean_m"]),
        "smoothness_l1_mean": float(row["smoothness_l1_mean"]),
        "vel_violations_pct_overall": float(row["vel_violations_pct_overall"]),
        "acc_violations_pct_overall": float(row["acc_violations_pct_overall"]),
        "jerk_violations_pct_overall": float(row["jerk_violations_pct_overall"]),
    }


def read_computation_times(data_dir):
    """Read average_computation_times.csv and return dict of metrics.

    The CSV has a non-standard format:
      Line 1: title
      Line 2: Folder,...
      Line 3: NaN handling,...
      Line 4: empty
      Line 5: header row
      Lines 6-N: per-file rows
      Line N+1: empty
      Line N+2: Overall row

    We parse the Overall row for weighted averages.
    """
    csv_path = os.path.join(data_dir, "average_computation_times.csv")
    if not os.path.exists(csv_path):
        print("ERROR: {} not found".format(csv_path))
        sys.exit(1)

    with open(csv_path, "r") as f:
        lines = f.readlines()

    # Find the header and "Overall" lines
    overall_line = None
    header_line = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("file_name,"):
            header_line = stripped
        if stripped.startswith("Overall"):
            overall_line = stripped
            break

    if overall_line is None or header_line is None:
        print("ERROR: Could not find Overall row or header in {}".format(csv_path))
        sys.exit(1)

    # Parse header to find column indices
    header_fields = [h.strip() for h in header_line.split(",")]
    overall_fields = [f.strip() for f in overall_line.split(",")]

    # Map header names to indices
    idx_map = {}
    for i, name in enumerate(header_fields):
        idx_map[name] = i

    def safe_float(s):
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    total_replan = safe_float(overall_fields[idx_map["avg_total_replan_ms"]])
    total_whole = safe_float(overall_fields[idx_map["avg_total_whole_ms"]])
    total_safe = safe_float(overall_fields[idx_map["avg_total_safe_ms"]])

    return {
        "total_replan_ms": total_replan,
        "total_local_whole_ms": total_whole,
        "total_local_safe_ms": total_safe,
    }


def read_goal_reached(data_dir):
    """Read goal_reached_status.csv and compute success rate."""
    csv_path = os.path.join(data_dir, "goal_reached_status.csv")
    if not os.path.exists(csv_path):
        print("ERROR: {} not found".format(csv_path))
        sys.exit(1)

    total = 0
    reached = 0
    reached_bags = set()
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row["status"].strip() == "reached":
                reached += 1
                # Build bag filename from sim_num to cross-reference with collision data
                sim_num = row.get("sim_num", "").strip()
                if sim_num:
                    reached_bags.add("faster_num_{}.bag".format(sim_num))

    if total == 0:
        print("ERROR: No rows in goal_reached_status.csv")
        sys.exit(1)

    return {
        "total": total,
        "reached": reached,
        "reached_bags": reached_bags,
        "success_rate_pct": (float(reached) / float(total)) * 100.0,
    }


def read_collision_check(data_dir):
    """Read collision_check.csv and return per-bag collision status."""
    csv_path = os.path.join(data_dir, "collision_check.csv")
    if not os.path.exists(csv_path):
        print("WARNING: {} not found, skipping collision cross-check".format(csv_path))
        return {"total": 0, "collisions": 0, "collision_free": True, "collided_bags": set()}

    total = 0
    collisions = 0
    collided_bags = set()
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row["collision_status"].strip() == "collision":
                collisions += 1
                collided_bags.add(row["bag_file"].strip())

    return {
        "total": total,
        "collisions": collisions,
        "collision_free": collisions == 0,
        "collided_bags": collided_bags,
    }


TIMEOUT_SECONDS = 50.0


def read_per_bag_summary(data_dir):
    """Read faster_per_bag_summary.csv and return list of per-bag dicts."""
    csv_path = os.path.join(data_dir, "faster_per_bag_summary.csv")
    if not os.path.exists(csv_path):
        print("WARNING: {} not found, skipping per-bag timeout filter".format(csv_path))
        return []

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


ENV_DISPLAY_NAMES = {
    "easy_forest": "Easy",
    "medium_forest": "Medium",
    "hard_forest": "Hard",
}


def format_faster_row(summary, comp_times, goal_status, collision_info,
                      per_bag_rows=None, env_name=None, row_label="FASTER (CP)"):
    """Format the FASTER LaTeX table row.

    New table layout (12 columns):
    Env & Algorithm & Constr. & R_succ & T^total_opt & T^total_replan
    & T_trav & L_path & S_jerk & rho_vel & rho_acc & rho_jerk

    FASTER rows are sub-rows within an environment group, so they start
    with " & FASTER & {Hard} & ..."
    """
    # Success rate: a trial is successful ONLY if goal_reached AND collision_free
    # AND travel_time <= TIMEOUT_SECONDS
    collided_bags = collision_info.get("collided_bags", set())
    reached_bags = goal_status.get("reached_bags", set())

    # Build set of bags that timed out (travel_time > TIMEOUT_SECONDS)
    timed_out_bags = set()
    if per_bag_rows:
        for row in per_bag_rows:
            bag_name = row.get("bag_file", "").strip()
            try:
                tt = float(row.get("travel_time_s", 0))
            except (ValueError, TypeError):
                tt = 0.0
            if tt > TIMEOUT_SECONDS:
                timed_out_bags.add(bag_name)

    successful = len(reached_bags - collided_bags - timed_out_bags)
    success_rate = (float(successful) / float(goal_status["total"])) * 100.0

    whole_ms = comp_times["total_local_whole_ms"]
    safe_ms = comp_times["total_local_safe_ms"]
    replan_ms = comp_times["total_replan_ms"]

    if whole_ms is not None and safe_ms is not None:
        t_opt_str = "{{{:.1f} | {:.1f}}}".format(whole_ms, safe_ms)
    elif whole_ms is not None:
        t_opt_str = "{{{:.1f}}}".format(whole_ms)
    else:
        t_opt_str = "{N/A}"

    if replan_ms is not None:
        t_replan_str = "{:.1f}".format(replan_ms)
    else:
        t_replan_str = "N/A"

    t_trav = summary["travel_time_mean_s"]
    l_path = summary["path_length_mean_m"]
    s_jerk = summary["smoothness_l1_mean"]
    rho_vel = summary["vel_violations_pct_overall"]
    rho_acc = summary["acc_violations_pct_overall"]
    rho_jerk = summary["jerk_violations_pct_overall"]

    row = (
        "       & {algo} & Hard & $L_\\infty$ & {{{succ:.1f}}} & {t_opt} & {t_replan}"
        " & {{{trav:.1f}}} & {{{path:.1f}}} & {{{jerk:.1f}}}"
        " & {{{vel:.1f}}} & {{{acc:.1f}}} & {{{jrk:.1f}}} \\\\"
    ).format(
        algo=row_label,
        succ=success_rate,
        t_opt=t_opt_str,
        t_replan=t_replan_str,
        trav=t_trav,
        path=l_path,
        jerk=s_jerk,
        vel=rho_vel,
        acc=rho_acc,
        jrk=rho_jerk,
    )

    return row


ENV_TO_INDEX = {
    "easy_forest": 0,
    "medium_forest": 1,
    "hard_forest": 2,
}


def update_latex_table(latex_path, new_faster_row, env_name=None, row_label="FASTER (CP)"):
    """Read the LaTeX table, replace the Nth row whose algorithm == row_label.

    The table has three such rows (Easy, Medium, Hard) in order.
    We find and replace the Nth occurrence based on env_name.
    """
    if not os.path.exists(latex_path):
        print("ERROR: LaTeX table not found: {}".format(latex_path))
        sys.exit(1)

    with open(latex_path, "r") as f:
        lines = f.readlines()

    target_idx = ENV_TO_INDEX.get(env_name, 0) if env_name else 0
    env_display = ENV_DISPLAY_NAMES.get(env_name, env_name) if env_name else None

    # Match the row whose Algorithm cell equals row_label exactly. The
    # "& <label> &" delimiters avoid matching "FASTER" inside "FASTER (CP)".
    match_token = "& {} &".format(row_label)

    new_lines = []
    replaced = False
    faster_count = 0

    for line in lines:
        stripped = line.strip()

        if (not replaced
                and match_token in stripped
                and "\\\\" in stripped):
            if faster_count == target_idx:
                new_lines.append(new_faster_row + "\n")
                replaced = True
            else:
                new_lines.append(line)
            faster_count += 1
        else:
            new_lines.append(line)

    if not replaced:
        label = "{} in {} section".format(row_label, env_display) if env_display else row_label
        print("WARNING: Could not find '{}' row in {}. No changes made.".format(
            label, latex_path))
        return False

    with open(latex_path, "w") as f:
        f.writelines(new_lines)

    label = "FASTER in {} section".format(env_display) if env_display else "FASTER"
    print("Updated '{}' row in: {}".format(label, latex_path))
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate/update FASTER row in static_benchmark.tex"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to benchmark data directory (e.g., ../data)",
    )
    parser.add_argument(
        "--latex-table",
        type=str,
        default="/home/kkondo/paper_writing/SANDO_paper/tables/static_benchmark.tex",
        help="Path to static_benchmark.tex",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default=None,
        help="Environment name (e.g., easy_forest, medium_forest, hard_forest). "
             "If provided, updates the per-environment row.",
    )
    parser.add_argument(
        "--row-label",
        type=str,
        default="FASTER (CP)",
        help="Algorithm label of the table row to update, e.g. 'FASTER (CP)' "
             "(safe_faster data) or 'FASTER' (original_faster data).",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FASTER Static Benchmark Row Generator")
    print("=" * 70)

    print("\nReading data from: {}".format(args.data_dir))

    print("  1. Overall summary...")
    summary = read_overall_summary(args.data_dir)

    print("  2. Computation times...")
    comp_times = read_computation_times(args.data_dir)

    print("  3. Goal reached status...")
    goal_status = read_goal_reached(args.data_dir)

    print("  4. Collision check...")
    collision_info = read_collision_check(args.data_dir)

    print("  5. Per-bag summary (for timeout filter)...")
    per_bag_rows = read_per_bag_summary(args.data_dir)

    print("\n--- Computed Values ---")
    print("  R_succ:        {:.1f}%  ({}/{} reached, {} collisions)".format(
        goal_status["success_rate_pct"],
        goal_status["reached"],
        goal_status["total"],
        collision_info["collisions"],
    ))
    print("  T^total_opt:   {:.1f} | {:.1f} ms  (total_local_whole | total_local_safe)".format(
        comp_times["total_local_whole_ms"] or 0,
        comp_times["total_local_safe_ms"] or 0,
    ))
    print("  T^total_replan: {:.1f} ms".format(comp_times["total_replan_ms"] or 0))
    print("  T_trav:        {:.1f} s".format(summary["travel_time_mean_s"]))
    print("  L_path:        {:.1f} m".format(summary["path_length_mean_m"]))
    print("  S_jerk:        {:.1f} m/s^2".format(summary["smoothness_l1_mean"]))
    print("  rho_vel:       {:.2f}%".format(summary["vel_violations_pct_overall"]))
    print("  rho_acc:       {:.1f}%".format(summary["acc_violations_pct_overall"]))
    print("  rho_jerk:      {:.1f}%".format(summary["jerk_violations_pct_overall"]))

    new_row = format_faster_row(summary, comp_times, goal_status, collision_info,
                               per_bag_rows=per_bag_rows, env_name=args.env_name,
                               row_label=args.row_label)
    print("\n--- Generated LaTeX Row ---")
    print(new_row)

    print("\nUpdating: {}".format(args.latex_table))
    success = update_latex_table(args.latex_table, new_row, env_name=args.env_name,
                                 row_label=args.row_label)

    if success:
        env_label = args.env_name or "FASTER"
        print("\nDone. '{}' row updated successfully.".format(env_label))
    else:
        print("\nFailed to update FASTER row.")
        sys.exit(1)


if __name__ == "__main__":
    main()
