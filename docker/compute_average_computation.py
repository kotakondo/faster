#!/usr/bin/env python
import os
import sys
import glob
import csv
import math

def _to_float_or_nan(s):
    """Parse float; return NaN on failure."""
    try:
        return float(str(s).strip())
    except Exception:
        return float('nan')

def _is_valid(x):
    """True if x is a finite float."""
    try:
        return (x is not None) and (not math.isnan(x)) and (not math.isinf(x))
    except Exception:
        return False

def compute_averages_for_file(filepath):
    """
    Expected CSV layout:
      line1: title (skip)
      line2: header: total_replan, jps, gurobi_whole, total_local_whole, gurobi_safe, total_local_safe
      lines3+: numeric rows, possibly containing 'nan'

    Computes averages for:
      - total_replan
      - total_local_whole
      - total_local_safe

    NaNs are skipped per-field.
    Returns dict with sums/counts and averages.
    """
    # sums and counts (independent counts per metric since we skip NaNs)
    sums = {"total_replan": 0.0, "total_local_whole": 0.0, "total_local_safe": 0.0}
    cnts = {"total_replan": 0,   "total_local_whole": 0,   "total_local_safe": 0}

    rows_seen = 0

    try:
        f = open(filepath, "r")
    except Exception:
        return None

    try:
        # Skip first line (title)
        f.readline()

        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return None

        # Build a column index map from header text (robust to whitespace)
        header_map = {}
        for i, name in enumerate(header):
            key = str(name).strip()
            header_map[key] = i

        required = ["total_replan", "total_local_whole", "total_local_safe"]
        for k in required:
            if k not in header_map:
                # Older file format or unexpected header
                return None

        for row in reader:
            if not row:
                continue
            rows_seen += 1

            tr = _to_float_or_nan(row[header_map["total_replan"]])
            tw = _to_float_or_nan(row[header_map["total_local_whole"]])
            ts = _to_float_or_nan(row[header_map["total_local_safe"]])

            if _is_valid(tr):
                sums["total_replan"] += tr
                cnts["total_replan"] += 1
            if _is_valid(tw):
                sums["total_local_whole"] += tw
                cnts["total_local_whole"] += 1
            if _is_valid(ts):
                sums["total_local_safe"] += ts
                cnts["total_local_safe"] += 1

    finally:
        f.close()

    # Averages
    avgs = {}
    for k in sums.keys():
        avgs[k] = (sums[k] / cnts[k]) if cnts[k] > 0 else None

    return {
        "rows_seen": rows_seen,
        "sums": sums,
        "cnts": cnts,
        "avgs": avgs,
    }

def _fmt_ms(x):
    return "N/A" if x is None else "{:.2f}".format(x)

def _fmt_count(n):
    return "{:d}".format(int(n))

def main():
    if len(sys.argv) < 2:
        print("Usage: {} <folder_path>".format(sys.argv[0]))
        sys.exit(1)

    folder = sys.argv[1]
    pattern = os.path.join(folder, "computation_times_num_*.csv")
    file_list = sorted(glob.glob(pattern))
    if not file_list:
        print("No files matching pattern found in folder: {}".format(folder))
        sys.exit(1)

    per_file_rows = []
    # overall sums/counts (weighted by valid sample counts per metric)
    overall_sums = {"total_replan": 0.0, "total_local_whole": 0.0, "total_local_safe": 0.0}
    overall_cnts = {"total_replan": 0,   "total_local_whole": 0,   "total_local_safe": 0}

    print("")
    print("Computation Time Summary (ms)")
    print("Folder: {}".format(folder))
    print("Files:  {}".format(len(file_list)))
    print("")

    # Header for console table
    print("{:<32} {:>10} {:>18} {:>16} {:>10} {:>10} {:>10}".format(
        "file", "rows", "avg_total_replan", "avg_total_whole", "avg_total_safe",
        "n_replan", "n_safe"
    ))
    print("-" * 110)

    for filepath in file_list:
        filename = os.path.basename(filepath)
        result = compute_averages_for_file(filepath)
        if result is None:
            print("{:<32} {:>10} {:>18} {:>16} {:>10} {:>10} {:>10}".format(
                filename[:32], "ERR", "N/A", "N/A", "N/A", "0", "0"
            ))
            continue

        avgs = result["avgs"]
        cnts = result["cnts"]

        # Update overall (metric-wise)
        for k in overall_sums.keys():
            if cnts[k] > 0:
                overall_sums[k] += avgs[k] * cnts[k]
                overall_cnts[k] += cnts[k]

        per_file_rows.append([
            filename,
            _fmt_ms(avgs["total_replan"]),
            _fmt_ms(avgs["total_local_whole"]),
            _fmt_ms(avgs["total_local_safe"]),
            str(cnts["total_replan"]),
            str(cnts["total_local_whole"]),
            str(cnts["total_local_safe"]),
        ])

        print("{:<32} {:>10} {:>18} {:>16} {:>10} {:>10} {:>10} {:>10}".format(
            filename[:32],
            _fmt_count(result["rows_seen"]),
            _fmt_ms(avgs["total_replan"]),
            _fmt_ms(avgs["total_local_whole"]),
            _fmt_ms(avgs["total_local_safe"]),
            _fmt_count(cnts["total_replan"]),
            _fmt_count(cnts["total_local_safe"]),
            ""
        ))

    # Overall averages
    overall_avgs = {}
    for k in overall_sums.keys():
        overall_avgs[k] = (overall_sums[k] / overall_cnts[k]) if overall_cnts[k] > 0 else None

    print("-" * 110)
    print("OVERALL (weighted by valid samples):")
    print("  avg_total_replan     : {} ms  (n={})".format(_fmt_ms(overall_avgs["total_replan"]), overall_cnts["total_replan"]))
    print("  avg_total_whole      : {} ms  (n={})".format(_fmt_ms(overall_avgs["total_local_whole"]), overall_cnts["total_local_whole"]))
    print("  avg_total_safe       : {} ms  (n={})".format(_fmt_ms(overall_avgs["total_local_safe"]), overall_cnts["total_local_safe"]))
    print("")

    # Write CSV output
    output_csv = os.path.join(folder, "average_computation_times.csv")
    rows = []
    rows.append([
        "file_name",
        "avg_total_replan_ms",
        "avg_total_local_whole_ms",
        "avg_total_local_safe_ms",
        "n_total_replan",
        "n_total_local_whole",
        "n_total_local_safe",
    ])
    rows.extend(per_file_rows)
    rows.append([
        "Overall",
        _fmt_ms(overall_avgs["total_replan"]),
        _fmt_ms(overall_avgs["total_local_whole"]),
        _fmt_ms(overall_avgs["total_local_safe"]),
        str(overall_cnts["total_replan"]),
        str(overall_cnts["total_local_whole"]),
        str(overall_cnts["total_local_safe"]),
    ])

    f = open(output_csv, "wb")
    try:
        writer = csv.writer(f)
        writer.writerows(rows)
    finally:
        f.close()

    print("Saved averages to: {}".format(output_csv))

if __name__ == "__main__":
    main()
