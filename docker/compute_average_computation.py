#!/usr/bin/env python
import os
import sys
import glob
import csv
import math

def _to_float_or_nan(s):
    try:
        return float(str(s).strip())
    except Exception:
        return float('nan')

def _is_valid(x):
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
    """
    sums = {"total_replan": 0.0, "total_local_whole": 0.0, "total_local_safe": 0.0}
    cnts = {"total_replan": 0,   "total_local_whole": 0,   "total_local_safe": 0}
    rows_seen = 0

    try:
        f = open(filepath, "r")
    except Exception:
        return None

    try:
        f.readline()  # skip title
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return None

        header_map = {}
        for i, name in enumerate(header):
            header_map[str(name).strip()] = i

        required = ["total_replan", "total_local_whole", "total_local_safe"]
        for k in required:
            if k not in header_map:
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

    avgs = {}
    for k in sums.keys():
        avgs[k] = (sums[k] / cnts[k]) if cnts[k] > 0 else None

    # Safe success rate relative to total replans with valid total_replan
    safe_rate = None
    if cnts["total_replan"] > 0:
        safe_rate = 100.0 * float(cnts["total_local_safe"]) / float(cnts["total_replan"])

    return {
        "rows_seen": rows_seen,
        "sums": sums,
        "cnts": cnts,
        "avgs": avgs,
        "safe_rate_pct": safe_rate,
    }

def _fmt_ms(x):
    return "N/A" if x is None else "{:.2f}".format(x)

def _fmt_count(n):
    return "{:d}".format(int(n))

def _csv_num(x, ndigits=6):
    """
    For CSV: write numbers as numbers; missing as empty string.
    """
    if x is None:
        return ""
    if not _is_valid(x):
        return ""
    # keep as float with reasonable precision
    return round(float(x), ndigits)

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

    # overall sums/counts (weighted by valid sample counts per metric)
    overall_sums = {"total_replan": 0.0, "total_local_whole": 0.0, "total_local_safe": 0.0}
    overall_cnts = {"total_replan": 0,   "total_local_whole": 0,   "total_local_safe": 0}

    # Store structured per-file stats for nicer CSV output
    per_file = []

    print("")
    print("Computation Time Summary (ms)")
    print("Folder: {}".format(folder))
    print("Files:  {}".format(len(file_list)))
    print("")
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

        # Update overall (metric-wise, weighted)
        for k in overall_sums.keys():
            if cnts[k] > 0 and avgs[k] is not None:
                overall_sums[k] += avgs[k] * cnts[k]
                overall_cnts[k] += cnts[k]

        per_file.append({
            "file": filename,
            "rows_seen": result["rows_seen"],
            "avg_total_replan": avgs["total_replan"],
            "n_total_replan": cnts["total_replan"],
            "avg_total_whole": avgs["total_local_whole"],
            "n_total_whole": cnts["total_local_whole"],
            "avg_total_safe": avgs["total_local_safe"],
            "n_total_safe": cnts["total_local_safe"],
            "safe_rate_pct": result["safe_rate_pct"],
        })

        print("{:<32} {:>10} {:>18} {:>16} {:>10} {:>10} {:>10}".format(
            filename[:32],
            _fmt_count(result["rows_seen"]),
            _fmt_ms(avgs["total_replan"]),
            _fmt_ms(avgs["total_local_whole"]),
            _fmt_ms(avgs["total_local_safe"]),
            _fmt_count(cnts["total_replan"]),
            _fmt_count(cnts["total_local_safe"]),
        ))

    # Overall averages
    overall_avgs = {}
    for k in overall_sums.keys():
        overall_avgs[k] = (overall_sums[k] / overall_cnts[k]) if overall_cnts[k] > 0 else None

    overall_safe_rate = None
    if overall_cnts["total_replan"] > 0:
        overall_safe_rate = 100.0 * float(overall_cnts["total_local_safe"]) / float(overall_cnts["total_replan"])

    print("-" * 110)
    print("OVERALL (weighted by valid samples):")
    print("  avg_total_replan     : {} ms  (n={})".format(_fmt_ms(overall_avgs["total_replan"]), overall_cnts["total_replan"]))
    print("  avg_total_whole      : {} ms  (n={})".format(_fmt_ms(overall_avgs["total_local_whole"]), overall_cnts["total_local_whole"]))
    print("  avg_total_safe       : {} ms  (n={})".format(_fmt_ms(overall_avgs["total_local_safe"]), overall_cnts["total_local_safe"]))
    print("  safe_success_rate    : {} %".format("N/A" if overall_safe_rate is None else "{:.2f}".format(overall_safe_rate)))
    print("")

    # ---------------- Nicer CSV output ----------------
    output_csv = os.path.join(folder, "total_computation_times.csv")
    f = open(output_csv, "wb")
    try:
        w = csv.writer(f)

        # Title row + metadata-like note rows (still valid CSV)
        w.writerow(["Average Computation Times (ms)"])
        w.writerow(["Folder", folder])
        w.writerow(["NaN handling", "NaN/Inf values skipped per metric"])
        w.writerow([])

        # Column headers (clean, consistent, sortable)
        w.writerow([
            "file_name",
            "rows_seen",
            "avg_total_replan_ms", "n_total_replan",
            "avg_total_whole_ms",  "n_total_whole",
            "avg_total_safe_ms",   "n_total_safe",
            "safe_success_rate_pct"
        ])

        # Data rows: numeric values stay numeric; missing are blank
        for r in per_file:
            w.writerow([
                r["file"],
                r["rows_seen"],
                _csv_num(r["avg_total_replan"]), r["n_total_replan"],
                _csv_num(r["avg_total_whole"]),  r["n_total_whole"],
                _csv_num(r["avg_total_safe"]),   r["n_total_safe"],
                _csv_num(r["safe_rate_pct"], ndigits=2)
            ])

        # Blank line + Overall row
        w.writerow([])
        w.writerow([
            "Overall (weighted by valid samples)",
            "",
            _csv_num(overall_avgs["total_replan"]), overall_cnts["total_replan"],
            _csv_num(overall_avgs["total_local_whole"]), overall_cnts["total_local_whole"],
            _csv_num(overall_avgs["total_local_safe"]), overall_cnts["total_local_safe"],
            _csv_num(overall_safe_rate, ndigits=2)
        ])

    finally:
        f.close()

    print("Saved averages to: {}".format(output_csv))

if __name__ == "__main__":
    main()
