#!/usr/bin/env python
import os
import sys
import glob
import csv

def compute_averages_for_file(filepath):
    """
    Reads a CSV file that has two header lines:
      - First line: a title (skipped)
      - Second line: column headers: total_replan, jps, gurobi_whole, gurobi_safe
    Then, for each subsequent row, the four columns are parsed (as floats).
    For gurobi_safe, only nonzero values are accumulated and counted.
    Returns a tuple:
      (avg_total_replan, avg_jps, avg_gurobi_whole, avg_gurobi_safe, sample_count, safe_sample_count)
    If the file is empty or the header cannot be read, returns None.
    """
    total_replan_sum = 0.0
    jps_sum = 0.0
    gurobi_whole_sum = 0.0
    gurobi_safe_sum = 0.0
    count = 0
    safe_count = 0

    try:
        with open(filepath, "r") as f:
            # Skip first header line (title)
            f.readline()
            reader = csv.reader(f)
            # Read the column header line: "total_replan, jps, gurobi_whole, gurobi_safe"
            try:
                header = next(reader)
            except StopIteration:
                return None  # File has no header
            for row in reader:
                if not row or len(row) < 4:
                    continue
                try:
                    total_replan = float(row[0].strip())
                    jps = float(row[1].strip())
                    gurobi_whole = float(row[2].strip())
                    gurobi_safe = float(row[3].strip())
                    total_replan_sum += total_replan
                    jps_sum += jps
                    gurobi_whole_sum += gurobi_whole
                    count += 1
                    # Only count nonzero safe values.
                    if gurobi_safe != 0.0:
                        gurobi_safe_sum += gurobi_safe
                        safe_count += 1
                except Exception:
                    continue
    except Exception:
        return None  # In case the file cannot be opened/read

    if count > 0:
        avg_total_replan = total_replan_sum / count
        avg_jps = jps_sum / count
        avg_gurobi_whole = gurobi_whole_sum / count
    else:
        avg_total_replan = None
        avg_jps = None
        avg_gurobi_whole = None

    if safe_count > 0:
        avg_gurobi_safe = gurobi_safe_sum / safe_count
    else:
        avg_gurobi_safe = None

    return avg_total_replan, avg_jps, avg_gurobi_whole, avg_gurobi_safe, count, safe_count

def main():
    if len(sys.argv) < 2:
        print("Usage: {} <folder_path>".format(sys.argv[0]))
        sys.exit(1)
    
    folder = sys.argv[1]
    pattern = os.path.join(folder, "computation_times_num_*.csv")
    file_list = glob.glob(pattern)
    if not file_list:
        print("No files matching pattern found in folder: {}".format(folder))
        sys.exit(1)
    
    overall_total_replan_sum = 0.0
    overall_jps_sum = 0.0
    overall_gurobi_whole_sum = 0.0
    overall_gurobi_safe_sum = 0.0
    overall_count = 0
    overall_safe_count = 0
    results = []
    
    # Process each CSV file.
    for filepath in sorted(file_list):
        result = compute_averages_for_file(filepath)
        filename = os.path.basename(filepath)
        if result is None:
            print("File {}: Could not process (file may be empty or missing header).".format(filename))
            continue

        avg_total_replan, avg_jps, avg_gurobi_whole, avg_gurobi_safe, count, safe_count = result

        if count > 0:
            safe_avg_str = "{:.2f}".format(avg_gurobi_safe) if avg_gurobi_safe is not None else "N/A"
            results.append((filename, "{:.2f}".format(avg_total_replan), "{:.2f}".format(avg_jps),
                            "{:.2f}".format(avg_gurobi_whole), safe_avg_str))
            overall_total_replan_sum += avg_total_replan * count
            overall_jps_sum += avg_jps * count
            overall_gurobi_whole_sum += avg_gurobi_whole * count
            if avg_gurobi_safe is not None:
                overall_gurobi_safe_sum += avg_gurobi_safe * safe_count
                overall_safe_count += safe_count
            overall_count += count
            print("File {}: avg_total_replan = {:.2f} ms, avg_jps = {:.2f} ms, avg_gurobi_whole = {:.2f} ms, avg_gurobi_safe = {} ms ({} samples, {} safe samples)".format(
                filename, avg_total_replan, avg_jps, avg_gurobi_whole, safe_avg_str, count, safe_count))
        else:
            results.append((filename, "N/A", "N/A", "N/A", "N/A"))
            print("File {}: No valid samples found.".format(filename))
    
    if overall_count > 0:
        overall_avg_total_replan = overall_total_replan_sum / overall_count
        overall_avg_jps = overall_jps_sum / overall_count
        overall_avg_gurobi_whole = overall_gurobi_whole_sum / overall_count
    else:
        overall_avg_total_replan = "N/A"
        overall_avg_jps = "N/A"
        overall_avg_gurobi_whole = "N/A"
    
    if overall_safe_count > 0:
        overall_avg_gurobi_safe = overall_gurobi_safe_sum / overall_safe_count
        overall_safe_avg_str = "{:.2f}".format(overall_avg_gurobi_safe)
    else:
        overall_avg_gurobi_safe = "N/A"
        overall_safe_avg_str = "N/A"
    
    # Prepare rows for the output CSV.
    rows = []
    rows.append(["file_name", "avg_total_replan (ms)", "avg_jps (ms)", "avg_gurobi_whole (ms)", "avg_gurobi_safe (ms)"])
    for row in results:
        rows.append(row)
    rows.append(["Overall", "{:.2f}".format(overall_avg_total_replan),
                 "{:.2f}".format(overall_avg_jps),
                 "{:.2f}".format(overall_avg_gurobi_whole),
                 overall_safe_avg_str])
    
    # Write results to average_computation_times.csv in the given folder.
    output_csv = os.path.join(folder, "average_computation_times.csv")
    with open(output_csv, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print("Overall averages (across {} samples, {} safe samples):".format(overall_count, overall_safe_count))
    print("  total_replan = {:.2f} ms, jps = {:.2f} ms, gurobi_whole = {:.2f} ms, gurobi_safe = {}".format(
          overall_avg_total_replan, overall_avg_jps, overall_avg_gurobi_whole, overall_safe_avg_str))
    print("Averages saved to {}".format(output_csv))

if __name__ == "__main__":
    main()
