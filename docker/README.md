# FASTER Benchmarking Pipeline

## Prerequisites

1. **Docker** with NVIDIA GPU support (`nvidia-docker2`)
2. **Gurobi license** at `docker/gurobi.lic`
3. **World files** from the DYNUS repo at `~/code/dynus_ws/src/dynus/worlds/`
4. **Obstacle CSVs** at `~/code/dynus_ws/src/dynus/benchmark_data/static/`
5. **acl-mapping** on `faster_for_bench` branch at `~/code/faster_ws/src/acl-mapping/`

## Quick Reference

```bash
cd ~/code/faster_ws/src/faster/docker

# Build Docker image
make build

# Run static benchmark (easy/medium/hard, 10 trials each)
make run-static-benchmark

# Re-analyze existing data and update LaTeX table
make analyze-static-benchmark STATIC_DIR=static_20260216_172052
```

## Building the Docker Image

```bash
make build
```

This builds the `faster` Docker image. Key details:
- **Build context** is `src/` (parent of `faster/`), so both `acl-mapping/` and `faster/` are COPYed in
- `acl-mapping` is copied from local (no git fetch) — edit it locally, then rebuild
- The `faster` repo is also copied in, but at runtime it is **volume-mounted** over the copy, so C++ and config changes take effect without rebuilding
- A `catkin build` runs inside the container at the start of each benchmark run to compile any C++ changes from the mounted volume
- Use `make build` after changing `acl-mapping` or the Dockerfile

### Cache busting

The Makefile passes `--build-arg ACL_MAPPING_CACHEBUST=<timestamp>` to bust the Docker layer cache for the `COPY acl-mapping` step. This ensures local changes to `acl-mapping` are always picked up.

## Static Benchmark

### Running

```bash
make run-static-benchmark
```

This runs 10 trials for each of 3 environments (`easy_forest`, `medium_forest`, `hard_forest`).

**Default parameters** (override with `VAR=value`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_RUNS` | 10 | Trials per environment |
| `MAX_DURATION` | 50 | Timeout in seconds (starts after agent moves >1m) |
| `STATIC_GOAL` | 105.0,0.0,2.0 | Goal position (x,y,z) |
| `STATIC_START` | 0.0,0.0,2.0 | Start position (x,y,z) |
| `GOAL_TOL` | 0.5 | Goal reached threshold (meters) |
| `V_MAX` | 5.0 | Velocity constraint for violation checking |
| `A_MAX` | 20.0 | Acceleration constraint for violation checking |
| `J_MAX` | 100.0 | Jerk constraint for violation checking |
| `DRONE_RADIUS` | 0 | Drone radius for collision checking |
| `WORLDS_SRC` | ~/code/dynus_ws/src/dynus/worlds | Path to .world files |
| `OBSTACLE_CSV` | ~/code/dynus_ws/src/dynus/benchmark_data/static | Path to obstacle CSVs |

Example with custom parameters:

```bash
make run-static-benchmark NUM_RUNS=5 MAX_DURATION=60 V_MAX=3.0
```

### What happens during a run

For each environment:

1. World files are patched from SDF 1.7 to SDF 1.6 (Gazebo 9 compatibility)
2. Docker container starts with volume mounts for the faster repo, world files, and obstacle CSVs
3. `catkin build` runs inside the container to compile any C++ changes
4. `benchmarking.py` runs N trials sequentially:
   - Launches Gazebo (headless), perfect tracker, global mapper, FASTER
   - Starts rosbag recording
   - Publishes goal after 10s warmup
   - Waits for agent to move >1m from start (up to 120s)
   - Starts timeout clock after movement detected
   - Monitors until goal reached or timeout
   - Records `reached`/`timeout` status to `goal_reached_status.csv`
   - Kills all processes, waits 5s, starts next trial
5. `analyze_bag.py` processes all rosbags
6. `compute_average_computation.py` averages computation time CSVs
7. `collision_checker.py` checks trajectories against obstacle CSVs

### Data directory structure

```
data/
  static_YYYYMMDD_HHMMSS/          # auto-generated timestamp
    easy_forest.world               # patched world file
    medium_forest.world
    hard_forest.world
    easy_forest/
      faster_num_0.bag              # rosbag for trial 0
      faster_num_1.bag
      ...
      faster_num_9.bag
      computation_times_num_0.csv   # per-replan computation times
      computation_times_num_1.csv
      ...
      goal_reached_status.csv       # reached/timeout per trial
      average_computation_times.csv # averaged computation times
      faster_per_bag_summary.csv    # per-bag metrics (travel time, path length, violations, etc.)
      faster_overall_summary.csv    # overall metrics across all bags
      faster_statistics.txt         # human-readable statistics
      collision_check.csv           # per-bag collision status
      faster_num_0_velocity_axis_hist.pdf        # velocity histogram
      faster_num_0_vel_accel_jerk_axes_snap.pdf  # time history plot
      ...
    medium_forest/
      (same structure)
    hard_forest/
      (same structure)
```

### Analyzing / re-analyzing

To re-run analysis on existing data (e.g., after changing analysis scripts) and update the LaTeX table:

```bash
make analyze-static-benchmark STATIC_DIR=static_20260216_172052
```

This:
1. Re-runs `analyze_bag.py` and `collision_checker.py` inside Docker (needs rosbag Python 2 environment)
2. Runs `generate_static_benchmark_row.py` on the host (Python 3) for each environment
3. Updates the FASTER row in the LaTeX table at `LATEX_TABLE`

To change which LaTeX table is updated:

```bash
make analyze-static-benchmark STATIC_DIR=static_20260216_172052 \
    LATEX_TABLE=/path/to/your/table.tex
```

## Key Configuration Files

### FASTER planner parameters (`faster/param/faster.yaml`)

Dynamic constraints and planner tuning:

```yaml
v_max: 5.0          # max velocity (m/s)
a_max: 20.0         # max acceleration (m/s^2)
j_max: 100.0        # max jerk (m/s^3)
N_whole: 10          # number of segments for whole trajectory
N_safe: 10           # number of segments for safe trajectory
Ra: 10.0             # planning horizon radius
gamma_whole: 40      # time allocation search range (backward)
gammap_whole: 40     # time allocation search range (forward)
increment_whole: 0.5 # time allocation factor increment
force_goal_height: false
goal_height: 2.0
```

These are baked into the Docker image but overridden at runtime via the volume-mounted repo (catkin build runs inside the container).

### Global mapper parameters (`acl-mapping/.../global_mapper.yaml`)

```yaml
origin: [0.0, 0.0, 2.0]
world_dimensions: [15.0, 15.0, 4.0]
resolution: 0.3
Ra: 10.0             # sensing radius
r2: 16.0             # unknown space radius
init_value: -1.0     # MUST be -1.0 (means "unknown"); 0.4 breaks the unknown grid
```

**Important:** `acl-mapping` is NOT volume-mounted — it is baked into the Docker image. Changes to `global_mapper.yaml` require `make build`.

## Metrics Computed

### Constraint Violation (`analyze_bag.py`)

Per-axis L-infinity check on each `/SQ01s/goal` command message:

- **Velocity violation**: `|vx| > v_max + 1e-3` OR `|vy| > v_max + 1e-3` OR `|vz| > v_max + 1e-3`
- **Acceleration violation**: same pattern with `a_max`
- **Jerk violation**: same pattern with `j_max`
- **Percentage** = `(violating_commands / total_commands) * 100`

### Success Rate (`generate_static_benchmark_row.py`)

A trial is successful only if ALL of:
1. Goal was reached (not timeout)
2. No collision detected
3. Travel time <= `TIMEOUT_SECONDS` (50s)

### Collision Check (`collision_checker.py`)

Checks each `/SQ01s/goal` position in the rosbag against obstacle cylinders from the obstacle CSV. Reports collision if the drone position is within `cylinder_radius + drone_radius` of any obstacle.

### Smoothness

- **S_jerk** (L1 smoothness): `integral(||jerk|| dt)` over the trajectory
- **J_smooth** (RMS jerk): `sqrt(integral(||jerk||^2 dt) / T)`
- **S_eff** (RMS snap): `sqrt(integral(||snap||^2 dt) / T)` (snap = d(jerk)/dt)

### Computation Time (`compute_average_computation.py`)

Averaged from `computation_times_num_*.csv` files written by the FASTER node. Reports:
- `total_replan_ms`: total replan callback time
- `total_local_whole_ms`: time for whole trajectory optimization
- `total_local_safe_ms`: time for safe trajectory optimization

## Troubleshooting

### `init_value` in `global_mapper.yaml`

Must be `-1.0`. If set to `0.4`, all voxels start as "free" (below the 0.6 threshold), the unknown grid is empty, and FASTER never plans.

### Gazebo GUI in Docker

Gazebo GUI is disabled (`gui:=false`) because OpenGL inside Docker causes `SIGABRT` (exit code 134). If you need the GUI, run `xhost +local:docker` on the host and set `gui:=true` in `benchmarking.py`.

### FASTER segfault (exit code -11)

Common cause: the `updateMap` function had a mutex leak on early return when the unknown cloud was empty. This has been fixed. If it recurs, check that `mtx_map` and `mtx_unk` are always unlocked on all code paths.

### Docker layer caching

If changes to `acl-mapping` are not reflected after rebuild, the cache-busting `ARG ACL_MAPPING_CACHEBUST` should handle this automatically. If not, try `docker build --no-cache`.

### Volume mounting

- `faster/` repo is volume-mounted at runtime — C++ and config changes take effect after `catkin build` inside the container
- `acl-mapping/` and `acl-gazebo/` are NOT mounted — they are baked into the image and require `make build` to update
