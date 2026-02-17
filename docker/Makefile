# FASTER Docker Benchmark Pipeline
# ==================================
# Usage:
#   make build              Build Docker image
#   make run                Run Docker container (interactive)
#   make run-benchmark      Run full benchmark pipeline inside Docker
#   make analyze-benchmark  Generate FASTER LaTeX row from CSVs (host-side)
#   make help               Show available targets

# ---------- Parameters ----------
NUM_RUNS     ?= 10
MAX_DURATION ?= 50
GOAL         ?= 30.0,0.0,1.0
GOAL_TOL     ?= 0.5
V_MAX        ?= 5.0
A_MAX        ?= 20.0
J_MAX        ?= 100.0
DRONE_RADIUS ?= 0

LATEX_TABLE  ?= /home/kkondo/paper_writing/DYNUS_v3/tables/static_benchmark.tex
REPO_DIR     := $(shell cd .. && pwd)
WS_SRC_DIR   := $(shell cd ../.. && pwd)
DATA_DIR     := $(REPO_DIR)/data

# Timestamped run directory (auto-generated or user-specified)
# Usage: make run-benchmark              -> creates data/20260209_135000/
#        make analyze-benchmark RUN_DIR=20260209_135000
RUN_DIR      ?= $(shell date +%Y%m%d_%H%M%S)

# ---------- Docker ----------
# Auto-detect DISPLAY: use env var if set, otherwise find active X session
DISPLAY_VAL := $(or $(DISPLAY),$(shell who 2>/dev/null | grep -oP '\(:\d+\)' | head -1 | tr -d '()'))
$(if $(DISPLAY_VAL),,$(warning No DISPLAY found — Gazebo camera rendering will not work))

DOCKER_RUN := docker run -it \
		--gpus all \
		--env="NVIDIA_DRIVER_CAPABILITIES=all" \
		--privileged \
		-e DISPLAY=$(DISPLAY_VAL) \
		-e XAUTHORITY=/tmp/.docker.xauth \
		-e GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic \
		--network=host \
		--env="QT_X11_NO_MITSHM=1" \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-v $(HOME)/.Xauthority:/tmp/.docker.xauth:ro \
		--volume=${PWD}/gurobi.lic:/opt/gurobi/gurobi.lic:ro \
		-v $(DATA_DIR):/home/kota/data \
		-v $(REPO_DIR):/home/kota/ws/src/faster \
		--rm faster

# ---------- Static Benchmark ----------
WORLDS_SRC    ?= /home/kkondo/code/dynus_ws/src/dynus/worlds
OBSTACLE_CSV  ?= /home/kkondo/code/dynus_ws/src/dynus/benchmark_data/static
STATIC_ENVS   := easy_forest medium_forest hard_forest
STATIC_GOAL   ?= 105.0,0.0,2.0
STATIC_START  ?= 0.0,0.0,2.0
STATIC_DIR    ?= static_$(shell date +%Y%m%d_%H%M%S)

DOCKER_RUN_STATIC := docker run -it \
		--gpus all \
		--env="NVIDIA_DRIVER_CAPABILITIES=all" \
		--privileged \
		-e DISPLAY=$(DISPLAY_VAL) \
		-e XAUTHORITY=/tmp/.docker.xauth \
		-e GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic \
		--network=host \
		--env="QT_X11_NO_MITSHM=1" \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-v $(HOME)/.Xauthority:/tmp/.docker.xauth:ro \
		--volume=${PWD}/gurobi.lic:/opt/gurobi/gurobi.lic:ro \
		-v $(DATA_DIR):/home/kota/data \
		-v $(REPO_DIR):/home/kota/ws/src/faster \
		-v $(WORLDS_SRC):/home/kota/worlds:ro \
		-v $(OBSTACLE_CSV):/home/kota/obstacle_csv:ro \
		--rm faster

.PHONY: build build-clone run run-benchmark analyze-benchmark \
        run-static-benchmark analyze-static-benchmark help

build:
	docker build --build-arg ACL_MAPPING_CACHEBUST=`date +%s` -t faster -f $(REPO_DIR)/docker/Dockerfile $(REPO_DIR)/..

build-clone:
	docker build --build-arg DUMMY=`date +%s` -t faster -f $(REPO_DIR)/docker/Dockerfile $(REPO_DIR)/..

run:
	$(DOCKER_RUN)

run-benchmark:
	@echo "=== Benchmark run directory: $(RUN_DIR) ==="
	$(DOCKER_RUN) bash -c '\
		RUN=/home/kota/data/$(RUN_DIR) && \
		mkdir -p $$RUN && \
		source /home/kota/ws/devel/setup.bash && \
		source /home/kota/mid360_ws/devel/setup.bash && \
		cd /home/kota/ws/src/faster/docker && \
		python benchmarking.py $(NUM_RUNS) $(MAX_DURATION) $(GOAL) $(GOAL_TOL) $$RUN && \
		python analyze_bag.py $$RUN $(GOAL_TOL) $(V_MAX) $(A_MAX) $(J_MAX) && \
		python compute_average_computation.py $$RUN && \
		cd $$RUN && \
		python /home/kota/ws/src/faster/docker/collision_checker.py \
			/home/kota/data/hard_forest_obstacle_parameters.csv \
			$$RUN $(DRONE_RADIUS)'
	@echo "=== Data saved to: data/$(RUN_DIR)/ ==="

analyze-benchmark:
	python3 generate_static_benchmark_row.py \
		--data-dir $(DATA_DIR)/$(RUN_DIR) \
		--latex-table $(LATEX_TABLE)

run-static-benchmark:
	@echo "=== Static Benchmark: $(STATIC_DIR) ==="
	@echo "    Environments: $(STATIC_ENVS)"
	@echo "    Start: $(STATIC_START)  Goal: $(STATIC_GOAL)"
	@echo "    Trials per environment: $(NUM_RUNS)"
	@mkdir -p $(DATA_DIR)/$(STATIC_DIR)
	@echo "--- Patching world files for Gazebo 9 (ROS Melodic) ---"
	@for env in $(STATIC_ENVS); do \
		cp $(WORLDS_SRC)/$$env.world $(DATA_DIR)/$(STATIC_DIR)/$$env.world; \
		sed -i \
			-e "s/sdf version='1.7'/sdf version='1.6'/" \
			-e '/disable_physics/d' \
			-e '/<plugin name=.gazebo_ros_state/,/<\/plugin>/d' \
			$(DATA_DIR)/$(STATIC_DIR)/$$env.world; \
		echo "  Patched $$env.world"; \
	done
	@for env in $(STATIC_ENVS); do \
		echo ""; \
		echo "========================================"; \
		echo "=== Running $$env ($(NUM_RUNS) trials) ==="; \
		echo "========================================"; \
		$(DOCKER_RUN_STATIC) bash -c '\
			ENV_DIR=/home/kota/data/$(STATIC_DIR)/'"$$env"' && \
			mkdir -p $$ENV_DIR && \
			export GUROBI_HOME="/opt/gurobi952/linux64" && \
			export PATH="$$PATH:$$GUROBI_HOME/bin" && \
			export LD_LIBRARY_PATH="$$GUROBI_HOME/lib" && \
			cd /home/kota/ws && \
			source /opt/ros/melodic/setup.bash && \
			catkin build 2>&1 | tail -5 && \
			source /home/kota/ws/devel/setup.bash && \
			source /home/kota/mid360_ws/devel/setup.bash && \
			(roscore &) && sleep 3 && \
			cd /home/kota/ws/src/faster/docker && \
			python benchmarking.py $(NUM_RUNS) $(MAX_DURATION) $(STATIC_GOAL) $(GOAL_TOL) \
				$$ENV_DIR /home/kota/data/$(STATIC_DIR)/'"$$env"'.world $(STATIC_START) && \
			(mv /home/kota/data/computation_times_num_*.csv $$ENV_DIR/ 2>/dev/null || true) && \
			python analyze_bag.py $$ENV_DIR $(GOAL_TOL) $(V_MAX) $(A_MAX) $(J_MAX) && \
			python compute_average_computation.py $$ENV_DIR && \
			cd $$ENV_DIR && \
			python /home/kota/ws/src/faster/docker/collision_checker.py \
				/home/kota/obstacle_csv/'"$$env"'_obstacle_parameters.csv \
				$$ENV_DIR $(DRONE_RADIUS)'; \
	done
	@echo ""
	@echo "=== Static benchmark complete: data/$(STATIC_DIR)/ ==="

analyze-static-benchmark:
	@for env in $(STATIC_ENVS); do \
		echo "--- Re-analyzing bags for $$env ---"; \
		$(DOCKER_RUN_STATIC) bash -c '\
			ENV_DIR=/home/kota/data/$(STATIC_DIR)/'"$$env"' && \
			source /opt/ros/melodic/setup.bash && \
			source /home/kota/ws/devel/setup.bash && \
			source /home/kota/mid360_ws/devel/setup.bash && \
			cd /home/kota/ws/src/faster/docker && \
			python analyze_bag.py $$ENV_DIR $(GOAL_TOL) $(V_MAX) $(A_MAX) $(J_MAX) && \
			cd $$ENV_DIR && \
			python /home/kota/ws/src/faster/docker/collision_checker.py \
				/home/kota/obstacle_csv/'"$$env"'_obstacle_parameters.csv \
				$$ENV_DIR $(DRONE_RADIUS)'; \
	done
	@for env in $(STATIC_ENVS); do \
		echo "--- Analyzing $$env ---"; \
		python3 generate_static_benchmark_row.py \
			--data-dir $(DATA_DIR)/$(STATIC_DIR)/$$env \
			--latex-table $(LATEX_TABLE) \
			--env-name $$env; \
	done
	@echo ""
	@echo "=== LaTeX table updated: $(LATEX_TABLE) ==="

help:
	@echo "FASTER Docker Benchmark Pipeline"
	@echo "================================="
	@echo ""
	@echo "Targets:"
	@echo "  build                   Build Docker image"
	@echo "  build-clone             Build Docker image (bust cache for COPY)"
	@echo "  run                     Run Docker container (interactive)"
	@echo "  run-benchmark           Run full benchmark pipeline inside Docker"
	@echo "  analyze-benchmark       Generate FASTER LaTeX row from CSVs (host-side)"
	@echo "  run-static-benchmark    Run static benchmarks (easy/medium/hard, 10 trials each)"
	@echo "  analyze-static-benchmark  Analyze static benchmark data and update LaTeX table"
	@echo "  help                    Show this help message"
	@echo ""
	@echo "Parameters (override with VAR=value):"
	@echo "  NUM_RUNS=$(NUM_RUNS)  MAX_DURATION=$(MAX_DURATION)  GOAL=$(GOAL)"
	@echo "  GOAL_TOL=$(GOAL_TOL)  V_MAX=$(V_MAX)  A_MAX=$(A_MAX)  J_MAX=$(J_MAX)"
	@echo "  DRONE_RADIUS=$(DRONE_RADIUS)"
	@echo "  LATEX_TABLE=$(LATEX_TABLE)"
	@echo ""
	@echo "  RUN_DIR=<timestamp>       Specify run directory (default: auto-generated)"
	@echo "  STATIC_DIR=<dirname>      Specify static benchmark directory"
	@echo "  WORLDS_SRC=<path>         Path to .world files directory"
	@echo "  STATIC_GOAL=$(STATIC_GOAL)  Goal position for static benchmarks"
	@echo "  STATIC_START=$(STATIC_START)  Start position for static benchmarks"
