#!/usr/bin/env python3
import subprocess
import sys
import time

def run_command(cmd):
    """Launch a command via bash -c and return the Popen handle."""
    return subprocess.Popen(["bash", "-c", cmd])

def main():
    if len(sys.argv) < 2:
        print("Usage: {} <simulation_number>".format(sys.argv[0]))
        sys.exit(1)

    sim_num = sys.argv[1]

    # Use the faster workspace setup as in your tmuxp file.
    env_source = "source /home/kota/ws/devel/setup.bash"

    # Define the commands as per your tmuxp file:
    roscore_cmd = f"{env_source} && roscore"
    start_world_cmd = f"{env_source} && roslaunch --wait acl_sim start_world.launch"
    perfect_tracker_cmd = f"{env_source} && roslaunch --wait acl_sim perfect_tracker_and_sim.launch x:=0.0 y:=0.0 z:=3.0 yaw:=0.0"
    global_mapper_cmd = f"{env_source} && roslaunch --wait global_mapper_ros global_mapper_node.launch"
    faster_interface_cmd = f"{env_source} && roslaunch --wait faster faster_interface.launch"
    faster_launch_cmd = f"{env_source} && roslaunch --wait faster faster.launch simulation_number:={sim_num}"
    rviz_cmd = f"{env_source} && rviz rviz -d /home/kota/ws/src/faster/faster/rviz_cfgs/cvx_SQ01s.rviz"
    goal_pub_cmd = f"""{env_source} && sleep 10 && rostopic pub /SQ01s/faster/term_goal geometry_msgs/PoseStamped '{{header: {{frame_id: "world"}}, pose: {{position: {{x: 105.0, y: 0.0, z: 3.0}}, orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}}}' -1"""

    bag_file = f"/home/kota/data/faster_num_{sim_num}.bag"
    rosbag_cmd = f"{env_source} && rosbag record -a -O {bag_file}"

    processes = []

    print("Launching roscore...")
    processes.append(run_command(roscore_cmd))
    time.sleep(2)  # Wait for roscore to initialize

    print("Launching start_world.launch...")
    processes.append(run_command(start_world_cmd))
    time.sleep(2)

    print("Launching perfect_tracker_and_sim.launch...")
    processes.append(run_command(perfect_tracker_cmd))
    time.sleep(2)

    print("Launching global_mapper_ros global_mapper_node.launch...")
    processes.append(run_command(global_mapper_cmd))
    time.sleep(2)

    print("Launching faster_interface.launch...")
    processes.append(run_command(faster_interface_cmd))
    time.sleep(2)

    print("Launching faster.launch...")
    processes.append(run_command(faster_launch_cmd))
    time.sleep(2)

    print("Launching RViz...")
    processes.append(run_command(rviz_cmd))
    time.sleep(2)

    print("Publishing goal...")
    processes.append(run_command(goal_pub_cmd))
    time.sleep(2)

    print(f"Launching rosbag recording to {bag_file} ...")
    processes.append(run_command(rosbag_cmd))
    time.sleep(2)

    print("All processes launched. Press Ctrl+C to terminate.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Terminating all processes...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait()
        print("Done.")

if __name__ == "__main__":
    main()
