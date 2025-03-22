#!/usr/bin/env python3
import subprocess
import sys
import time
import rospy
import rosgraph
from geometry_msgs.msg import PoseStamped
from snapstack_msgs.msg import Goal  # adjust if needed
import math
import os

# you need to run roscore outside of this script



CSV_PATH = "/home/kota/data/goal_reached_status.csv"

class SimulationMonitor:
    def __init__(self, goal, threshold):
        """
        goal: tuple (x, y, z) for desired goal position.
        threshold: distance (in meters) to consider the goal reached.
        """
        self.goal = goal
        self.threshold = threshold
        self.current_pose = None
        # Subscribe to the topic that publishes the current position.
        # Adjust the topic name as necessary.
        rospy.Subscriber("/SQ01s/goal", Goal, self.pose_callback)

    def pose_callback(self, msg):
        self.current_pose = (msg.p.x, msg.p.y, msg.p.z)
        rospy.loginfo("Current position: {}".format(self.current_pose))

    def reached_goal(self):
        if self.current_pose is None:
            return False
        dx = self.current_pose[0] - self.goal[0]
        dy = self.current_pose[1] - self.goal[1]
        dz = self.current_pose[2] - self.goal[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        rospy.loginfo("Distance to goal: {:.3f}".format(dist))
        return dist <= self.threshold

def wait_for_ros_master(timeout=30):
    """Wait for the ROS master to be available, or exit after timeout."""
    start_time = time.time()
    master = rosgraph.Master('/simulation_monitor')
    while True:
        try:
            master.getUri()
            rospy.loginfo("ROS master is available!")
            return True
        except Exception:
            if time.time() - start_time > timeout:
                rospy.logerr("Timeout waiting for ROS master!")
                return False
            rospy.loginfo("Waiting for ROS master...")
            time.sleep(1)

def run_command(cmd):
    """Launch a command via bash -c and return the Popen handle."""
    return subprocess.Popen(["bash", "-c", cmd])

def launch_simulation(sim_num, env_source):
    """
    Launch simulation processes (excluding roscore) and return a list of Popen processes.
    Commands are based on your tmuxp file.
    """
    # Define commands (adjust paths and topics as needed)
    start_world_cmd = f"{env_source} && roslaunch --wait acl_sim start_world.launch"
    perfect_tracker_cmd = f"{env_source} && roslaunch --wait acl_sim perfect_tracker_and_sim.launch x:=0.0 y:=0.0 z:=3.0 yaw:=0.0"
    global_mapper_cmd   = f"{env_source} && roslaunch --wait global_mapper_ros global_mapper_node.launch"
    faster_interface_cmd = f"{env_source} && roslaunch --wait faster faster_interface.launch"
    faster_launch_cmd   = f"{env_source} && roslaunch --wait faster faster.launch simulation_number:={sim_num}"
    rviz_cmd            = f"{env_source} && rviz rviz -d /home/kota/ws/src/faster/faster/rviz_cfgs/cvx_SQ01s.rviz"
    goal_pub_cmd = (f"{env_source} && sleep 10 && "
                    "rostopic pub /SQ01s/faster/term_goal geometry_msgs/PoseStamped "
                    "'{header: {frame_id: \"world\"}, pose: {position: {x: 105.0, y: 0.0, z: 3.0}, "
                    "orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}' -1")
    bag_file = f"/home/kota/data/faster_num_{sim_num}.bag"
    rosbag_cmd = f"{env_source} && rosbag record /SQ01s/faster/term_goal /SQ01s/goal /tf /SQ01s/faster/point_G_term /SQ01s/cvx/point_G /SQ01s/setpoint /SQ01s/faster/traj_whole /SQ01s/global_mapper_ros/unknown_grid -O {bag_file}"

    processes = []

    rospy.loginfo("Launching start_world.launch...")
    processes.append(run_command(start_world_cmd))
    time.sleep(2)

    rospy.loginfo("Launching perfect_tracker_and_sim.launch...")
    processes.append(run_command(perfect_tracker_cmd))
    time.sleep(2)

    rospy.loginfo("Launching global_mapper_ros global_mapper_node.launch...")
    processes.append(run_command(global_mapper_cmd))
    time.sleep(2)

    rospy.loginfo("Launching faster_interface.launch...")
    processes.append(run_command(faster_interface_cmd))
    time.sleep(2)

    rospy.loginfo("Launching faster.launch...")
    processes.append(run_command(faster_launch_cmd))
    time.sleep(2)

    rospy.loginfo("Launching RViz...")
    processes.append(run_command(rviz_cmd))
    time.sleep(2)

    rospy.loginfo("Publishing goal...")
    processes.append(run_command(goal_pub_cmd))
    time.sleep(2)

    rospy.loginfo(f"Launching rosbag recording to {bag_file} ...")
    processes.append(run_command(rosbag_cmd))
    time.sleep(2)

    return processes

def kill_processes(processes):
    for p in processes:
        p.terminate()
    for p in processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: {} <total_sim_runs> <max_duration_seconds> <goal_x,goal_y,goal_z> <goal_threshold>".format(sys.argv[0]))
        sys.exit(1)

    total_sim_runs = int(sys.argv[1])
    max_duration = float(sys.argv[2])
    goal_coords = tuple(map(float, sys.argv[3].split(',')))
    threshold = float(sys.argv[4])
    env_source = "source /home/kota/ws/devel/setup.bash"

    # Initialize the ROS node (now that roscore is running).
    rospy.init_node("simulation_monitor", anonymous=True)
    
    # Prepare CSV file: write header if file does not exist.
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w") as f:
            f.write("sim_num,status\n")

    sim_num = 0
    while not rospy.is_shutdown() and sim_num < total_sim_runs:
        rospy.loginfo("\n==== Starting simulation number {} ====".format(sim_num))
        
        # Create a new SimulationMonitor subscriber.
        monitor = SimulationMonitor(goal=goal_coords, threshold=threshold)
        sim_start_time = time.time()
        processes = launch_simulation(sim_num, env_source)
        goal_reached = False

        # Monitor simulation until goal reached or max duration exceeded.
        while time.time() - sim_start_time < max_duration and not rospy.is_shutdown():
            if monitor.reached_goal():
                rospy.loginfo("Goal reached!")
                goal_reached = True
                break
            time.sleep(1)

        travel_time = time.time() - sim_start_time
        if goal_reached:
            status = "reached"
            rospy.loginfo("Simulation {} ended successfully in {:.1f} seconds.".format(sim_num, travel_time))
        else:
            status = "timeout"
            rospy.loginfo("Simulation {} timed out after {:.1f} seconds.".format(sim_num, travel_time))

        # Write run status to CSV.
        with open(CSV_PATH, "a") as csv_file:
            csv_file.write(f"{sim_num},{status}\n")

        # Terminate all simulation processes (they do not include roscore).
        kill_processes(processes)
        rospy.loginfo("Simulation processes terminated.")

        sim_num += 1
        rospy.loginfo("Restarting simulation in 5 seconds...")
        time.sleep(5)

    rospy.loginfo("All simulation runs complete. Terminating roscore.")
    rospy.loginfo("Simulation monitor terminated.")
