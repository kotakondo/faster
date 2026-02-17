#!/usr/bin/env python2
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

DATA_DIR = "/home/kota/data"
CSV_PATH = None  # set in main after DATA_DIR is resolved


class SimulationMonitor(object):
    def __init__(self, goal, threshold, start_pos=(0.0, 0.0, 2.0)):
        """
        goal: tuple (x, y, z) for desired goal position.
        threshold: distance (in meters) to consider the goal reached.
        start_pos: tuple (x, y, z) of initial position.
        """
        self.goal = goal
        self.threshold = threshold
        self.start_pos = start_pos
        self.current_pose = None
        self.started_moving = False
        self.move_time = None  # wall-clock time when agent first moved

        # Subscribe to the topic that publishes the current position.
        rospy.Subscriber("/SQ01s/goal", Goal, self.pose_callback)

    def pose_callback(self, msg):
        self.current_pose = (msg.p.x, msg.p.y, msg.p.z)
        # Detect when agent starts moving (>1m from start)
        if not self.started_moving and self.current_pose is not None:
            dx = self.current_pose[0] - self.start_pos[0]
            dy = self.current_pose[1] - self.start_pos[1]
            dz = self.current_pose[2] - self.start_pos[2]
            if math.sqrt(dx*dx + dy*dy + dz*dz) > 1.0:
                self.started_moving = True
                self.move_time = time.time()
                rospy.loginfo("Agent started moving!")

    def reached_goal(self):
        if self.current_pose is None:
            return False
        dx = self.current_pose[0] - self.goal[0]
        dy = self.current_pose[1] - self.goal[1]
        dz = self.current_pose[2] - self.goal[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
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


def launch_simulation(sim_num, env_source, goal_coords, world_path=None, start_pos=(0.0, 0.0, 2.0)):
    """
    Launch simulation processes (excluding roscore) and return a list of Popen processes.
    Commands are based on your tmuxp file.
    """
    # Define commands (adjust paths and topics as needed)
    if world_path:
        start_world_cmd = "{0} && roslaunch --wait acl_sim start_world.launch world_name:={1} gui:=false".format(env_source, world_path)
    else:
        start_world_cmd = "{0} && roslaunch --wait acl_sim start_world.launch gui:=false".format(env_source)
    perfect_tracker_cmd = (
        "{0} && roslaunch --wait acl_sim perfect_tracker_and_sim.launch x:={1} y:={2} z:={3} yaw:=0.0"
        .format(env_source, start_pos[0], start_pos[1], start_pos[2])
    )
    global_mapper_cmd = "{0} && roslaunch --wait global_mapper_ros global_mapper_node.launch".format(env_source)
    faster_interface_cmd = "{0} && roslaunch --wait faster faster_interface.launch".format(env_source)
    faster_launch_cmd = "{0} && roslaunch --wait faster faster.launch simulation_number:={1} data_dir:={2}".format(env_source, sim_num, DATA_DIR)
    rviz_cmd = "{0} && rviz rviz -d /home/kota/ws/src/faster/faster/rviz_cfgs/cvx_SQ01s.rviz".format(env_source)

    goal_pub_cmd = (
        "{0} && sleep 10 && "
        "rostopic pub /SQ01s/faster/term_goal geometry_msgs/PoseStamped "
        "'{{header: {{frame_id: \"world\"}}, pose: {{position: {{x: {1}, y: {2}, z: {3}}}, "
        "orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}}}' -1"
    ).format(env_source, goal_coords[0], goal_coords[1], goal_coords[2])

    bag_file = os.path.join(DATA_DIR, "faster_num_{0}.bag".format(sim_num))
    rosbag_cmd = (
        "{0} && rosbag record "
        "/SQ01s/faster/term_goal /SQ01s/goal /tf /SQ01s/faster/point_G_term "
        "/SQ01s/cvx/point_G /SQ01s/setpoint /SQ01s/faster/traj_whole "
        "/SQ01s/global_mapper_ros/unknown_grid "
        "-O {1}"
    ).format(env_source, bag_file)

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

    rospy.loginfo("Launching rosbag recording to {0} ...".format(bag_file))
    processes.append(run_command(rosbag_cmd))
    time.sleep(10)

    rospy.loginfo("Publishing goal...")
    processes.append(run_command(goal_pub_cmd))
    time.sleep(2)

    return processes


def _wait_process_terminate(p, timeout_sec):
    """
    Python 2-friendly "wait with timeout".
    Returns True if process exited, False if still running after timeout.
    """
    t0 = time.time()
    while True:
        rc = p.poll()
        if rc is not None:
            return True
        if time.time() - t0 > timeout_sec:
            return False
        time.sleep(0.1)


def kill_processes(processes):
    # First ask nicely
    for p in processes:
        try:
            p.terminate()
        except Exception:
            pass

    # Then wait a bit and force kill if needed
    for p in processes:
        try:
            exited = _wait_process_terminate(p, timeout_sec=5.0)
            if not exited:
                try:
                    p.kill()
                except Exception:
                    pass
        except Exception:
            # If anything weird happens, attempt a kill as last resort
            try:
                p.kill()
            except Exception:
                pass


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: {0} <total_sim_runs> <max_duration_seconds> <goal_x,goal_y,goal_z> <goal_threshold> [data_dir]"
              .format(sys.argv[0]))
        sys.exit(1)

    total_sim_runs = int(sys.argv[1])
    max_duration = float(sys.argv[2])
    goal_coords = tuple(map(float, sys.argv[3].split(',')))
    threshold = float(sys.argv[4])
    if len(sys.argv) >= 6:
        DATA_DIR = sys.argv[5]
    world_path = None
    if len(sys.argv) >= 7:
        world_path = sys.argv[6]
    start_pos = (0.0, 0.0, 2.0)
    if len(sys.argv) >= 8:
        start_pos = tuple(map(float, sys.argv[7].split(',')))
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    CSV_PATH = os.path.join(DATA_DIR, "goal_reached_status.csv")
    env_source = "source /home/kota/ws/devel/setup.bash && source /home/kota/mid360_ws/devel/setup.bash"

    # Optional but useful: verify roscore is up before init_node / subscribing
    # (You said roscore is run outside; this gives a clearer failure mode.)
    if not wait_for_ros_master(timeout=30):
        sys.exit(2)

    # Initialize the ROS node (now that roscore is running).
    rospy.init_node("simulation_monitor", anonymous=True)

    # Prepare CSV file: write header if file does not exist.
    if not os.path.exists(CSV_PATH):
        f = open(CSV_PATH, "w")
        try:
            f.write("sim_num,status\n")
        finally:
            f.close()

    sim_num = 0
    while (not rospy.is_shutdown()) and (sim_num < total_sim_runs):
        rospy.loginfo("\n==== Starting simulation number {0} ====".format(sim_num))

        # Create a new SimulationMonitor subscriber.
        monitor = SimulationMonitor(goal=goal_coords, threshold=threshold, start_pos=start_pos)
        processes = launch_simulation(sim_num, env_source, goal_coords, world_path, start_pos)
        goal_reached = False

        # Wait up to 120s for agent to start moving (startup grace period)
        startup_deadline = time.time() + 120.0
        while not monitor.started_moving and time.time() < startup_deadline and not rospy.is_shutdown():
            time.sleep(0.5)

        if not monitor.started_moving:
            rospy.logwarn("Agent never started moving within 120s, marking as timeout.")

        # Timeout clock starts when agent starts moving
        move_start = monitor.move_time if monitor.move_time else time.time()

        # Monitor simulation until goal reached or max duration exceeded.
        while (time.time() - move_start < max_duration) and (not rospy.is_shutdown()):
            if monitor.reached_goal():
                rospy.loginfo("Goal reached!")
                goal_reached = True
                break
            time.sleep(1)

        travel_time = time.time() - move_start
        if goal_reached:
            status = "reached"
            rospy.loginfo("Simulation {0} ended successfully in {1:.1f} seconds."
                          .format(sim_num, travel_time))
        else:
            status = "timeout"
            rospy.loginfo("Simulation {0} timed out after {1:.1f} seconds."
                          .format(sim_num, travel_time))

        # Write run status to CSV.
        csv_file = open(CSV_PATH, "a")
        try:
            csv_file.write("{0},{1}\n".format(sim_num, status))
        finally:
            csv_file.close()

        # Terminate all simulation processes (they do not include roscore).
        kill_processes(processes)
        rospy.loginfo("Simulation processes terminated.")

        sim_num += 1
        rospy.loginfo("Restarting simulation in 5 seconds...")
        time.sleep(5)

    rospy.loginfo("All simulation runs complete. Simulation monitor terminated.")
