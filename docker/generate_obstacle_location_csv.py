#!/usr/bin/env python
import rosbag
import csv
import glob
import os
import sys
import math
from geometry_msgs.msg import PoseStamped

def load_forest_parameters(csv_path):
    """Load forest parameters from CSV and return a list of cylinder dictionaries."""
    cylinders = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cyl = {
                "id": row["id"],
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "radius": float(row["radius"]),
                "height": float(row["height"])
            }
            cylinders.append(cyl)
    return cylinders

def check_collision(agent_pos, cylinder):
    """
    Check if agent_pos (tuple of x,y,z) is colliding with the given cylinder.
    Cylinder is defined by (x, y, z, radius, height), where z is the center's z.
    In our forest, the cylinder is spawned with pos_z = height/2, meaning it spans z=0 to z=height.
    """
    dx = agent_pos[0] - cylinder["x"]
    dy = agent_pos[1] - cylinder["y"]
    horizontal_dist = math.sqrt(dx*dx + dy*dy)
    if horizontal_dist <= cylinder["radius"]:
        # Check vertical collision: assuming ground at z=0, cylinder extends to z=cylinder["height"]
        if 0 <= agent_pos[2] <= cylinder["height"]:
            return True
    return False

def process_bag(bag_path, cylinders, topic="/agent_pose"):
    """Check for collision in a bag file given the forest cylinders. Returns True if collision is found."""
    collision_found = False
    try:
        bag = rosbag.Bag(bag_path)
    except Exception as e:
        print("Error opening bag {}: {}".format(bag_path, e))
        return False

    for topic_name, msg, t in bag.read_messages(topics=[topic]):
        # Assume msg is of type PoseStamped.
        agent_pos = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        for cyl in cylinders:
            if check_collision(agent_pos, cyl):
                collision_found = True
                break
        if collision_found:
            break
    bag.close()
    return collision_found

def main():
    if len(sys.argv) < 3:
        print("Usage: {} <forest_csv_path> <bag_folder>".format(sys.argv[0]))
        sys.exit(1)
    forest_csv = sys.argv[1]
    bag_folder = sys.argv[2]

    cylinders = load_forest_parameters(forest_csv)
    bag_files = glob.glob(os.path.join(bag_folder, "*.bag"))
    results = []

    for bag_file in bag_files:
        print("Processing bag: {}".format(bag_file))
        collision = process_bag(bag_file, cylinders, topic="/agent_pose")
        results.append((os.path.basename(bag_file), "collision" if collision else "no collision"))

    # Write results to collision_check.csv.
    output_csv = "collision_check.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bag_file", "collision_status"])
        for bag_file, status in results:
            writer.writerow([bag_file, status])
    print("Collision check complete. Results written to {}".format(output_csv))

if __name__ == "__main__":
    main()
