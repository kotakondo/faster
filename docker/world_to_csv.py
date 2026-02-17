#!/usr/bin/env python3
"""Extract cylinder obstacle parameters from a Gazebo .world (SDF) file.

Outputs a CSV with columns: id, x, y, z, radius, height
compatible with collision_checker.py.

Usage:
    python3 world_to_csv.py <world_file> <output_csv>
"""

import sys
import csv
import xml.etree.ElementTree as ET


# Models to skip (not obstacles)
SKIP_MODELS = {"ground_plane", "ground_plane_map", "sun"}


def parse_world(world_path):
    """Parse .world SDF file and extract cylinder obstacles."""
    tree = ET.parse(world_path)
    root = tree.getroot()

    # SDF structure: <sdf><world><model>...</model></world></sdf>
    world = root.find("world")
    if world is None:
        world = root

    cylinders = []
    for model in world.findall("model"):
        name = model.get("name", "")
        if name in SKIP_MODELS:
            continue

        # Find cylinder geometry anywhere in the model
        cylinder = None
        for geom in model.iter("cylinder"):
            cylinder = geom
            break

        if cylinder is None:
            continue

        radius_elem = cylinder.find("radius")
        length_elem = cylinder.find("length")
        if radius_elem is None or length_elem is None:
            continue

        radius = float(radius_elem.text)
        height = float(length_elem.text)

        # Extract pose (x y z roll pitch yaw)
        pose_elem = model.find("pose")
        if pose_elem is None:
            # Try link/pose as fallback
            link = model.find("link")
            if link is not None:
                pose_elem = link.find("pose")

        if pose_elem is not None:
            parts = pose_elem.text.strip().split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        else:
            x, y, z = 0.0, 0.0, 0.0

        cylinders.append({
            "id": name,
            "x": x,
            "y": y,
            "z": z,
            "radius": radius,
            "height": height,
        })

    return cylinders


def main():
    if len(sys.argv) < 3:
        print("Usage: {} <world_file> <output_csv>".format(sys.argv[0]))
        sys.exit(1)

    world_path = sys.argv[1]
    output_csv = sys.argv[2]

    cylinders = parse_world(world_path)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "z", "radius", "height"])
        for cyl in cylinders:
            writer.writerow([
                cyl["id"], cyl["x"], cyl["y"], cyl["z"],
                cyl["radius"], cyl["height"],
            ])

    print("Extracted {} cylinders from {} -> {}".format(
        len(cylinders), world_path, output_csv))


if __name__ == "__main__":
    main()
