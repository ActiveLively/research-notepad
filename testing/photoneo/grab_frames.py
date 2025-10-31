import argparse
import os
import sys
import time
import threading
import pathlib
from datetime import datetime
import cv2
import requests
import numpy as np
import open3d as o3d

#!/usr/bin/env python3
"""
grab_frames.py

Simple utility to grab color frames (via RTSP/HTTP/OpenCV) and download/load point-clouds (PLY/PCD) from a Photoneo camera.
This script is generic: supply your camera's video stream URL (rtsp/http) and a point-cloud file URL (HTTP returning .ply or .pcd).
It can save periodic snapshots and point-cloud files to disk.

Dependencies:
    - opencv-python
    - requests
    - numpy
    - open3d (optional, only required for visualization/loading into memory)

Example usage:
    python grab_frames.py --video-url "rtsp://192.168.1.100/stream" --pc-url "http://192.168.1.100/pointcloud.ply" --out ./captures --frame-interval 1.0 --pc-interval 2.0

Notes:
    - Photoneo devices may expose different HTTP endpoints for point cloud dumps or may require using the PhoXi SDK / ROS driver.
        If you have the Photoneo SDK, use its Python API for low-latency/camera-triggered capture. This script is useful for quick grabs
        from a web/rtsp stream and downloading produced point cloud files.
"""


try:
except Exception as e:
        print("ERROR: opencv-python is required. pip install opencv-python", file=sys.stderr)
        raise

try:
except Exception as e:
        print("ERROR: requests is required. pip install requests", file=sys.stderr)
        raise

try:
except Exception as e:
        print("ERROR: numpy is required. pip install numpy", file=sys.stderr)
        raise

# open3d is optional
try:
        _HAS_O3D = True
except Exception:
        _HAS_O3D = False


def ensure_dir(path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        return os.path.abspath(path)


def timestamp():
        return datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3] + "Z"


def grab_video_stream(video_url, out_dir, frame_interval=1.0, show=False, stop_event=None):
        """
        Capture frames from video_url using OpenCV VideoCapture.
        Saves images every frame_interval seconds to out_dir.
        """
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
                print(f"Failed to open video stream: {video_url}", file=sys.stderr)
                return

        last_save = 0.0
        frame_count = 0
        print("Video capture started")

        while not (stop_event and stop_event.is_set()):
                ret, frame = cap.read()
                if not ret:
                        # try reconnect after short wait
                        time.sleep(0.5)
                        continue

                frame_count += 1
                now = time.time()
                if now - last_save >= frame_interval:
                        fname = os.path.join(out_dir, f"frame_{timestamp()}_{frame_count:06d}.jpg")
                        cv2.imwrite(fname, frame)
                        last_save = now
                        print(f"Saved frame: {fname}")

                if show:
                        cv2.imshow("frame", frame)
                        # waitKey required for imshow to work
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                                if stop_event:
                                        stop_event.set()
                                break

        cap.release()
        if show:
                cv2.destroyAllWindows()
        print("Video capture stopped")


def download_pointcloud(pc_url, out_dir, pc_interval=2.0, save_only=True, stop_event=None):
        """
        Periodically download a point cloud file from pc_url and save it.
        If open3d is installed and save_only is False, also load it into memory and print simple stats.
        """
        session = requests.Session()
        last_save = 0.0
        iter_count = 0
        print("Point cloud downloader started")

        while not (stop_event and stop_event.is_set()):
                now = time.time()
                if now - last_save >= pc_interval:
                        iter_count += 1
                        try:
                                resp = session.get(pc_url, timeout=10)
                                resp.raise_for_status()
                                # Guess extension from URL or content-type
                                ext = None
                                parsed = os.path.splitext(pc_url)
                                if len(parsed) > 1 and parsed[1]:
                                        ext = parsed[1]
                                ct = resp.headers.get("Content-Type", "")
                                if not ext:
                                        if "ply" in ct:
                                                ext = ".ply"
                                        elif "pcd" in ct:
                                                ext = ".pcd"
                                        else:
                                                ext = ".bin"

                                fname = os.path.join(out_dir, f"pointcloud_{timestamp()}_{iter_count:04d}{ext}")
                                with open(fname, "wb") as f:
                                        f.write(resp.content)
                                print(f"Saved point cloud: {fname}")

                                if not save_only and _HAS_O3D:
                                        try:
                                                pcd = o3d.io.read_point_cloud(fname)
                                                npt = np.asarray(pcd.points).shape[0]
                                                print(f"Loaded point cloud ({npt} points) from {fname}")
                                        except Exception as e:
                                                print(f"Warning: failed to load point cloud with open3d: {e}", file=sys.stderr)

                                last_save = now
                        except Exception as e:
                                print(f"Point cloud download failed: {e}", file=sys.stderr)
                else:
                        time.sleep(0.1)

        print("Point cloud downloader stopped")


def main():
        parser = argparse.ArgumentParser(description="Grab frames and point clouds from Photoneo camera (generic).")
        parser.add_argument("--video-url", type=str, help="RTSP/HTTP/OpenCV-compatible video stream URL (e.g. rtsp://IP/...).")
        parser.add_argument("--pc-url", type=str, help="HTTP URL returning a point cloud file (PLY/PCD).")
        parser.add_argument("--out", type=str, default="./captures", help="Output directory to save frames and point clouds.")
        parser.add_argument("--frame-interval", type=float, default=1.0, help="Seconds between saved frames.")
        parser.add_argument("--pc-interval", type=float, default=2.0, help="Seconds between downloaded point clouds.")
        parser.add_argument("--show", action="store_true", help="Show video frames in a window (press q to quit).")
        parser.add_argument("--save-only", action="store_true", help="Only save point cloud files; don't attempt to load/visualize them.")
        args = parser.parse_args()

        out_dir = ensure_dir(args.out)

        stop_event = threading.Event()
        threads = []

        if args.video_url:
                t = threading.Thread(
                        target=grab_video_stream,
                        args=(args.video_url, out_dir, args.frame_interval, args.show, stop_event),
                        daemon=True,
                )
                threads.append(t)
                t.start()

        if args.pc_url:
                t2 = threading.Thread(
                        target=download_pointcloud,
                        args=(args.pc_url, out_dir, args.pc_interval, args.save_only, stop_event),
                        daemon=True,
                )
                threads.append(t2)
                t2.start()

        if not threads:
                print("No video-url or pc-url provided. Exiting.", file=sys.stderr)
                sys.exit(1)

        try:
                while any(t.is_alive() for t in threads):
                        time.sleep(0.2)
        except KeyboardInterrupt:
                stop_event.set()
                print("Interrupted, stopping...")

        # join threads briefly
        for t in threads:
                t.join(timeout=1.0)

        print("Done.")


if __name__ == "__main__":
        main()