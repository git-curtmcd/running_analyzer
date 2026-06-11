import argparse
import json
import os
import statistics
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


COCO_KEYPOINTS = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


def calc_angle(a, b, c):
    """Angle at point b, formed by a-b-c."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def calc_trunk_lean(shoulder_mid, hip_mid):
    vertical = np.array([0, -1])
    torso = shoulder_mid - hip_mid
    cos_angle = np.dot(torso, vertical) / (np.linalg.norm(torso) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def safe_point(person, name):
    point = person[COCO_KEYPOINTS[name]]
    if point[0] <= 0 and point[1] <= 0:
        return None
    return point


def summarize(values):
    clean = [float(v) for v in values if np.isfinite(v)]
    if not clean:
        return None
    return {
        "min": round(min(clean), 1),
        "max": round(max(clean), 1),
        "avg": round(statistics.mean(clean), 1),
    }


def estimate_cadence(left_ankle_y, right_ankle_y, fps):
    if fps <= 0:
        return None

    peaks = 0
    for series in (left_ankle_y, right_ankle_y):
        if len(series) < 5:
            continue
        centered = np.array(series) - np.mean(series)
        threshold = np.std(centered) * 0.4
        for idx in range(1, len(centered) - 1):
            if (
                centered[idx] > threshold
                and centered[idx] > centered[idx - 1]
                and centered[idx] > centered[idx + 1]
            ):
                peaks += 1

    duration_minutes = (max(len(left_ankle_y), len(right_ankle_y)) / fps) / 60
    if duration_minutes <= 0 or peaks == 0:
        return None
    return round(peaks / duration_minutes)


def choose_person(keypoints):
    if len(keypoints) == 0:
        return None
    # The runner is usually the largest/clearest detection; YOLO returns people in confidence order.
    return keypoints[0]


def draw_metric(frame, label, point, color):
    x, y = int(point[0]), int(point[1])
    cv2.putText(frame, label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def analyze_video(args):
    source = 0 if args.source == "webcam" else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = round(frame_count / fps, 1) if frame_count > 0 else None

    writer = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    model = YOLO(args.model_path)
    metrics = defaultdict(list)
    frames_analyzed = 0
    frame_index = 0

    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % args.sample_every != 0:
            if writer:
                writer.write(frame)
            frame_index += 1
            continue

        results = model(frame, verbose=False)
        person = None
        if results and results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()
            person = choose_person(keypoints)

        if person is not None:
            left_hip = safe_point(person, "left_hip")
            right_hip = safe_point(person, "right_hip")
            left_knee = safe_point(person, "left_knee")
            right_knee = safe_point(person, "right_knee")
            left_ankle = safe_point(person, "left_ankle")
            right_ankle = safe_point(person, "right_ankle")
            left_shoulder = safe_point(person, "left_shoulder")
            right_shoulder = safe_point(person, "right_shoulder")

            if all(p is not None for p in (left_hip, left_knee, left_ankle)):
                left_knee_angle = calc_angle(left_hip, left_knee, left_ankle)
                metrics["left_knee_angle"].append(left_knee_angle)
                draw_metric(frame, f"L knee: {left_knee_angle:.0f}", left_knee, (0, 200, 255))

            if all(p is not None for p in (right_hip, right_knee, right_ankle)):
                right_knee_angle = calc_angle(right_hip, right_knee, right_ankle)
                metrics["right_knee_angle"].append(right_knee_angle)
                draw_metric(frame, f"R knee: {right_knee_angle:.0f}", right_knee, (0, 200, 255))

            if all(p is not None for p in (left_shoulder, right_shoulder, left_hip, right_hip)):
                shoulder_mid = (left_shoulder + right_shoulder) / 2
                hip_mid = (left_hip + right_hip) / 2
                trunk_lean = calc_trunk_lean(shoulder_mid, hip_mid)
                metrics["trunk_lean"].append(trunk_lean)
                draw_metric(frame, f"Trunk: {trunk_lean:.0f} deg", hip_mid, (255, 150, 0))

            if left_ankle is not None:
                metrics["left_ankle_y"].append(left_ankle[1])
            if right_ankle is not None:
                metrics["right_ankle_y"].append(right_ankle[1])

            frames_analyzed += 1

        if writer:
            writer.write(frame)

        if args.show:
            cv2.imshow("Running Form Analyzer", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_index += 1

    capture.release()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    analyzed_fps = fps / max(args.sample_every, 1)
    summary = {
        "source": args.source,
        "duration_seconds": duration_seconds,
        "frames_analyzed": frames_analyzed,
        "sample_every": args.sample_every,
        "left_knee_angle_degrees": summarize(metrics["left_knee_angle"]),
        "right_knee_angle_degrees": summarize(metrics["right_knee_angle"]),
        "trunk_lean_degrees": summarize(metrics["trunk_lean"]),
        "estimated_cadence_spm": estimate_cadence(
            metrics["left_ankle_y"], metrics["right_ankle_y"], analyzed_fps
        ),
    }
    return summary


def build_coach_prompt(summary):
    return f"""
You are a running form coach. Analyze these pose-estimation metrics from a runner's video.
Give concise, practical feedback.

Important limitations:
- The metrics come from 2D video pose estimation, so mention uncertainty when appropriate.
- Do not diagnose injuries or give medical advice.
- Focus on observable running form and drills/cues.

Metrics:
{json.dumps(summary, indent=2)}

Respond with:
1. Top 3 observations
2. Likely form priorities
3. 3 specific cues or drills to try next run
4. What video angle or extra data would improve the analysis
""".strip()


def ask_ollama(summary, host, model):
    url = f"http://{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": build_coach_prompt(summary),
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "").strip()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {url}: {exc.reason}") from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze running form from a webcam or local video.")
    parser.add_argument(
        "--source",
        default="webcam",
        help='Use "webcam" or a local video path, for example ./runs/side-view.mp4.',
    )
    parser.add_argument("--model-path", default="yolov8n-pose.pt", help="YOLO pose model path.")
    parser.add_argument("--sample-every", type=int, default=3, help="Analyze every Nth frame.")
    parser.add_argument("--output", help="Optional path for an annotated MP4.")
    parser.add_argument("--show", action="store_true", help="Preview the annotated video while analyzing.")
    parser.add_argument("--json-out", help="Optional path to save the numeric metric summary.")
    parser.add_argument("--ollama", action="store_true", help="Send the metric summary to Ollama.")
    parser.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", "10.0.0.201:11434"))
    parser.add_argument("--ollama-model", default=os.getenv("OLLAMA_MODEL", "llama3.2"))
    return parser.parse_args()


def main():
    args = parse_args()
    summary = analyze_video(args)
    print(json.dumps(summary, indent=2))

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.ollama:
        print("\nOllama coaching feedback:\n")
        print(ask_ollama(summary, args.ollama_host, args.ollama_model))


if __name__ == "__main__":
    main()
