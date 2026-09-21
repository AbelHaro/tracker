import argparse

import matplotlib.pyplot as plt
import numpy as np
import tracker

RANDOM_NOISE_SCALE = 0.05


def main() -> None:
    functions = {"linear": linear_function}
    parser = argparse.ArgumentParser(
        description="Plot detections and tracker estimates."
    )
    parser.add_argument(
        "--limits", nargs=2, type=float, default=(0, 100), metavar=("MIN", "MAX")
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--function", choices=functions, default="linear")
    args = parser.parse_args()
    xmin, xmax = args.limits
    if not np.isfinite([xmin, xmax]).all() or xmin >= xmax or xmin < 0:
        parser.error("--limits requires two finite values with MIN < MAX")
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    function = functions[args.function]

    plt.switch_backend("QtAgg")
    plt.ion()

    tracker_config = tracker.ByteTrackerConfig()
    object_tracker = tracker.ByteTracker(tracker_config)

    observations: np.ndarray = np.array(
        [
            [
                [x, function(x), 100, 100, 0.9, 1],
                [x, function(x) + 200, 100, 100, 0.9, 2],
            ]
            for x in np.linspace(xmin, xmax, args.steps)
        ],
        dtype=np.float32,
    )
    ymin, ymax = observations[:, :, 1].min(), observations[:, :, 1].max()
    margin = max(float(ymax - ymin) * 0.1, 1e-6)
    plt.figure(figsize=(8, 5))
    plt.title("Object Tracking")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.xlim(xmin, xmax)
    plt.ylim(float(ymin - margin), float(ymax + margin))

    observed_y, estimated_y = [], []
    id_labels = {}

    for i, detections in enumerate(observations):
        trackings = object_tracker.update(detections)

        print(
            f"Frame {i + 1}/{args.steps}: {len(detections)} detections, {len(trackings)} predictions"
        )

        plt.scatter(
            *detections[:, :2].T,
            c="tab:blue",
            marker="o",
            label="Detections" if i == 0 else None,
        )
        for tracking in trackings:
            x, y = tracking[:2]
            track_id = int(tracking[4])
            color = f"C{track_id % 10}"
            plt.scatter(
                x,
                y,
                c=color,
                marker="x",
                label=f"Prediction ID {track_id}"
                if track_id not in id_labels
                else None,
            )
            if track_id in id_labels:
                id_labels[track_id].remove()
            id_labels[track_id] = plt.annotate(
                f"ID {track_id}",
                (x, y),
                color=color,
                xytext=(-5, 5),
                textcoords="offset points",
                ha="right",
            )
        plt.legend()
        plt.pause(0.001)

        # There is one object per frame; missing estimates have no paired error.
        if len(trackings) == 1:
            observed_y.append(float(detections[0, 1]))
            estimated_y.append(float(trackings[0, 1]))

    if estimated_y:
        error = mse(np.array(observed_y), np.array(estimated_y))
        print(
            f"Y MSE: {error:.4f} ({len(estimated_y)}/{args.steps} observations evaluated)"
        )
    else:
        print("MSE unavailable: no matched estimates.")

    plt.ioff()
    plt.show()


def linear_function(x: float) -> float:
    return (2 * x + 1) * np.random.uniform(
        1 - RANDOM_NOISE_SCALE, 1 + RANDOM_NOISE_SCALE
    )


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("MSE requires non-empty arrays with the same shape")
    return float(np.mean((y_true.astype(np.float64) - y_pred.astype(np.float64)) ** 2))


if __name__ == "__main__":
    main()
