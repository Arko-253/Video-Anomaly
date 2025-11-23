import numpy as np
import os
import re

GT_FILE = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/UCSDped2.m"
TEST_ROOT = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test"


def parse_matlab_m_file(path):
    """
    Reads UCSDped2.m and extracts the anomaly frame lists.
    Returns a list of lists, one per video.
    """
    with open(path, "r") as f:
        text = f.read()

    # Pattern: [61:180] or [1:146]
    pattern = r"\[(\d+):(\d+)\]"
    matches = re.findall(pattern, text)

    gt = []

    for start, end in matches:
        start = int(start)
        end = int(end)
        frames = list(range(start, end + 1))
        gt.append(frames)

    return gt


def count_video_frames(video_folder):
    return len([f for f in os.listdir(video_folder) if f.endswith(".tif")])


if __name__ == "__main__":
    print("Reading annotations from:", GT_FILE)
    gt_lists = parse_matlab_m_file(GT_FILE)

    all_labels = []

    for i, anomaly_frames in enumerate(gt_lists):
        video_id = f"Test{i+1:03d}"
        video_path = os.path.join(TEST_ROOT, video_id)

        frame_count = count_video_frames(video_path)

        labels = np.zeros(frame_count, dtype=np.uint8)

        for f in anomaly_frames:
            if 1 <= f <= frame_count:
                labels[f - 1] = 1   # convert to 0-index

        all_labels.append(labels)

        print(f"{video_id}: frames={frame_count}, anomalies={len(anomaly_frames)}")

    # Flatten all labels into one long list
    all_labels = np.concatenate(all_labels)

    out_path = os.path.join(TEST_ROOT, "test_labels.npy")
    np.save(out_path, all_labels)

    print("\nSaved:", out_path)
    print("Total labels:", len(all_labels))
