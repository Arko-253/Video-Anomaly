import scipy.io as sio
import numpy as np

def convert_gt_m_to_labels(gt_file, test_folder, total_frames_per_video=200):
    """
    UCSD Ped2 videos all have 120-200 frames.
    Reads gt_frame.m and produces a list of labels.
    """
    gt = sio.loadmat(gt_file)["gt_frame"][0]

    all_labels = []

    # gt_frame is a cell array, each cell contains a list of anomalous frames
    for video_idx in range(len(gt)):
        abnormal_frames = gt[video_idx].flatten()
        labels = np.zeros(total_frames_per_video)

        for f in abnormal_frames:
            if f-1 < len(labels):
                labels[f-1] = 1

        all_labels.append(labels)

    return np.concatenate(all_labels)
