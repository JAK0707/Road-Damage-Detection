#Sort tracker implementation

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

def iou(bb_test, bb_gt):
    """Compute Intersection-over-Union (IoU) between two bounding boxes."""
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + 
              (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o

class KalmanBoxTracker:
    """Represents the state of a tracked object."""
    count = 0

    def __init__(self, bbox):
        """Initialize tracker using initial bounding box."""
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)  # State transition matrix
        self.kf.H = np.eye(4, 7)  # Measurement matrix
        self.kf.R[2:, 2:] *= 10.  # Measurement noise
        self.kf.P[4:, 4:] *= 1000.  # State uncertainty
        self.kf.P *= 10.  # Initial state covariance
        self.kf.x[:4] = bbox.reshape((4, 1))  # Initial state
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []

    def update(self, bbox):
        """Update state vector with observed bounding box."""
        self.time_since_update = 0
        self.kf.update(bbox)

    def predict(self):
        """Advance the state vector and return predicted bounding box."""
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.x[:4].reshape((1, 4))

class Sort:
    """SORT: Simple Online and Realtime Tracking."""
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age  # Maximum age of missed updates before deletion
        self.min_hits = min_hits  # Minimum detections to initialize tracking
        self.iou_threshold = iou_threshold  # IoU threshold for matching
        self.trackers = []  # List of active trackers

    def update(self, detections):
        """Update trackers with new detections."""
        # Predict all tracker positions
        trks = np.zeros((len(self.trackers), 4))
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t, :] = pos

        # Match detections to trackers using IoU
        matched, unmatched_dets, unmatched_trks = self._associate(detections, trks)

        # Update matched trackers with new detections
        for m in matched:
            self.trackers[m[1]].update(detections[m[0], :])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[i]))

        # Return updated tracker information with IDs
        return np.array([[*trk.predict()[0], trk.id] for trk in self.trackers])

    def _associate(self, detections, trks):
        """Associate detections to trackers using IoU."""
        iou_matrix = np.zeros((len(detections), len(trks)))

        # Compute IoU between all detections and trackers
        for d, det in enumerate(detections):
            for t, trk in enumerate(trks):
                iou_matrix[d, t] = iou(det, trk)

        # Match using the Hungarian algorithm (minimize negative IoU)
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(matched_indices).T

        unmatched_dets = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
        unmatched_trks = [t for t in range(len(trks)) if t not in matched_indices[:, 1]]

        return matched_indices, unmatched_dets, unmatched_trks
