from dataclasses import dataclass


def _iou(a, b) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    det: object
    hits: int = 1
    misses: int = 0


class TemporalFilter:
    def __init__(self, start_conf=0.5, min_hits=3, max_misses=8, iou_thr=0.3):
        self.start_conf = start_conf   # confidence needed to begin a new track
        self.min_hits = min_hits       # frames in a row before it is shown
        self.max_misses = max_misses   # frames a box is kept after a miss
        self.iou_thr = iou_thr         # overlap needed to match across frames
        self.tracks = []

    def update(self, detections):
        unmatched = list(self.tracks)
        new_tracks = []

        for det in sorted(detections, key=lambda d: -d.confidence):
            best, best_iou = None, self.iou_thr
            for t in unmatched:
                v = _iou(det, t.det)
                if v >= best_iou:
                    best, best_iou = t, v
            if best is not None:
                unmatched.remove(best)
                best.det = det
                best.hits += 1
                best.misses = 0
            elif det.confidence >= self.start_conf:
                new_tracks.append(_Track(det))

        for t in unmatched:
            t.misses += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses] + new_tracks
        return [t.det for t in self.tracks if t.hits >= self.min_hits]