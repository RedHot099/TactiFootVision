from collections import defaultdict
from collections.abc import Sequence


def build_frames_by_tid(rows: Sequence[Sequence[float]]) -> dict[int, set[int]]:
    frames_by_tid: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        if len(row) < 2:
            continue
        frames_by_tid[int(row[1])].add(int(row[0]))
    return dict(frames_by_tid)


def compute_identity_stability_ratio(
    frames_by_tid: dict[int, set[int]],
) -> dict[str, float]:
    if not frames_by_tid:
        return {"isr_mean": 0.0, "isr_median": 0.0}
    scores: list[float] = []
    for frames in frames_by_tid.values():
        ordered = sorted(frames)
        if not ordered:
            continue
        longest = current = 1
        for previous, current_frame in zip(ordered, ordered[1:], strict=False):
            if current_frame == previous + 1:
                current += 1
            else:
                longest = max(longest, current)
                current = 1
        longest = max(longest, current)
        scores.append(longest / len(ordered))
    if not scores:
        return {"isr_mean": 0.0, "isr_median": 0.0}
    sorted_scores = sorted(scores)
    mid = len(sorted_scores) // 2
    median = (
        sorted_scores[mid]
        if len(sorted_scores) % 2
        else (sorted_scores[mid - 1] + sorted_scores[mid]) / 2.0
    )
    return {"isr_mean": sum(scores) / len(scores), "isr_median": median}


def compute_all_stability_metrics(rows: Sequence[Sequence[float]]) -> dict[str, float]:
    return compute_identity_stability_ratio(build_frames_by_tid(rows))
