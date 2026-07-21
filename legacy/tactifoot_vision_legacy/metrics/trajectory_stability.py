"""Trajectory stability metrics for MOT evaluation.

This module provides metrics that go beyond standard MOT metrics (MOTA, IDF1, HOTA)
to measure the physical plausibility and temporal consistency of tracked trajectories.

Metrics implemented:
- ISR (Identity Stability Ratio): Measures trajectory continuity
- ORC (Occlusion Recovery Consistency): Measures recovery after occlusions
- DRR (Direction Reversal Rate): Measures trajectory jitter/oscillations
- AOR (Acceleration Outlier Rate): Detects abnormal acceleration events
- PPS (Physical Plausibility Score): Measures adherence to physical constraints
- MSS (Motion Smoothness Score): Measures velocity consistency
- TCI (Trajectory Consistency Index): Aggregate metric
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence, Tuple

import numpy as np


def _build_track_history(
    rows: Sequence[Sequence[float]],
) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """Build track history from MOT rows.
    
    Args:
        rows: MOT format rows [frame, track_id, x, y, w, h, conf, ...]
        
    Returns:
        Dict mapping track_id to list of (frame, cx, cy, w, h) tuples, sorted by frame.
    """
    track_history: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    
    for row in rows:
        if len(row) < 6:
            continue
        try:
            frame = int(row[0])
            tid = int(row[1])
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
        except (ValueError, IndexError):
            continue
        
        cx = x + w / 2.0
        cy = y + h / 2.0
        track_history[tid].append((frame, cx, cy, w, h))
    
    # Sort each track by frame
    for tid in track_history:
        track_history[tid] = sorted(track_history[tid], key=lambda x: x[0])
    
    return dict(track_history)


def _build_frames_by_tid(rows: Sequence[Sequence[float]]) -> dict[int, set[int]]:
    """Build mapping of track_id to set of frames.
    
    Args:
        rows: MOT format rows [frame, track_id, x, y, w, h, conf, ...]
        
    Returns:
        Dict mapping track_id to set of frame numbers.
    """
    frames_by_tid: dict[int, set[int]] = defaultdict(set)
    
    for row in rows:
        if len(row) < 2:
            continue
        try:
            frame = int(row[0])
            tid = int(row[1])
        except (ValueError, IndexError):
            continue
        frames_by_tid[tid].add(frame)
    
    return dict(frames_by_tid)


def compute_isr(frames_by_tid: dict[int, set[int]]) -> dict[str, float]:
    """Compute Identity Stability Ratio for all tracks.
    
    ISR measures trajectory continuity as the ratio of the longest continuous
    segment to the total trajectory length. A continuous segment is a sequence
    of consecutive frames (Δframe = 1).
    
    Args:
        frames_by_tid: Dict mapping track_id to set of frame numbers.
        
    Returns:
        Dict with keys:
        - isr_mean: Mean ISR across all tracks
        - isr_median: Median ISR
        - isr_ge_0.8: Fraction of tracks with ISR >= 0.8
        - isr_ge_0.9: Fraction of tracks with ISR >= 0.9
    """
    if not frames_by_tid:
        return {
            "isr_mean": 0.0,
            "isr_median": 0.0,
            "isr_ge_0.8": 0.0,
            "isr_ge_0.9": 0.0,
        }
    
    isr_values: list[float] = []
    
    for tid, frames in frames_by_tid.items():
        if len(frames) < 2:
            # Single frame track is perfectly stable
            isr_values.append(1.0)
            continue
        
        sorted_frames = sorted(frames)
        
        # Find continuous segments
        segments: list[int] = []
        current_segment_len = 1
        
        for i in range(1, len(sorted_frames)):
            if sorted_frames[i] - sorted_frames[i - 1] == 1:
                current_segment_len += 1
            else:
                segments.append(current_segment_len)
                current_segment_len = 1
        segments.append(current_segment_len)
        
        max_segment = max(segments)
        total_length = len(frames)
        isr = max_segment / total_length
        isr_values.append(isr)
    
    if not isr_values:
        return {
            "isr_mean": 0.0,
            "isr_median": 0.0,
            "isr_ge_0.8": 0.0,
            "isr_ge_0.9": 0.0,
        }
    
    arr = np.array(isr_values)
    return {
        "isr_mean": float(np.mean(arr)),
        "isr_median": float(np.median(arr)),
        "isr_ge_0.8": float(np.mean(arr >= 0.8)),
        "isr_ge_0.9": float(np.mean(arr >= 0.9)),
    }


def compute_orc(
    rows: Sequence[Sequence[float]],
    gap_thresholds: Tuple[int, ...] = (15, 30, 60),
    max_distance_px: float = 100.0,
) -> dict[str, float]:
    """Compute Occlusion Recovery Consistency at multiple gap thresholds.
    
    ORC measures whether the tracker returns to a spatially consistent position
    after gaps in observation. For each gap > k frames, we check if the distance
    between positions before and after the gap is within a threshold.
    
    Args:
        rows: MOT format rows [frame, track_id, x, y, w, h, conf, ...]
        gap_thresholds: Tuple of gap thresholds in frames (e.g., 15=0.5s @30fps)
        max_distance_px: Maximum allowed distance in pixels for consistent recovery
        
    Returns:
        Dict with keys like "orc@15", "orc@30", "orc@60" with values 0-1
    """
    track_history = _build_track_history(rows)
    
    if not track_history:
        return {f"orc@{k}": 1.0 for k in gap_thresholds}
    
    results: dict[str, float] = {}
    
    for gap_k in gap_thresholds:
        recoveries_total = 0
        recoveries_consistent = 0
        
        for tid, history in track_history.items():
            if len(history) < 2:
                continue
            
            for i in range(1, len(history)):
                frame_gap = history[i][0] - history[i - 1][0]
                
                if frame_gap > gap_k:
                    recoveries_total += 1
                    
                    # Check spatial consistency (centroid distance)
                    cx_before, cy_before = history[i - 1][1], history[i - 1][2]
                    cx_after, cy_after = history[i][1], history[i][2]
                    dist = np.sqrt((cx_after - cx_before) ** 2 + (cy_after - cy_before) ** 2)
                    
                    if dist < max_distance_px:
                        recoveries_consistent += 1
        
        orc = recoveries_consistent / recoveries_total if recoveries_total > 0 else 1.0
        results[f"orc@{gap_k}"] = float(orc)
    
    return results


def compute_drr(
    rows: Sequence[Sequence[float]],
    angle_threshold_deg: float = 90.0,
    min_velocity_px: float = 2.0,
) -> dict[str, float]:
    """Compute Direction Reversal Rate - measures trajectory jitter/oscillations.
    
    DRR detects rapid direction changes that indicate tracking instability.
    A direction reversal occurs when the angle between consecutive velocity 
    vectors exceeds the threshold (e.g., 90 degrees = going back on yourself).
    
    This is more sensitive than TCVR because it detects subtle jitter, not just
    catastrophic teleportations. Lower DRR is better.
    
    Args:
        rows: MOT format rows [frame, track_id, x, y, w, h, conf, ...]
        angle_threshold_deg: Minimum angle change to count as a reversal (degrees)
        min_velocity_px: Minimum velocity magnitude to consider (filters out stationary noise)
        
    Returns:
        Dict with keys:
        - drr: Mean direction reversal rate across tracks (reversals per frame, lower is better)
        - drr_tracks_affected: Fraction of tracks with at least one reversal
        - drr_total_reversals: Total reversal events
    """
    track_history = _build_track_history(rows)
    
    if not track_history:
        return {
            "drr": 0.0,
            "drr_tracks_affected": 0.0,
            "drr_total_reversals": 0,
        }
    
    angle_threshold_rad = np.deg2rad(angle_threshold_deg)
    total_reversals = 0
    total_transitions = 0
    tracks_with_reversals = 0
    
    for tid, history in track_history.items():
        if len(history) < 3:
            continue
        
        # Compute velocity vectors
        velocities: list[tuple[float, float]] = []
        for i in range(1, len(history)):
            dt = history[i][0] - history[i - 1][0]
            if dt == 0:
                continue
            vx = (history[i][1] - history[i - 1][1]) / dt
            vy = (history[i][2] - history[i - 1][2]) / dt
            velocities.append((vx, vy))
        
        if len(velocities) < 2:
            continue
        
        # Count direction reversals
        track_reversals = 0
        for i in range(1, len(velocities)):
            v1 = velocities[i - 1]
            v2 = velocities[i]
            
            mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
            
            # Skip if either velocity is too small (stationary)
            if mag1 < min_velocity_px or mag2 < min_velocity_px:
                continue
            
            # Compute angle between velocity vectors
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            cos_angle = dot / (mag1 * mag2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            total_transitions += 1
            if angle > angle_threshold_rad:
                track_reversals += 1
        
        total_reversals += track_reversals
        if track_reversals > 0:
            tracks_with_reversals += 1
    
    drr = total_reversals / total_transitions if total_transitions > 0 else 0.0
    tracks_affected = tracks_with_reversals / len(track_history) if track_history else 0.0
    
    return {
        "drr": float(drr),
        "drr_tracks_affected": float(tracks_affected),
        "drr_total_reversals": int(total_reversals),
    }


def compute_aor(
    rows: Sequence[Sequence[float]],
    sigma_threshold: float = 2.0,
) -> dict[str, float]:
    """Compute Acceleration Outlier Rate - detects abnormal acceleration events.
    
    AOR measures the fraction of frames where acceleration exceeds a statistical
    threshold (N standard deviations from the mean). High AOR indicates tracking
    instability with sudden jumps.
    
    Unlike TCVR which uses a fixed physical threshold, AOR adapts to each
    trajectory's characteristics, making it more sensitive to relative anomalies.
    
    Args:
        rows: MOT format rows [frame, track_id, x, y, w, h, conf, ...]
        sigma_threshold: Number of standard deviations to consider as outlier
        
    Returns:
        Dict with keys:
        - aor: Mean acceleration outlier rate (fraction of outlier frames, lower is better)
        - aor_median: Median AOR across tracks
        - aor_total_outliers: Total number of outlier acceleration events
    """
    track_history = _build_track_history(rows)
    
    if not track_history:
        return {
            "aor": 0.0,
            "aor_median": 0.0,
            "aor_total_outliers": 0,
        }
    
    track_aor_values: list[float] = []
    total_outliers = 0
    
    for tid, history in track_history.items():
        if len(history) < 4:
            # Too few points for meaningful acceleration analysis
            track_aor_values.append(0.0)
            continue
        
        # Compute velocity vectors
        velocities: list[tuple[float, float]] = []
        for i in range(1, len(history)):
            dt = history[i][0] - history[i - 1][0]
            if dt == 0:
                continue
            vx = (history[i][1] - history[i - 1][1]) / dt
            vy = (history[i][2] - history[i - 1][2]) / dt
            velocities.append((vx, vy))
        
        if len(velocities) < 2:
            track_aor_values.append(0.0)
            continue
        
        # Compute acceleration magnitudes
        accelerations: list[float] = []
        for i in range(1, len(velocities)):
            ax = velocities[i][0] - velocities[i - 1][0]
            ay = velocities[i][1] - velocities[i - 1][1]
            accel_mag = float(np.sqrt(ax**2 + ay**2))
            accelerations.append(accel_mag)
        
        if len(accelerations) < 2:
            track_aor_values.append(0.0)
            continue
        
        # Compute outliers based on statistical threshold
        accel_arr = np.array(accelerations)
        mean_accel = np.mean(accel_arr)
        std_accel = np.std(accel_arr)
        
        if std_accel < 1e-6:
            # No variance - no outliers
            track_aor_values.append(0.0)
            continue
        
        threshold = mean_accel + sigma_threshold * std_accel
        outliers = np.sum(accel_arr > threshold)
        aor = outliers / len(accelerations)
        
        track_aor_values.append(float(aor))
        total_outliers += int(outliers)
    
    if not track_aor_values:
        return {
            "aor": 0.0,
            "aor_median": 0.0,
            "aor_total_outliers": 0,
        }
    
    arr = np.array(track_aor_values)
    return {
        "aor": float(np.mean(arr)),
        "aor_median": float(np.median(arr)),
        "aor_total_outliers": int(total_outliers),
    }


def compute_pps(
    rows: Sequence[Sequence[float]],
    max_speed_m_per_s: float = 12.0,
    max_accel_m_per_s2: float = 6.0,
    pixels_per_meter: float = 18.3,  # ~1920px / 105m field
    frame_rate: float = 25.0,
) -> dict[str, float]:
    """Compute Physical Plausibility Score - measures adherence to physical constraints.
    
    PPS evaluates whether trajectories respect the physical limits of human movement.
    It checks both speed and acceleration against biomechanical limits for football players.
    
    Physical constraints (default values based on elite athletes):
    - Max sprint speed: 12 m/s (~43 km/h, Usain Bolt peak)
    - Max acceleration: 6 m/s² (typical peak acceleration for football players)
    
    PPS is computed as the mean per-track plausibility rate, where a track's score
    is 1 - (violating_steps / total_steps). This avoids the metric collapsing to
    zero when a single outlier occurs.
    
    Args:
        rows: MOT format rows [frame, track_id, x, y, w, h, conf, ...]
        max_speed_m_per_s: Maximum plausible speed in m/s
        max_accel_m_per_s2: Maximum plausible acceleration in m/s²
        pixels_per_meter: Conversion factor (depends on camera/field setup)
        frame_rate: Video frame rate in fps
        
    Returns:
        Dict with keys:
        - pps: Physical plausibility score (mean per-track plausibility, higher is better)
        - pps_speed_violations: Total count of speed violations
        - pps_accel_violations: Total count of acceleration violations
        - pps_max_speed_observed: Maximum observed speed in m/s
        - pps_max_accel_observed: Maximum observed acceleration in m/s²
    """
    track_history = _build_track_history(rows)
    
    if not track_history:
        return {
            "pps": 1.0,
            "pps_speed_violations": 0,
            "pps_accel_violations": 0,
            "pps_max_speed_observed": 0.0,
            "pps_max_accel_observed": 0.0,
        }
    
    # Convert thresholds to pixels/frame
    max_speed_px_per_frame = max_speed_m_per_s * pixels_per_meter / frame_rate
    max_accel_px_per_frame2 = max_accel_m_per_s2 * pixels_per_meter / (frame_rate ** 2)
    
    track_scores: list[float] = []
    speed_violations = 0
    accel_violations = 0
    max_speed_observed_px = 0.0
    max_accel_observed_px = 0.0
    
    for tid, history in track_history.items():
        if len(history) < 2:
            track_scores.append(1.0)
            continue
        
        track_speed_violations = 0
        track_accel_violations = 0
        total_steps = 0
        
        # Compute velocities
        velocities: list[tuple[float, float, float]] = []  # (vx, vy, speed)
        for i in range(1, len(history)):
            dt = history[i][0] - history[i - 1][0]
            if dt == 0:
                continue
            vx = (history[i][1] - history[i - 1][1]) / dt
            vy = (history[i][2] - history[i - 1][2]) / dt
            speed = np.sqrt(vx**2 + vy**2)
            velocities.append((vx, vy, speed))
            
            max_speed_observed_px = max(max_speed_observed_px, speed)
            total_steps += 1
            if speed > max_speed_px_per_frame:
                track_speed_violations += 1
        
        # Compute accelerations
        for i in range(1, len(velocities)):
            ax = velocities[i][0] - velocities[i - 1][0]
            ay = velocities[i][1] - velocities[i - 1][1]
            accel = np.sqrt(ax**2 + ay**2)
            
            max_accel_observed_px = max(max_accel_observed_px, accel)
            total_steps += 1
            if accel > max_accel_px_per_frame2:
                track_accel_violations += 1
        
        total_violations = track_speed_violations + track_accel_violations
        if total_steps > 0:
            track_score = 1.0 - (total_violations / total_steps)
        else:
            track_score = 1.0
        track_scores.append(float(track_score))
        speed_violations += track_speed_violations
        accel_violations += track_accel_violations
    
    pps = float(np.mean(track_scores)) if track_scores else 1.0
    
    # Convert observed values back to physical units
    max_speed_m_s = max_speed_observed_px * frame_rate / pixels_per_meter
    max_accel_m_s2 = max_accel_observed_px * (frame_rate ** 2) / pixels_per_meter
    
    return {
        "pps": float(pps),
        "pps_speed_violations": int(speed_violations),
        "pps_accel_violations": int(accel_violations),
        "pps_max_speed_observed": float(max_speed_m_s),
        "pps_max_accel_observed": float(max_accel_m_s2),
    }


def compute_tcvr(
    rows: Sequence[Sequence[float]],
    image_width: int = 1920,
    max_speed_px_per_frame: float = 15.0,
) -> dict[str, float]:
    """Compute Team-Consistency Violation Rate and teleportation count.
    
    TCVR detects "teleportations" - physically impossible movements that indicate
    ID switches. A trajectory has a violation if any position jump exceeds the
    physically possible speed.
    
    Default assumptions for broadcast football footage:
    - Field ~105m mapped to ~1920px width
    - Max sprint speed ~35 km/h = ~10 m/s
    - At 30fps: ~15 px/frame max (with safety margin)
    
    Args:
        rows: MOT format rows [frame, track_id, x, y, w, h, conf, ...]
        image_width: Image width in pixels (for reference, not used in current impl)
        max_speed_px_per_frame: Maximum allowed speed in pixels per frame
        
    Returns:
        Dict with keys:
        - tcvr: Violation rate 0-1 (lower is better) - fraction of tracks with any violation
        - tcvr_violations: Number of tracks with at least one violation
        - tcvr_total_tracks: Total number of tracks
        - tcvr_teleportations_total: Total count of individual teleportation events
        - tcvr_teleportations_per_track: Average teleportations per track (all tracks)
        - tcvr_teleportations_per_violated: Average teleportations per violated track
        - tcvr_max_teleportations: Max teleportations in any single track
    """
    track_history = _build_track_history(rows)
    
    if not track_history:
        return {
            "tcvr": 0.0,
            "tcvr_violations": 0,
            "tcvr_total_tracks": 0,
            "tcvr_teleportations_total": 0,
            "tcvr_teleportations_per_track": 0.0,
            "tcvr_teleportations_per_violated": 0.0,
            "tcvr_max_teleportations": 0,
        }
    
    tracks_with_violations = 0
    total_tracks = len(track_history)
    total_teleportations = 0
    teleportations_per_track: list[int] = []
    
    for tid, history in track_history.items():
        if len(history) < 2:
            teleportations_per_track.append(0)
            continue
        
        track_teleportations = 0
        for i in range(1, len(history)):
            frame_gap = history[i][0] - history[i - 1][0]
            if frame_gap == 0:
                continue
            
            # Focus on X-axis (horizontal) for cross-field teleportation
            x_delta = abs(history[i][1] - history[i - 1][1])
            speed_x = x_delta / frame_gap
            
            if speed_x > max_speed_px_per_frame:
                track_teleportations += 1
        
        teleportations_per_track.append(track_teleportations)
        total_teleportations += track_teleportations
        if track_teleportations > 0:
            tracks_with_violations += 1
    
    tcvr = tracks_with_violations / total_tracks if total_tracks > 0 else 0.0
    avg_per_track = total_teleportations / total_tracks if total_tracks > 0 else 0.0
    avg_per_violated = (
        total_teleportations / tracks_with_violations 
        if tracks_with_violations > 0 else 0.0
    )
    max_teleportations = max(teleportations_per_track) if teleportations_per_track else 0
    
    return {
        "tcvr": float(tcvr),
        "tcvr_violations": int(tracks_with_violations),
        "tcvr_total_tracks": int(total_tracks),
        "tcvr_teleportations_total": int(total_teleportations),
        "tcvr_teleportations_per_track": float(avg_per_track),
        "tcvr_teleportations_per_violated": float(avg_per_violated),
        "tcvr_max_teleportations": int(max_teleportations),
    }


def compute_mss(rows: Sequence[Sequence[float]], max_accel_px_per_frame2: float = 5.0) -> dict[str, float]:
    """Compute Motion Smoothness Score based on acceleration.
    
    MSS measures trajectory smoothness using acceleration magnitude. A smooth 
    trajectory has low acceleration (constant velocity motion). This approach
    is more robust than velocity-based smoothness because:
    - It ignores constant velocity differences between objects
    - It focuses on sudden direction/speed changes that indicate tracking errors
    - It's less sensitive to mask centroid jitter (which affects velocity but not acceleration)
    
    For each trajectory, we compute:
    - accelerations = |v[i] - v[i-1]| for each pair of consecutive velocity vectors
    - mean_accel = mean(accelerations)
    - MSS = 1 / (1 + mean_accel / normalization_factor)
    
    Args:
        rows: MOT format rows [frame, track_id, x, y, w, h, conf, ...]
        max_accel_px_per_frame2: Normalization factor for acceleration (pixels/frame^2)
        
    Returns:
        Dict with keys:
        - mss_mean: Mean MSS across all tracks
        - mss_median: Median MSS
    """
    track_history = _build_track_history(rows)
    
    if not track_history:
        return {
            "mss_mean": 0.0,
            "mss_median": 0.0,
        }
    
    smoothness_scores: list[float] = []
    
    for tid, history in track_history.items():
        if len(history) < 4:
            # Too few points to compute meaningful acceleration
            smoothness_scores.append(1.0)
            continue
        
        # Compute velocity vectors
        velocities: list[tuple[float, float]] = []
        
        for i in range(1, len(history)):
            dt = history[i][0] - history[i - 1][0]
            if dt == 0:
                continue
            
            vx = (history[i][1] - history[i - 1][1]) / dt
            vy = (history[i][2] - history[i - 1][2]) / dt
            velocities.append((vx, vy))
        
        if len(velocities) < 2:
            smoothness_scores.append(1.0)
            continue
        
        # Compute acceleration magnitudes (change in velocity)
        accelerations: list[float] = []
        for i in range(1, len(velocities)):
            ax = velocities[i][0] - velocities[i - 1][0]
            ay = velocities[i][1] - velocities[i - 1][1]
            accel_mag = float(np.sqrt(ax ** 2 + ay ** 2))
            accelerations.append(accel_mag)
        
        if not accelerations:
            smoothness_scores.append(1.0)
            continue
        
        mean_accel = float(np.mean(accelerations))
        # Normalize: MSS = 1 for zero acceleration, decreases for higher acceleration
        mss = 1.0 / (1.0 + mean_accel / max_accel_px_per_frame2)
        smoothness_scores.append(mss)
    
    if not smoothness_scores:
        return {
            "mss_mean": 0.0,
            "mss_median": 0.0,
        }
    
    arr = np.array(smoothness_scores)
    return {
        "mss_mean": float(np.mean(arr)),
        "mss_median": float(np.median(arr)),
    }


def compute_tci(
    isr_mean: float,
    drr: float,
    aor: float,
    pps: float,
    orc_30: float,
    weights: Tuple[float, float, float, float, float] = (0.25, 0.20, 0.15, 0.25, 0.15),
) -> float:
    """Compute Trajectory Consistency Index.
    
    TCI is an aggregate metric combining ISR, DRR, AOR, PPS, and ORC into a single
    score for easy comparison between trackers.
    
    Formula:
        TCI = w1*ISR + w2*(1-DRR) + w3*(1-AOR) + w4*PPS + w5*ORC@30
    
    Note: DRR and AOR are inverted (1-x) because lower is better for those metrics.
    
    Args:
        isr_mean: Mean Identity Stability Ratio (higher is better)
        drr: Direction Reversal Rate (lower is better)
        aor: Acceleration Outlier Rate (lower is better)
        pps: Physical Plausibility Score (higher is better)
        orc_30: Occlusion Recovery Consistency at 30-frame gap (higher is better)
        weights: Weights for (ISR, 1-DRR, 1-AOR, PPS, ORC) components
        
    Returns:
        TCI value 0-1 (higher is better)
    """
    w1, w2, w3, w4, w5 = weights
    
    # Clamp inputs to valid ranges
    isr_mean = max(0.0, min(1.0, isr_mean))
    drr = max(0.0, min(1.0, drr))
    aor = max(0.0, min(1.0, aor))
    pps = max(0.0, min(1.0, pps))
    orc_30 = max(0.0, min(1.0, orc_30))
    
    tci = w1 * isr_mean + w2 * (1.0 - drr) + w3 * (1.0 - aor) + w4 * pps + w5 * orc_30
    
    return float(tci)


def compute_all_stability_metrics(
    rows: Sequence[Sequence[float]],
    image_width: int = 1920,
    frame_rate: int = 25,
    orc_gap_thresholds: Tuple[int, ...] = (15, 30, 60),
    orc_max_distance_px: float = 100.0,
    drr_angle_threshold_deg: float = 90.0,
    pps_max_speed_m_per_s: float = 12.0,
    pps_max_accel_m_per_s2: float = 6.0,
    pps_pixels_per_meter: float = 18.3,
    tci_weights: Tuple[float, float, float, float, float] = (0.25, 0.20, 0.15, 0.25, 0.15),
) -> dict[str, float]:
    """Compute all trajectory stability metrics in one pass.
    
    This is the main entry point for computing stability metrics from MOT rows.
    
    Args:
        rows: MOT format rows [frame, track_id, x, y, w, h, conf, ...]
        image_width: Image width in pixels
        frame_rate: Video frame rate
        orc_gap_thresholds: Gap thresholds for ORC computation
        orc_max_distance_px: Max distance for ORC consistency
        drr_angle_threshold_deg: Angle threshold for direction reversal detection
        pps_max_speed_m_per_s: Max plausible speed for PPS
        pps_max_accel_m_per_s2: Max plausible acceleration for PPS
        pps_pixels_per_meter: Conversion factor for PPS
        tci_weights: Weights for TCI aggregation (ISR, 1-DRR, 1-AOR, PPS, ORC)
        
    Returns:
        Dict with all stability metrics:
        - isr_mean, isr_median, isr_ge_0.8, isr_ge_0.9
        - orc@15, orc@30, orc@60
        - drr, drr_tracks_affected, drr_total_reversals
        - aor, aor_median, aor_total_outliers
        - pps, pps_speed_violations, pps_accel_violations, pps_max_speed/accel_observed
        - mss_mean, mss_median
        - tci
    """
    # Build common data structures
    frames_by_tid = _build_frames_by_tid(rows)
    
    # Compute individual metrics
    isr_metrics = compute_isr(frames_by_tid)
    orc_metrics = compute_orc(rows, gap_thresholds=orc_gap_thresholds, max_distance_px=orc_max_distance_px)
    drr_metrics = compute_drr(rows, angle_threshold_deg=drr_angle_threshold_deg)
    aor_metrics = compute_aor(rows)
    pps_metrics = compute_pps(
        rows, 
        max_speed_m_per_s=pps_max_speed_m_per_s,
        max_accel_m_per_s2=pps_max_accel_m_per_s2,
        pixels_per_meter=pps_pixels_per_meter,
        frame_rate=float(frame_rate),
    )
    mss_metrics = compute_mss(rows)
    
    # Compute aggregate TCI
    tci = compute_tci(
        isr_mean=isr_metrics["isr_mean"],
        drr=drr_metrics["drr"],
        aor=aor_metrics["aor"],
        pps=pps_metrics["pps"],
        orc_30=orc_metrics.get("orc@30", 1.0),
        weights=tci_weights,
    )
    
    # Combine all metrics
    result: dict[str, float] = {}
    result.update(isr_metrics)
    result.update(orc_metrics)
    result.update(drr_metrics)
    result.update(aor_metrics)
    result.update(pps_metrics)
    result.update(mss_metrics)
    result["tci"] = tci
    
    return result
