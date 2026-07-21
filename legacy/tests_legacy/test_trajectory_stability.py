"""Unit tests for trajectory stability metrics."""

import pytest
import numpy as np

from tactifoot_vision.metrics.trajectory_stability import (
    compute_aor,
    compute_drr,
    compute_isr,
    compute_orc,
    compute_pps,
    compute_tcvr,
    compute_mss,
    compute_tci,
    compute_all_stability_metrics,
    _build_frames_by_tid,
    _build_track_history,
)


class TestBuildFramesByTid:
    """Tests for _build_frames_by_tid helper."""
    
    def test_empty_rows(self):
        result = _build_frames_by_tid([])
        assert result == {}
    
    def test_single_track(self):
        rows = [
            [1, 1, 100, 100, 50, 50, 1.0, -1, -1, -1],
            [2, 1, 105, 100, 50, 50, 1.0, -1, -1, -1],
            [3, 1, 110, 100, 50, 50, 1.0, -1, -1, -1],
        ]
        result = _build_frames_by_tid(rows)
        assert result == {1: {1, 2, 3}}
    
    def test_multiple_tracks(self):
        rows = [
            [1, 1, 100, 100, 50, 50, 1.0, -1, -1, -1],
            [1, 2, 200, 100, 50, 50, 1.0, -1, -1, -1],
            [2, 1, 105, 100, 50, 50, 1.0, -1, -1, -1],
        ]
        result = _build_frames_by_tid(rows)
        assert result == {1: {1, 2}, 2: {1}}


class TestComputeISR:
    """Tests for Identity Stability Ratio computation."""
    
    def test_empty_input(self):
        result = compute_isr({})
        assert result["isr_mean"] == 0.0
        assert result["isr_median"] == 0.0
        assert result["isr_ge_0.8"] == 0.0
        assert result["isr_ge_0.9"] == 0.0
    
    def test_perfect_continuous_trajectory(self):
        """A trajectory with consecutive frames should have ISR = 1.0"""
        frames_by_tid = {1: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
        result = compute_isr(frames_by_tid)
        assert result["isr_mean"] == 1.0
        assert result["isr_median"] == 1.0
        assert result["isr_ge_0.8"] == 1.0
        assert result["isr_ge_0.9"] == 1.0
    
    def test_fragmented_trajectory(self):
        """A trajectory with gaps should have ISR < 1.0"""
        # 10 frames with a gap: 1-5 (5 frames) and 10-14 (5 frames)
        frames_by_tid = {1: {1, 2, 3, 4, 5, 10, 11, 12, 13, 14}}
        result = compute_isr(frames_by_tid)
        # Max segment = 5, total = 10, ISR = 0.5
        assert result["isr_mean"] == 0.5
        assert result["isr_ge_0.8"] == 0.0  # 0.5 < 0.8
    
    def test_single_frame_trajectory(self):
        """Single frame trajectory should have ISR = 1.0 (perfectly stable)"""
        frames_by_tid = {1: {5}}
        result = compute_isr(frames_by_tid)
        assert result["isr_mean"] == 1.0
    
    def test_multiple_trajectories(self):
        """Average ISR across multiple trajectories"""
        frames_by_tid = {
            1: {1, 2, 3, 4, 5},  # ISR = 1.0 (continuous)
            2: {1, 2, 10, 11},   # ISR = 0.5 (2 segments of 2)
        }
        result = compute_isr(frames_by_tid)
        expected_mean = (1.0 + 0.5) / 2
        assert abs(result["isr_mean"] - expected_mean) < 1e-6


class TestComputeORC:
    """Tests for Occlusion Recovery Consistency computation."""
    
    def test_empty_input(self):
        result = compute_orc([])
        assert result["orc@15"] == 1.0
        assert result["orc@30"] == 1.0
        assert result["orc@60"] == 1.0
    
    def test_no_gaps(self):
        """Continuous trajectory with no gaps should have ORC = 1.0"""
        rows = [[i, 1, 100 + i, 100, 50, 50, 1.0, -1, -1, -1] for i in range(1, 50)]
        result = compute_orc(rows)
        assert result["orc@15"] == 1.0
        assert result["orc@30"] == 1.0
    
    def test_consistent_recovery(self):
        """Recovery within distance threshold should be counted as consistent"""
        rows = []
        # Frames 1-10
        for i in range(1, 11):
            rows.append([i, 1, 100, 100, 50, 50, 1.0, -1, -1, -1])
        # Gap of 20 frames, then frames 31-40, same position
        for i in range(31, 41):
            rows.append([i, 1, 110, 100, 50, 50, 1.0, -1, -1, -1])
        
        result = compute_orc(rows, gap_thresholds=(15,), max_distance_px=100.0)
        # Centroid moved ~10px, within 100px threshold
        assert result["orc@15"] == 1.0
    
    def test_inconsistent_recovery(self):
        """Recovery outside distance threshold should not be counted"""
        rows = []
        # Frames 1-10 at x=100
        for i in range(1, 11):
            rows.append([i, 1, 100, 100, 50, 50, 1.0, -1, -1, -1])
        # Gap of 20 frames, then frames 31-40 at x=500 (far away)
        for i in range(31, 41):
            rows.append([i, 1, 500, 100, 50, 50, 1.0, -1, -1, -1])
        
        result = compute_orc(rows, gap_thresholds=(15,), max_distance_px=100.0)
        # Centroid moved ~400px, outside 100px threshold
        assert result["orc@15"] == 0.0


class TestComputeTCVR:
    """Tests for Team-Consistency Violation Rate computation."""
    
    def test_empty_input(self):
        result = compute_tcvr([])
        assert result["tcvr"] == 0.0
        assert result["tcvr_violations"] == 0
        assert result["tcvr_total_tracks"] == 0
    
    def test_normal_movement(self):
        """Normal movement within speed limit should not be a violation"""
        rows = []
        for i in range(1, 50):
            # Move 5 pixels per frame (within 15 px/frame limit)
            rows.append([i, 1, 100 + i * 5, 100, 50, 50, 1.0, -1, -1, -1])
        
        result = compute_tcvr(rows, max_speed_px_per_frame=15.0)
        assert result["tcvr"] == 0.0
        assert result["tcvr_violations"] == 0
    
    def test_teleportation_detected(self):
        """Teleportation (impossible speed) should be detected"""
        rows = [
            [1, 1, 100, 100, 50, 50, 1.0, -1, -1, -1],
            [2, 1, 500, 100, 50, 50, 1.0, -1, -1, -1],  # 400px in 1 frame!
        ]
        result = compute_tcvr(rows, max_speed_px_per_frame=15.0)
        assert result["tcvr"] == 1.0
        assert result["tcvr_violations"] == 1
    
    def test_mixed_trajectories(self):
        """Mix of normal and violating trajectories"""
        rows = [
            # Track 1: normal movement
            [1, 1, 100, 100, 50, 50, 1.0, -1, -1, -1],
            [2, 1, 110, 100, 50, 50, 1.0, -1, -1, -1],
            # Track 2: teleportation
            [1, 2, 100, 100, 50, 50, 1.0, -1, -1, -1],
            [2, 2, 600, 100, 50, 50, 1.0, -1, -1, -1],
        ]
        result = compute_tcvr(rows, max_speed_px_per_frame=15.0)
        assert result["tcvr"] == 0.5  # 1 of 2 trajectories violated
        assert result["tcvr_violations"] == 1


class TestComputeMSS:
    """Tests for Motion Smoothness Score computation."""
    
    def test_empty_input(self):
        result = compute_mss([])
        assert result["mss_mean"] == 0.0
        assert result["mss_median"] == 0.0
    
    def test_constant_velocity(self):
        """Constant velocity trajectory should have high MSS"""
        rows = []
        for i in range(1, 100):
            # Move exactly 10 pixels per frame
            rows.append([i, 1, 100 + i * 10, 100, 50, 50, 1.0, -1, -1, -1])
        
        result = compute_mss(rows)
        # Constant velocity means std=0, cv=0, MSS=1.0
        assert result["mss_mean"] > 0.95
    
    def test_variable_velocity(self):
        """Variable velocity trajectory should have lower MSS"""
        rows = []
        for i in range(1, 50):
            # Alternating fast/slow movement
            speed = 50 if i % 2 == 0 else 5
            rows.append([i, 1, 100 + sum([50 if j % 2 == 0 else 5 for j in range(1, i + 1)]), 100, 50, 50, 1.0, -1, -1, -1])
        
        result = compute_mss(rows)
        # Variable velocity means higher cv, lower MSS
        assert result["mss_mean"] < 0.9
    
    def test_stationary_object(self):
        """Stationary object should have MSS = 1.0"""
        rows = []
        for i in range(1, 50):
            rows.append([i, 1, 100, 100, 50, 50, 1.0, -1, -1, -1])
        
        result = compute_mss(rows)
        assert result["mss_mean"] == 1.0


class TestComputeTCI:
    """Tests for Trajectory Consistency Index computation."""
    
    def test_perfect_scores(self):
        """Perfect scores should give TCI = 1.0"""
        tci = compute_tci(isr_mean=1.0, drr=0.0, aor=0.0, pps=1.0, orc_30=1.0)
        assert tci == 1.0
    
    def test_worst_scores(self):
        """Worst scores should give TCI = 0.0"""
        tci = compute_tci(isr_mean=0.0, drr=1.0, aor=1.0, pps=0.0, orc_30=0.0)
        assert tci == 0.0
    
    def test_mixed_scores(self):
        """Mixed scores should give weighted average"""
        tci = compute_tci(
            isr_mean=0.8,
            drr=0.2,
            aor=0.3,
            pps=0.9,
            orc_30=0.6,
            weights=(0.2, 0.2, 0.2, 0.2, 0.2),
        )
        # 0.2*(0.8) + 0.2*(1-0.2) + 0.2*(1-0.3) + 0.2*(0.9) + 0.2*(0.6)
        # = 0.16 + 0.16 + 0.14 + 0.18 + 0.12 = 0.76
        assert abs(tci - 0.76) < 1e-6


class TestComputeAllStabilityMetrics:
    """Tests for the combined metrics function."""
    
    def test_empty_input(self):
        result = compute_all_stability_metrics([])
        assert "isr_mean" in result
        assert "orc@30" in result
        assert "drr" in result
        assert "aor" in result
        assert "pps" in result
        assert "mss_mean" in result
        assert "tci" in result
    
    def test_complete_output(self):
        """Ensure all expected keys are present"""
        rows = [[i, 1, 100 + i, 100, 50, 50, 1.0, -1, -1, -1] for i in range(1, 50)]
        result = compute_all_stability_metrics(rows)
        
        expected_keys = [
            "isr_mean", "isr_median", "isr_ge_0.8", "isr_ge_0.9",
            "orc@15", "orc@30", "orc@60",
            "drr", "drr_tracks_affected", "drr_total_reversals",
            "aor", "aor_median", "aor_total_outliers",
            "pps", "pps_speed_violations", "pps_accel_violations", "pps_max_speed_observed", "pps_max_accel_observed",
            "mss_mean", "mss_median",
            "tci",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_perfect_trajectory(self):
        """Perfect continuous trajectory should have excellent scores"""
        rows = [[i, 1, 100 + i * 5, 100, 50, 50, 1.0, -1, -1, -1] for i in range(1, 100)]
        result = compute_all_stability_metrics(rows)
        
        assert result["isr_mean"] == 1.0
        assert result["tci"] > 0.9


class TestIntegration:
    """Integration tests with realistic data."""
    
    def test_realistic_tracking_output(self):
        """Test with data resembling real tracking output"""
        np.random.seed(42)
        rows = []
        
        # Simulate 3 players over 100 frames
        for track_id in [1, 2, 3]:
            base_x = 100 + track_id * 200
            for frame in range(1, 101):
                # Small random movement (realistic)
                x = base_x + frame * 2 + np.random.randn() * 3
                y = 200 + np.random.randn() * 3
                rows.append([frame, track_id, x, y, 50, 100, 1.0, -1, -1, -1])
        
        result = compute_all_stability_metrics(rows)
        
        # All trajectories should be continuous
        assert result["isr_mean"] == 1.0
        # Good overall score
        assert result["tci"] > 0.8
    
    def test_fragmented_tracking_output(self):
        """Test with fragmented tracking data"""
        rows = []
        
        # Track 1: continuous
        for frame in range(1, 51):
            rows.append([frame, 1, 100 + frame, 100, 50, 50, 1.0, -1, -1, -1])
        
        # Track 2: fragmented (appears, disappears, reappears)
        for frame in range(1, 20):
            rows.append([frame, 2, 300 + frame, 100, 50, 50, 1.0, -1, -1, -1])
        for frame in range(40, 51):
            rows.append([frame, 2, 350 + frame, 100, 50, 50, 1.0, -1, -1, -1])
        
        result = compute_all_stability_metrics(rows)
        
        # ISR should be < 1 due to fragmented track
        assert result["isr_mean"] < 1.0
        # At least one track has ISR = 1.0
        assert result["isr_ge_0.9"] > 0
