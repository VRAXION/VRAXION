/* D9.0d Progressive Planet — mock state.js
   Demo fixture so the renderer can be opened without running the sampler.
   To preview: copy this file to state.js (or rename) in the same directory
   as progressive_planet.html.
   Schema: d9.0d-1
*/
window.ATLAS_STATE = {
  "schema_version": "d9.0d-1",
  "run_id": "d9_0d_mock_run",
  "generated_at_utc": "2026-04-28T15:30:00Z",
  "last_updated": "2026-04-28T15:30:00Z",
  "source_samples_csv": "(mock — no real samples)",
  "source_run_id": "mock",
  "phase_status": "running",
  "stop_clock_active": false,

  "config": {
    "H": 256,
    "resolution": "16x32",
    "lat_bins": 16,
    "lon_bins": 32,
    "scout_eval_len": 100,
    "confirmed_eval_len": 1000,
    "scout_target_per_tile": 2,
    "confirmed_target_per_tile": 10,
    "checkpoint": "(mock checkpoint)"
  },

  "tiles": [
    /* one tile per state for visual smoke-test of the renderer */
    {
      "tile_id": "8_4", "lat_bin": 8, "lon_bin": 4,
      "lat_center": 0.0, "lon_center": -2.16,
      "x": -0.55, "y": -0.83, "z": 0.0,
      "n_scout": 0, "n_confirmed": 0, "target_n": 10,
      "mean_delta": null, "best_delta": null, "median_delta": null, "std_delta": null,
      "cliff_rate": null, "positive_rate": null, "mean_behavior_distance": null, "confidence": 0.0,
      "state": "UNKNOWN", "recommended_action": "scout",
      "dominant_mutation_type": null, "dominant_radius": null,
      "per_type": {
        "edge":      { "n": 0, "mean_delta": null, "best_delta": null, "std_delta": null, "cliff_rate": null, "positive_rate": null },
        "threshold": { "n": 0, "mean_delta": null, "best_delta": null, "std_delta": null, "cliff_rate": null, "positive_rate": null },
        "channel":   { "n": 0, "mean_delta": null, "best_delta": null, "std_delta": null, "cliff_rate": null, "positive_rate": null },
        "polarity":  { "n": 0, "mean_delta": null, "best_delta": null, "std_delta": null, "cliff_rate": null, "positive_rate": null }
      },
      "scan_priority": 0.5, "split_priority": -1.0
    },
    {
      "tile_id": "8_8", "lat_bin": 8, "lon_bin": 8,
      "lat_center": 0.0, "lon_center": -1.18,
      "x": 0.38, "y": -0.92, "z": 0.0,
      "n_scout": 2, "n_confirmed": 0, "target_n": 10,
      "mean_delta": -0.001, "best_delta": 0.005, "median_delta": -0.001, "std_delta": 0.004,
      "cliff_rate": 0.10, "positive_rate": 0.20, "mean_behavior_distance": 0.08, "confidence": 0.25,
      "state": "SCOUT", "recommended_action": "sample_more",
      "dominant_mutation_type": "edge", "dominant_radius": 1,
      "per_type": {
        "edge":      { "n": 2, "mean_delta": -0.001, "best_delta": 0.005, "std_delta": 0.004, "cliff_rate": 0.10, "positive_rate": 0.20 },
        "threshold": { "n": 0, "mean_delta": null, "best_delta": null, "std_delta": null, "cliff_rate": null, "positive_rate": null },
        "channel":   { "n": 0, "mean_delta": null, "best_delta": null, "std_delta": null, "cliff_rate": null, "positive_rate": null },
        "polarity":  { "n": 0, "mean_delta": null, "best_delta": null, "std_delta": null, "cliff_rate": null, "positive_rate": null }
      },
      "scan_priority": 1.2, "split_priority": -0.4
    },
    {
      "tile_id": "8_12", "lat_bin": 8, "lon_bin": 12,
      "lat_center": 0.0, "lon_center": -0.20,
      "x": 0.98, "y": -0.20, "z": 0.0,
      "n_scout": 4, "n_confirmed": 4, "target_n": 10,
      "mean_delta": 0.018, "best_delta": 0.045, "median_delta": 0.015, "std_delta": 0.012,
      "cliff_rate": 0.05, "positive_rate": 0.62, "mean_behavior_distance": 0.11, "confidence": 0.85,
      "state": "PROMISING", "recommended_action": "confirm_expensive",
      "dominant_mutation_type": "edge", "dominant_radius": 4,
      "per_type": {
        "edge":      { "n": 4, "mean_delta": 0.022, "best_delta": 0.045, "std_delta": 0.010, "cliff_rate": 0.00, "positive_rate": 0.75 },
        "threshold": { "n": 3, "mean_delta": 0.012, "best_delta": 0.030, "std_delta": 0.013, "cliff_rate": 0.10, "positive_rate": 0.55 },
        "channel":   { "n": 1, "mean_delta": 0.005, "best_delta": 0.005, "std_delta": 0.000, "cliff_rate": 0.00, "positive_rate": 0.50 },
        "polarity":  { "n": 0, "mean_delta": null, "best_delta": null, "std_delta": null, "cliff_rate": null, "positive_rate": null }
      },
      "scan_priority": 2.4, "split_priority": 0.3
    },
    {
      "tile_id": "8_16", "lat_bin": 8, "lon_bin": 16,
      "lat_center": 0.0, "lon_center": 0.78,
      "x": 0.71, "y": 0.71, "z": 0.0,
      "n_scout": 6, "n_confirmed": 8, "target_n": 10,
      "mean_delta": 0.030, "best_delta": 0.062, "median_delta": 0.028, "std_delta": 0.011,
      "cliff_rate": 0.02, "positive_rate": 0.78, "mean_behavior_distance": 0.13, "confidence": 1.00,
      "state": "CONFIRMED_GOOD", "recommended_action": "branch_candidate",
      "dominant_mutation_type": "edge", "dominant_radius": 4,
      "per_type": {
        "edge":      { "n": 8, "mean_delta": 0.034, "best_delta": 0.062, "std_delta": 0.010, "cliff_rate": 0.00, "positive_rate": 0.85 },
        "threshold": { "n": 4, "mean_delta": 0.025, "best_delta": 0.045, "std_delta": 0.011, "cliff_rate": 0.05, "positive_rate": 0.70 },
        "channel":   { "n": 2, "mean_delta": 0.010, "best_delta": 0.020, "std_delta": 0.010, "cliff_rate": 0.00, "positive_rate": 0.50 },
        "polarity":  { "n": 0, "mean_delta": null, "best_delta": null, "std_delta": null, "cliff_rate": null, "positive_rate": null }
      },
      "scan_priority": 1.8, "split_priority": 0.5
    },
    {
      "tile_id": "8_20", "lat_bin": 8, "lon_bin": 20,
      "lat_center": 0.0, "lon_center": 1.77,
      "x": -0.20, "y": 0.98, "z": 0.0,
      "n_scout": 5, "n_confirmed": 0, "target_n": 10,
      "mean_delta": 0.002, "best_delta": 0.040, "median_delta": -0.005, "std_delta": 0.028,
      "cliff_rate": 0.30, "positive_rate": 0.40, "mean_behavior_distance": 0.14, "confidence": 0.62,
      "state": "NOISY", "recommended_action": "split",
      "dominant_mutation_type": "threshold", "dominant_radius": 4,
      "per_type": {
        "edge":      { "n": 1, "mean_delta": 0.025, "best_delta": 0.040, "std_delta": 0.000, "cliff_rate": 0.00, "positive_rate": 1.00 },
        "threshold": { "n": 3, "mean_delta": 0.000, "best_delta": 0.020, "std_delta": 0.025, "cliff_rate": 0.33, "positive_rate": 0.33 },
        "channel":   { "n": 1, "mean_delta": -0.020, "best_delta": -0.020, "std_delta": 0.000, "cliff_rate": 1.00, "positive_rate": 0.00 },
        "polarity":  { "n": 0, "mean_delta": null, "best_delta": null, "std_delta": null, "cliff_rate": null, "positive_rate": null }
      },
      "scan_priority": 1.4, "split_priority": 1.9
    },
    {
      "tile_id": "8_24", "lat_bin": 8, "lon_bin": 24,
      "lat_center": 0.0, "lon_center": 2.75,
      "x": -0.92, "y": 0.38, "z": 0.0,
      "n_scout": 7, "n_confirmed": 4, "target_n": 10,
      "mean_delta": -0.008, "best_delta": 0.012, "median_delta": -0.010, "std_delta": 0.038,
      "cliff_rate": 0.55, "positive_rate": 0.18, "mean_behavior_distance": 0.16, "confidence": 0.95,
      "state": "SPLIT_CANDIDATE", "recommended_action": "split",
      "dominant_mutation_type": "threshold", "dominant_radius": 4,
      "per_type": {
        "edge":      { "n": 3, "mean_delta": 0.005, "best_delta": 0.012, "std_delta": 0.005, "cliff_rate": 0.00, "positive_rate": 0.66 },
        "threshold": { "n": 4, "mean_delta": -0.015, "best_delta": 0.005, "std_delta": 0.040, "cliff_rate": 0.75, "positive_rate": 0.25 },
        "channel":   { "n": 2, "mean_delta": -0.022, "best_delta": -0.010, "std_delta": 0.010, "cliff_rate": 1.00, "positive_rate": 0.00 },
        "polarity":  { "n": 2, "mean_delta": -0.030, "best_delta": -0.020, "std_delta": 0.010, "cliff_rate": 1.00, "positive_rate": 0.00 }
      },
      "scan_priority": 0.8, "split_priority": 2.6
    },
    {
      "tile_id": "8_28", "lat_bin": 8, "lon_bin": 28,
      "lat_center": 0.0, "lon_center": 3.73,
      "x": -0.83, "y": -0.55, "z": 0.0,
      "n_scout": 6, "n_confirmed": 2, "target_n": 10,
      "mean_delta": -0.025, "best_delta": -0.005, "median_delta": -0.025, "std_delta": 0.015,
      "cliff_rate": 0.85, "positive_rate": 0.05, "mean_behavior_distance": 0.18, "confidence": 0.85,
      "state": "CLIFFY", "recommended_action": "avoid/cliff",
      "dominant_mutation_type": "polarity", "dominant_radius": 1,
      "per_type": {
        "edge":      { "n": 1, "mean_delta": -0.005, "best_delta": -0.005, "std_delta": 0.000, "cliff_rate": 1.00, "positive_rate": 0.00 },
        "threshold": { "n": 1, "mean_delta": -0.020, "best_delta": -0.020, "std_delta": 0.000, "cliff_rate": 1.00, "positive_rate": 0.00 },
        "channel":   { "n": 2, "mean_delta": -0.025, "best_delta": -0.010, "std_delta": 0.015, "cliff_rate": 1.00, "positive_rate": 0.00 },
        "polarity":  { "n": 4, "mean_delta": -0.034, "best_delta": -0.020, "std_delta": 0.010, "cliff_rate": 1.00, "positive_rate": 0.00 }
      },
      "scan_priority": -1.6, "split_priority": -0.3
    },
    {
      "tile_id": "4_8", "lat_bin": 4, "lon_bin": 8,
      "lat_center": -0.79, "lon_center": -1.18,
      "x": 0.27, "y": -0.65, "z": -0.71,
      "n_scout": 4, "n_confirmed": 0, "target_n": 10,
      "mean_delta": -0.012, "best_delta": -0.005, "median_delta": -0.012, "std_delta": 0.004,
      "cliff_rate": 0.20, "positive_rate": 0.05, "mean_behavior_distance": 0.07, "confidence": 0.50,
      "state": "DESERT", "recommended_action": "retire",
      "dominant_mutation_type": "channel", "dominant_radius": 4,
      "per_type": {
        "edge":      { "n": 1, "mean_delta": -0.010, "best_delta": -0.005, "std_delta": 0.000, "cliff_rate": 0.00, "positive_rate": 0.00 },
        "threshold": { "n": 1, "mean_delta": -0.012, "best_delta": -0.010, "std_delta": 0.000, "cliff_rate": 0.00, "positive_rate": 0.00 },
        "channel":   { "n": 2, "mean_delta": -0.014, "best_delta": -0.010, "std_delta": 0.004, "cliff_rate": 0.50, "positive_rate": 0.00 },
        "polarity":  { "n": 0, "mean_delta": null, "best_delta": null, "std_delta": null, "cliff_rate": null, "positive_rate": null }
      },
      "scan_priority": -0.9, "split_priority": -1.4
    },
    {
      "tile_id": "12_16", "lat_bin": 12, "lon_bin": 16,
      "lat_center": 0.79, "lon_center": 0.78,
      "x": 0.50, "y": 0.50, "z": 0.71,
      "n_scout": 8, "n_confirmed": 6, "target_n": 10,
      "mean_delta": -0.040, "best_delta": -0.025, "median_delta": -0.040, "std_delta": 0.008,
      "cliff_rate": 0.95, "positive_rate": 0.00, "mean_behavior_distance": 0.20, "confidence": 1.00,
      "state": "RETIRED", "recommended_action": "retire",
      "dominant_mutation_type": "polarity", "dominant_radius": 1,
      "per_type": {
        "edge":      { "n": 2, "mean_delta": -0.025, "best_delta": -0.025, "std_delta": 0.000, "cliff_rate": 1.00, "positive_rate": 0.00 },
        "threshold": { "n": 2, "mean_delta": -0.038, "best_delta": -0.030, "std_delta": 0.008, "cliff_rate": 1.00, "positive_rate": 0.00 },
        "channel":   { "n": 4, "mean_delta": -0.045, "best_delta": -0.030, "std_delta": 0.010, "cliff_rate": 1.00, "positive_rate": 0.00 },
        "polarity":  { "n": 6, "mean_delta": -0.044, "best_delta": -0.025, "std_delta": 0.010, "cliff_rate": 1.00, "positive_rate": 0.00 }
      },
      "scan_priority": -2.3, "split_priority": -1.6
    }
  ],

  "queue": [
    { "tile_id": "8_12", "type": "edge",      "priority": 2.40, "next_action": "confirm_expensive" },
    { "tile_id": "8_24", "type": "threshold", "priority": 1.95, "next_action": "split" },
    { "tile_id": "8_16", "type": "edge",      "priority": 1.80, "next_action": "branch_candidate" },
    { "tile_id": "8_20", "type": "threshold", "priority": 1.40, "next_action": "split" },
    { "tile_id": "8_8",  "type": "edge",      "priority": 1.20, "next_action": "scout" },
    { "tile_id": "8_4",  "type": "all",       "priority": 0.50, "next_action": "scout" },
    { "tile_id": "8_24", "type": "polarity",  "priority": -0.10, "next_action": "observe" }
  ],

  "progress": {
    "total_tiles_possible": 512,
    "tiles_with_data": 9,
    "coverage": 0.018,
    "confident_fraction": 0.011,
    "promising_count": 1,
    "retired_count": 1,
    "split_candidate_count": 1,
    "samples_total": 50,
    "samples_scout": 36,
    "samples_confirmed": 14
  },

  "acquisition_weights": {
    "best": 0.35,
    "uncertainty": 0.25,
    "std": 0.20,
    "positive_rate": 0.10,
    "cliff_rate": -0.20,
    "confidence": -0.10
  }
};
