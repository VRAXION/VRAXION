"""
Restructure diamond-main.json dashboard layout.
One-shot migration script — run once, then delete.

New layout:
  1. Top Bar (y=0) — unchanged
  2. Main Training (y=4) — unchanged
  3. LTCM row (y=14) — renamed from L0 Memory, + promoted LCX panels
  4. Beings & Swarm row (y=48) — new row header
  5. Ant/Bit Intelligence row (y=55) — new row header
  6. Controls row (y=98) — renamed from Data Sources, absorbs Training + Live Controls
  7. Data Tables row (y=118) — collapsed
  8. Channel Analysis / Debug row (y=119) — collapsed
"""

import json
import sys
from pathlib import Path

DASHBOARD_PATH = Path(__file__).parent.parent / "grafana" / "dashboards" / "diamond-main.json"

# === Panel ID remapping (fix duplicates) ===
ID_REMAP = {
    # Score Margin nested in row 86 has id=117, collides with top-bar Stage stat
    # Will be handled specially since it's nested
}

# === IDs to delete (vestigial panels from collapsed row 86) ===
DELETE_IDS = {75, 76, 77, 80, 81, 82, 93, 95, 96}

# === IDs to promote from row 86 to top-level ===
PROMOTE_IDS = {101, 91, 92, 102, 103, 90, 110, 116, 117}  # 117 = Score Margin (will be renumbered to 156)

# === IDs to move into collapsed Data Tables row ===
DATA_TABLE_IDS = {40, 51, 52, 48, 49, 50}

# === IDs to move into collapsed Channel Analysis row ===
# Note: the second id=51 (LCX RGB Channel Norms) will be renumbered to 157
CHANNEL_TITLE_SUBSTRINGS = [
    "LCX RGB Channel Norms",
    "Memory Scratchpad",
]
CHANNEL_ID_41 = 41  # Ant-Bit Accuracy state-timeline

# === Row IDs to delete ===
DELETE_ROW_IDS = {127, 86}

# === New gridPos assignments for ALL panels ===
# Format: id -> {x, y, w, h}
GRID_POS = {
    # Section 1: Top Bar (y=0) — NO CHANGES
    1:   {"x": 0,  "y": 0,  "w": 3,  "h": 4},   # Step
    117: {"x": 3,  "y": 0,  "w": 2,  "h": 4},   # Stage (top-bar, keep id 117)
    5:   {"x": 5,  "y": 0,  "w": 3,  "h": 4},   # Byte Match
    6:   {"x": 8,  "y": 0,  "w": 4,  "h": 4},   # Speed
    2:   {"x": 12, "y": 0,  "w": 4,  "h": 4},   # Loss
    3:   {"x": 16, "y": 0,  "w": 4,  "h": 4},   # Accuracy
    4:   {"x": 20, "y": 0,  "w": 4,  "h": 4},   # Eval Acc

    # Section 2: Main Training (y=4) — NO CHANGES
    10:  {"x": 0,  "y": 4,  "w": 14, "h": 10},  # Training Overview
    152: {"x": 14, "y": 4,  "w": 10, "h": 6},   # Training Controls
    153: {"x": 14, "y": 10, "w": 10, "h": 4},   # Phase Control

    # Section 3: LTCM row header
    126: {"x": 0,  "y": 14, "w": 24, "h": 1},   # LTCM row

    # LTCM Stats row A (y=15)
    123: {"x": 0,  "y": 15, "w": 4,  "h": 2},   # Slots %
    122: {"x": 4,  "y": 15, "w": 4,  "h": 2},   # Part. Ratio %
    119: {"x": 8,  "y": 15, "w": 4,  "h": 2},   # Entropy %
    121: {"x": 12, "y": 15, "w": 4,  "h": 2},   # Val Diversity
    120: {"x": 16, "y": 15, "w": 4,  "h": 2},   # Top-1 Mass
    124: {"x": 20, "y": 15, "w": 4,  "h": 2},   # Top-6 Mass

    # LTCM Stats row B (y=17) — promoted from row 86
    101: {"x": 0,  "y": 17, "w": 8,  "h": 3},   # LCX Utilization
    91:  {"x": 8,  "y": 17, "w": 4,  "h": 3},   # Grad Alive
    92:  {"x": 12, "y": 17, "w": 4,  "h": 3},   # Write Mode
    156: {"x": 16, "y": 17, "w": 8,  "h": 3},   # Score Margin (renumbered from 117)

    # LTCM Visualizations (y=20)
    129: {"x": 0,  "y": 20, "w": 12, "h": 12},  # Write Heat
    110: {"x": 12, "y": 20, "w": 4,  "h": 12},  # L0 Heatmap
    130: {"x": 16, "y": 20, "w": 4,  "h": 12},  # Value Norms
    149: {"x": 20, "y": 20, "w": 4,  "h": 12},  # Key Norms

    # LTCM Heat Strip + Gradient Flow (y=32)
    116: {"x": 0,  "y": 32, "w": 12, "h": 6},   # LCX Heat Strip
    90:  {"x": 12, "y": 32, "w": 12, "h": 8},   # LCX Gradient Flow

    # LTCM Slot Dynamics (y=40)
    102: {"x": 0,  "y": 40, "w": 12, "h": 8},   # Hot Slots Over Time
    103: {"x": 12, "y": 40, "w": 12, "h": 8},   # Max Heat

    # Section 4: Beings & Swarm row header (y=48)
    158: {"x": 0,  "y": 48, "w": 24, "h": 1},   # Row header (NEW)
    20:  {"x": 0,  "y": 49, "w": 3,  "h": 6},   # Beings
    21:  {"x": 3,  "y": 49, "w": 6,  "h": 6},   # Biomass
    22:  {"x": 9,  "y": 49, "w": 3,  "h": 6},   # Unique
    57:  {"x": 12, "y": 49, "w": 2,  "h": 6},   # Shared
    25:  {"x": 14, "y": 49, "w": 2,  "h": 6},   # n_bits
    23:  {"x": 16, "y": 49, "w": 4,  "h": 6},   # Coverage
    24:  {"x": 20, "y": 49, "w": 4,  "h": 6},   # Ensemble Benefit

    # Section 5: Ant/Bit Intelligence row header (y=55)
    159: {"x": 0,  "y": 55, "w": 24, "h": 1},   # Row header (NEW)
    53:  {"x": 0,  "y": 56, "w": 3,  "h": 6},   # Bit Oracle
    54:  {"x": 3,  "y": 56, "w": 3,  "h": 6},   # Clustering
    55:  {"x": 6,  "y": 56, "w": 3,  "h": 6},   # Spread
    56:  {"x": 9,  "y": 56, "w": 13, "h": 6},   # Ring Health
    58:  {"x": 22, "y": 56, "w": 2,  "h": 6},   # Ctx
    46:  {"x": 0,  "y": 62, "w": 24, "h": 10},  # Ant Accuracy
    43:  {"x": 0,  "y": 72, "w": 24, "h": 10},  # Jump Gates
    32:  {"x": 0,  "y": 82, "w": 6,  "h": 6},   # Ant IQ (avg 5)
    34:  {"x": 6,  "y": 82, "w": 6,  "h": 6},   # Ant IQ+ (avg 50)
    33:  {"x": 12, "y": 82, "w": 6,  "h": 6},   # Bit IQ (avg 5)
    35:  {"x": 18, "y": 82, "w": 6,  "h": 6},   # Bit IQ+ (avg 50)
    45:  {"x": 0,  "y": 88, "w": 24, "h": 10},  # Bit Accuracy

    # Section 6: Controls row header (y=98)
    154: {"x": 0,  "y": 98, "w": 24, "h": 1},   # Row header (renamed)
    71:  {"x": 0,  "y": 99, "w": 24, "h": 6},   # Effort Level & Think Ticks
    155: {"x": 0,  "y": 105, "w": 10, "h": 10},  # Data Mix
    60:  {"x": 10, "y": 105, "w": 7,  "h": 10},  # Live Controls
    61:  {"x": 17, "y": 105, "w": 7,  "h": 10},  # Control Change Log
    70:  {"x": 0,  "y": 115, "w": 6,  "h": 3},   # Effort Level stat

    # Section 7: Data Tables row header (y=118) — COLLAPSED
    160: {"x": 0,  "y": 118, "w": 24, "h": 1},  # Row header (NEW)

    # Section 8: Channel Analysis row header (y=119) — COLLAPSED
    161: {"x": 0,  "y": 119, "w": 24, "h": 1},  # Row header (NEW)
}

# Nested panel positions inside collapsed rows
DATA_TABLE_GRID = {
    40: {"x": 0,  "y": 119, "w": 8,  "h": 14},  # Input
    51: {"x": 8,  "y": 119, "w": 8,  "h": 14},  # Bit Masks
    52: {"x": 16, "y": 119, "w": 8,  "h": 14},  # Output
    48: {"x": 0,  "y": 133, "w": 8,  "h": 10},  # Memory State
    49: {"x": 8,  "y": 133, "w": 8,  "h": 10},  # LCX Drift
    50: {"x": 16, "y": 133, "w": 8,  "h": 10},  # LCX Delta
}

CHANNEL_GRID = {
    157: {"x": 0,  "y": 120, "w": 24, "h": 8},   # LCX RGB Channel Norms (renumbered)
    47:  {"x": 0,  "y": 128, "w": 24, "h": 12},  # Memory Scratchpad
    41:  {"x": 0,  "y": 140, "w": 24, "h": 12},  # Ant-Bit Accuracy
}


def main():
    print(f"Reading {DASHBOARD_PATH}")
    with open(DASHBOARD_PATH, 'r', encoding='utf-8') as f:
        dashboard = json.load(f)

    old_panels = dashboard['panels']
    print(f"  Found {len(old_panels)} top-level panels")

    # --- Step 0: Index all panels ---
    # Separate row 86's nested panels
    row86 = None
    row86_nested = []
    top_panels = []

    for p in old_panels:
        if p.get('type') == 'row' and p.get('id') == 86:
            row86 = p
            row86_nested = p.get('panels', [])
            print(f"  Row 86 'LCX Memory' has {len(row86_nested)} nested panels")
        else:
            top_panels.append(p)

    # --- Step 1: Extract keepers from row 86 ---
    promoted = []
    for p in row86_nested:
        pid = p['id']
        if pid in DELETE_IDS:
            print(f"  DELETE nested panel {pid}: {p.get('title', '?')}")
            continue
        if pid in PROMOTE_IDS:
            # Renumber Score Margin from 117 to 156
            if pid == 117 and 'Score Margin' in p.get('title', ''):
                print(f"  PROMOTE+RENUMBER nested panel 117 -> 156: {p['title']}")
                p['id'] = 156
            else:
                print(f"  PROMOTE nested panel {pid}: {p.get('title', '?')}")
            promoted.append(p)
            continue
        # Unexpected panel in row 86
        print(f"  WARNING: unexpected panel {pid} in row 86: {p.get('title', '?')}")

    # --- Step 1b: Create Score Margin panel if not found in row 86 ---
    # (It was added in a previous session but may not be in the git-committed JSON)
    has_score_margin = any(p['id'] == 156 for p in promoted)
    if not has_score_margin:
        print("  CREATE Score Margin panel (id=156) — not found in row 86")
        score_margin_panel = {
            "id": 156,
            "title": "Score Margin (routing quality)",
            "description": "Top-K routing decisiveness. Margin = cosine gap between last winner (slot K) and first loser (slot K+1). High margin = meaningful routing, near-zero = arbitrary selection.",
            "type": "timeseries",
            "gridPos": {"x": 16, "y": 17, "w": 8, "h": 3},
            "datasource": {"type": "influxdb", "uid": "efd1xv51o3vuob"},
            "targets": [
                {
                    "refId": "A",
                    "query": "from(bucket: \"diamond\")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n  |> filter(fn: (r) => r._measurement == \"lcx_levels\")\n  |> filter(fn: (r) => r.run_id == \"${run_id}\")\n  |> filter(fn: (r) => r._field == \"L0_score_margin\")\n  |> group(columns: [\"_field\"])\n  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)"
                },
                {
                    "refId": "B",
                    "query": "from(bucket: \"diamond\")\n  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)\n  |> filter(fn: (r) => r._measurement == \"lcx_levels\")\n  |> filter(fn: (r) => r.run_id == \"${run_id}\")\n  |> filter(fn: (r) => r._field == \"L0_score_top1\")\n  |> group(columns: [\"_field\"])\n  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "palette-classic"},
                    "custom": {
                        "lineWidth": 2,
                        "fillOpacity": 10,
                        "spanNulls": False,
                        "axisSoftMin": 0,
                        "axisSoftMax": 0.5
                    }
                },
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "L0_score_margin"},
                        "properties": [
                            {"id": "displayName", "value": "Score Margin (sK - sK+1)"},
                            {"id": "color", "value": {"fixedColor": "green", "mode": "fixed"}}
                        ]
                    },
                    {
                        "matcher": {"id": "byName", "options": "L0_score_top1"},
                        "properties": [
                            {"id": "displayName", "value": "Best Match Score (s1)"},
                            {"id": "color", "value": {"fixedColor": "#28d3e2", "mode": "fixed"}}
                        ]
                    }
                ]
            },
            "options": {"tooltip": {"mode": "multi"}, "legend": {"displayMode": "list", "placement": "bottom"}}
        }
        promoted.append(score_margin_panel)

    # --- Step 2: Handle duplicate id=51 and channel panels ---
    # Find and rename LCX RGB Channel Norms (second id=51) to 157
    # Also identify panels to move to collapsed rows
    final_top = []
    data_table_panels = []
    channel_panels = []

    for p in top_panels:
        pid = p['id']
        title = p.get('title', '')

        # Delete row 127 (Training) and row 86 (already extracted above)
        if p.get('type') == 'row' and pid in DELETE_ROW_IDS:
            print(f"  DELETE row {pid}: {title}")
            continue

        # Rename duplicate id=51 LCX RGB Channel Norms -> 157
        if pid == 51 and 'RGB' in title:
            print(f"  RENUMBER panel 51 -> 157: {title}")
            p['id'] = 157
            pid = 157

        # Move data table panels to collapsed row
        if pid in DATA_TABLE_IDS:
            print(f"  MOVE to Data Tables row: {pid} {title}")
            data_table_panels.append(p)
            continue

        # Move channel analysis panels to collapsed row
        if any(sub in title for sub in CHANNEL_TITLE_SUBSTRINGS) or pid == CHANNEL_ID_41:
            print(f"  MOVE to Channel Analysis row: {pid} {title}")
            channel_panels.append(p)
            continue

        # Move Effort Level stat (id=70) — stays in top-level under Controls
        final_top.append(p)

    # Add promoted panels from row 86
    final_top.extend(promoted)

    # --- Step 3: Create new row panels ---
    new_rows = [
        {
            "id": 158,
            "title": "Beings & Swarm",
            "type": "row",
            "collapsed": False,
            "gridPos": {"x": 0, "y": 48, "w": 24, "h": 1},
            "panels": []
        },
        {
            "id": 159,
            "title": "Ant/Bit Intelligence",
            "type": "row",
            "collapsed": False,
            "gridPos": {"x": 0, "y": 55, "w": 24, "h": 1},
            "panels": []
        },
        {
            "id": 160,
            "title": "Data Tables",
            "type": "row",
            "collapsed": True,
            "gridPos": {"x": 0, "y": 118, "w": 24, "h": 1},
            "panels": []
        },
        {
            "id": 161,
            "title": "Channel Analysis / Debug",
            "type": "row",
            "collapsed": True,
            "gridPos": {"x": 0, "y": 119, "w": 24, "h": 1},
            "panels": []
        },
    ]
    final_top.extend(new_rows)

    # --- Step 4: Rename existing rows ---
    for p in final_top:
        if p.get('type') == 'row':
            if p['id'] == 126:
                p['title'] = 'LTCM (Long Term Consensus Matrix)'
                print(f"  RENAME row 126: 'L0 Memory' -> '{p['title']}'")
            elif p['id'] == 154:
                p['title'] = 'Controls'
                print(f"  RENAME row 154: 'Data Sources' -> '{p['title']}'")

    # --- Step 5: Update gridPos for all top-level panels ---
    for p in final_top:
        pid = p['id']
        if pid in GRID_POS:
            p['gridPos'] = GRID_POS[pid].copy()
        else:
            print(f"  WARNING: no gridPos for top-level panel {pid}: {p.get('title', '?')}")

    # --- Step 6: Populate collapsed rows ---
    # Data Tables (row 160)
    for p in data_table_panels:
        pid = p['id']
        if pid in DATA_TABLE_GRID:
            p['gridPos'] = DATA_TABLE_GRID[pid].copy()
        else:
            print(f"  WARNING: no gridPos for data table panel {pid}")

    # Channel Analysis (row 161)
    for p in channel_panels:
        pid = p['id']
        if pid in CHANNEL_GRID:
            p['gridPos'] = CHANNEL_GRID[pid].copy()
        else:
            print(f"  WARNING: no gridPos for channel panel {pid}")

    # Find the collapsed row panels and attach nested panels
    for p in final_top:
        if p['id'] == 160:
            p['panels'] = sorted(data_table_panels, key=lambda x: (x['gridPos']['y'], x['gridPos']['x']))
        elif p['id'] == 161:
            p['panels'] = sorted(channel_panels, key=lambda x: (x['gridPos']['y'], x['gridPos']['x']))

    # --- Step 7: Sort top-level panels by gridPos ---
    def sort_key(p):
        gp = p.get('gridPos', {})
        return (gp.get('y', 999), gp.get('x', 999))

    final_top.sort(key=sort_key)

    # --- Step 8: Validate ---
    # Check for duplicate IDs
    all_ids = set()
    dupes = []
    for p in final_top:
        if p['id'] in all_ids:
            dupes.append(p['id'])
        all_ids.add(p['id'])
        # Check nested panels in collapsed rows
        for np in p.get('panels', []):
            if np['id'] in all_ids:
                dupes.append(np['id'])
            all_ids.add(np['id'])

    if dupes:
        print(f"\n  ERROR: Duplicate IDs found: {dupes}")
        sys.exit(1)

    # Count panels
    n_top = len([p for p in final_top if p.get('type') != 'row'])
    n_rows = len([p for p in final_top if p.get('type') == 'row'])
    n_nested = sum(len(p.get('panels', [])) for p in final_top)
    n_total = n_top + n_rows + n_nested
    print(f"\n  Panel count: {n_top} top-level + {n_rows} rows + {n_nested} nested = {n_total} total")

    # --- Step 9: Write output ---
    dashboard['panels'] = final_top

    # Write to same file
    with open(DASHBOARD_PATH, 'w', encoding='utf-8') as f:
        json.dump(dashboard, f, indent=2, ensure_ascii=False)

    print(f"\n  Written to {DASHBOARD_PATH}")
    print("  Done! Restart Grafana to pick up changes.")


if __name__ == '__main__':
    main()
