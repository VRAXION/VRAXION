#!/usr/bin/env python3
"""Generate a self-contained Operator rank dashboard from rank artifacts."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


DEFAULT_E109 = Path("target/pilot_wave/e109_operator_rank_ladder_and_golden_watch_probation_mode")
DEFAULT_E110 = Path("target/pilot_wave/e110_promote_or_drop_operator_grind_wave1")
DEFAULT_E111 = Path("target/pilot_wave/e111_bronze_mutation_prune_promote_or_drop_wave")
DEFAULT_E112 = Path("target/pilot_wave/e112_gold_to_core_prune_heavy_probation_wave")
DEFAULT_E114 = Path("target/pilot_wave/e114_fineweb_next_limit_stability_projection")
SAMPLE_E109 = Path("docs/research/artifact_samples/e109_operator_rank_ladder_and_golden_watch_probation_mode")
SAMPLE_E110 = Path("docs/research/artifact_samples/e110_promote_or_drop_operator_grind_wave1")
SAMPLE_E111 = Path("docs/research/artifact_samples/e111_bronze_mutation_prune_promote_or_drop_wave")
SAMPLE_E112 = Path("docs/research/artifact_samples/e112_gold_to_core_prune_heavy_probation_wave")
DEFAULT_OUT = Path("target/pilot_wave/operator_rank_dashboard/index.html")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def existing_artifact_path(requested: Path, fallback: Path, required_file: str) -> Path:
    if (requested / required_file).exists():
        return requested
    if (fallback / required_file).exists():
        return fallback
    raise FileNotFoundError(f"missing artifact {required_file!r} in {requested} or {fallback}")


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = [
        "operator_id",
        "display_name",
        "scope",
        "family",
        "group_id",
        "e107_status",
        "e108_status",
        "rank",
        "watch_state",
        "qualified_activation",
        "positive",
        "neutral_valid",
        "neutral_waste",
        "neutral_waste_rate",
        "hard_negative",
        "rule_of_three_upper_failure_bound",
        "e107_family_coverage",
        "e108_family_coverage",
        "combined_family_coverage",
        "campaign_count",
        "counterfactual_value",
        "activated_gain",
        "ablation_loss",
        "reload_shadow_pass",
        "challenger_pass",
        "prune_pass",
        "e110_wave1_outcome",
        "qualified_activation_add",
        "rank_before",
        "rank_after",
        "e111_wave2_outcome",
        "selected_variant_id",
        "selected_variant_type",
        "selected_variant_net_score",
        "mutation_attempts",
        "accepted_mutations",
        "rejected_mutations",
        "rollback_count",
        "e112_wave3_outcome",
        "selected_prune_ratio",
        "long_horizon_no_harm_pass",
        "negative_scope_pass",
        "e114_current_run_calls",
        "e114_projected_full_fineweb_calls",
        "e114_projected_activation_after_full_fineweb",
        "e114_projected_reaches_permacore_probation",
        "e114_projected_remaining_after_full_fineweb",
        "e114_selected_variant",
    ]
    return [{key: row.get(key) for key in keep} for row in rows]


def rank_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = ["Bronze", "Silver", "Gold", "DiamondCandidate", "CoreMemoryCandidate", "RedFlag", "Deprecated"]
    return {key: sum(1 for row in rows if row.get("rank") == key) for key in keys}


def merge_e110(rows: list[dict[str, Any]], e110: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e110 or not (e110 / "wave_results.json").exists():
        return rows, None
    wave = read_json(e110 / "wave_results.json")["rows"]
    wave_by_id = {row["operator_id"]: row for row in wave}
    merged: list[dict[str, Any]] = []
    for row in rows:
        update = wave_by_id.get(row["operator_id"])
        if not update:
            merged.append(row)
            continue
        next_row = dict(row)
        for key in [
            "rank_after",
            "rank_before",
            "wave1_outcome",
            "qualified_activation",
            "qualified_activation_add",
            "positive",
            "neutral_valid",
            "neutral_waste",
            "neutral_waste_rate",
            "hard_negative",
            "rule_of_three_upper_failure_bound",
            "combined_family_coverage",
            "campaign_count",
            "counterfactual_value",
            "activated_gain",
            "ablation_loss",
            "reload_shadow_pass",
            "challenger_pass",
            "prune_pass",
        ]:
            if key in update:
                next_row[key if key != "wave1_outcome" else "e110_wave1_outcome"] = update[key]
        next_row["rank"] = update.get("rank_after", next_row["rank"])
        next_row["watch_state"] = "E110GoldConfirmed" if next_row["rank"] == "Gold" else next_row.get("watch_state")
        merged.append(next_row)
    return merged, {
        "summary": read_json(e110 / "summary.json"),
        "aggregate": read_json(e110 / "aggregate_metrics.json"),
        "promotion": read_json(e110 / "promotion_report.json"),
    }


def merge_e111(rows: list[dict[str, Any]], e111: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e111 or not (e111 / "wave_results.json").exists():
        return rows, None
    wave = read_json(e111 / "wave_results.json")["rows"]
    wave_by_id = {row["operator_id"]: row for row in wave}
    merged: list[dict[str, Any]] = []
    for row in rows:
        update = wave_by_id.get(row["operator_id"])
        if not update:
            merged.append(row)
            continue
        next_row = dict(row)
        for key in [
            "rank_after",
            "rank_before",
            "wave2_outcome",
            "selected_variant_id",
            "selected_variant_type",
            "selected_variant_net_score",
            "qualified_activation",
            "qualified_activation_add",
            "positive",
            "neutral_valid",
            "neutral_waste",
            "neutral_waste_rate",
            "hard_negative",
            "rule_of_three_upper_failure_bound",
            "combined_family_coverage",
            "campaign_count",
            "counterfactual_value",
            "activated_gain",
            "ablation_loss",
            "reload_shadow_pass",
            "challenger_pass",
            "prune_pass",
            "mutation_attempts",
            "accepted_mutations",
            "rejected_mutations",
            "rollback_count",
        ]:
            if key in update:
                next_row[key if key != "wave2_outcome" else "e111_wave2_outcome"] = update[key]
        next_row["rank"] = update.get("rank_after", next_row["rank"])
        next_row["watch_state"] = "E111MutatedGoldConfirmed" if next_row["rank"] == "Gold" else next_row.get("watch_state")
        merged.append(next_row)
    return merged, {
        "summary": read_json(e111 / "summary.json"),
        "aggregate": read_json(e111 / "aggregate_metrics.json"),
        "promotion": read_json(e111 / "promotion_report.json"),
        "mutation": read_json(e111 / "mutation_summary.json"),
        "duration": read_json(e111 / "duration_report.json"),
    }


def merge_e112(rows: list[dict[str, Any]], e112: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e112 or not (e112 / "wave_results.json").exists():
        return rows, None
    wave = read_json(e112 / "wave_results.json")["rows"]
    wave_by_id = {row["operator_id"]: row for row in wave}
    merged: list[dict[str, Any]] = []
    for row in rows:
        update = wave_by_id.get(row["operator_id"])
        if not update:
            merged.append(row)
            continue
        next_row = dict(row)
        for key in [
            "rank_after",
            "rank_before",
            "wave3_outcome",
            "selected_variant_id",
            "selected_variant_type",
            "selected_variant_net_score",
            "selected_prune_ratio",
            "qualified_activation",
            "qualified_activation_add",
            "positive",
            "neutral_valid",
            "neutral_waste",
            "neutral_waste_rate",
            "hard_negative",
            "rule_of_three_upper_failure_bound",
            "combined_family_coverage",
            "campaign_count",
            "counterfactual_value",
            "activated_gain",
            "ablation_loss",
            "reload_shadow_pass",
            "challenger_pass",
            "prune_pass",
            "long_horizon_no_harm_pass",
            "negative_scope_pass",
            "mutation_attempts",
            "accepted_mutations",
            "rejected_mutations",
            "rollback_count",
        ]:
            if key in update:
                next_row[key if key != "wave3_outcome" else "e112_wave3_outcome"] = update[key]
        next_row["rank"] = update.get("rank_after", next_row["rank"])
        next_row["watch_state"] = "E112CoreCandidateConfirmed" if next_row["rank"] == "CoreMemoryCandidate" else next_row.get("watch_state")
        merged.append(next_row)
    return merged, {
        "summary": read_json(e112 / "summary.json"),
        "aggregate": read_json(e112 / "aggregate_metrics.json"),
        "promotion": read_json(e112 / "promotion_report.json"),
        "mutation": read_json(e112 / "mutation_summary.json"),
        "duration": read_json(e112 / "duration_report.json"),
    }


def merge_e114(rows: list[dict[str, Any]], e114: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e114 or not (e114 / "operator_projection_report.json").exists():
        return rows, None
    projection = read_json(e114 / "operator_projection_report.json")["rows"]
    projection_by_id = {row["operator_id"]: row for row in projection}
    merged: list[dict[str, Any]] = []
    for row in rows:
        update = projection_by_id.get(row["operator_id"])
        if not update:
            merged.append(row)
            continue
        next_row = dict(row)
        next_row["e114_current_run_calls"] = update.get("current_run_calls")
        next_row["e114_projected_full_fineweb_calls"] = update.get("projected_full_fineweb_calls")
        next_row["e114_projected_activation_after_full_fineweb"] = update.get("projected_activation_after_full_fineweb")
        next_row["e114_projected_reaches_permacore_probation"] = update.get("projected_reaches_permacore_probation")
        next_row["e114_projected_remaining_after_full_fineweb"] = update.get("projected_remaining_after_full_fineweb")
        next_row["e114_selected_variant"] = update.get("selected_variant")
        merged.append(next_row)
    return merged, {
        "summary": read_json(e114 / "summary.json"),
        "aggregate": read_json(e114 / "aggregate_metrics.json"),
        "target": read_json(e114 / "target_sufficiency_report.json"),
        "stability": read_json(e114 / "stability_trend_report.json"),
    }


def build_payload(
    e109: Path,
    e110: Path | None = None,
    e111: Path | None = None,
    e112: Path | None = None,
    e114: Path | None = None,
) -> dict[str, Any]:
    rank_results = read_json(e109 / "rank_results.json")
    rows, e110_payload = merge_e110(compact_rows(rank_results["rows"]), e110)
    rows, e111_payload = merge_e111(rows, e111)
    rows, e112_payload = merge_e112(rows, e112)
    rows, e114_payload = merge_e114(rows, e114)
    counts = rank_counts(rows)
    aggregate = read_json(e109 / "aggregate_metrics.json")
    aggregate = {
        **aggregate,
        "bronze_count": counts["Bronze"],
        "silver_count": counts["Silver"],
        "gold_count": counts["Gold"],
        "diamond_candidate_count": counts["DiamondCandidate"],
        "core_memory_candidate_count": counts["CoreMemoryCandidate"],
        "red_flag_count": counts["RedFlag"],
        "deprecated_count": counts["Deprecated"],
        "qualified_activation_total": sum(int(row.get("qualified_activation") or 0) for row in rows),
        "e114_projected_reach_permacore_count": e114_payload["aggregate"]["projected_reach_permacore_count"] if e114_payload else None,
        "e114_projected_need_targeted_data_count": e114_payload["aggregate"]["projected_need_targeted_data_count"] if e114_payload else None,
        "e114_stability_trend": e114_payload["aggregate"]["stability_trend"] if e114_payload else None,
    }
    summary = read_json(e109 / "summary.json")
    summary = {
        **summary,
        "rank_counts": counts,
        "latest_wave": "E114 FineWeb projection" if e114_payload else "E112 Wave 3" if e112_payload else "E111 Wave 2" if e111_payload else "E110 Wave 1" if e110_payload else "E109",
    }
    return {
        "summary": summary,
        "e110": e110_payload,
        "e111": e111_payload,
        "e112": e112_payload,
        "e114": e114_payload,
        "aggregate": aggregate,
        "policy": read_json(e109 / "rank_policy_manifest.json"),
        "watch": read_json(e109 / "golden_watch_report.json"),
        "challenger": read_json(e109 / "challenger_prune_report.json"),
        "rows": rows,
    }


def render_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    escaped_data = data.replace("</", "<\\/")
    latest = html.escape(str(payload.get("summary", {}).get("latest_wave", "E109")))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VRAXION Operator Rank Dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0e1118;
      --panel: #171b25;
      --panel2: #111620;
      --line: #2c3446;
      --text: #edf2ff;
      --muted: #9aa7bd;
      --blue: #4da3ff;
      --green: #4be28a;
      --gold: #ffd35a;
      --silver: #cdd7e6;
      --bronze: #c88b57;
      --red: #ff5c7a;
      --violet: #b98cff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
      background: radial-gradient(circle at 18% 0%, #1b2740 0, transparent 34rem), var(--bg);
      color: var(--text);
      letter-spacing: 0;
    }}
    header {{
      padding: 24px 28px 14px;
      border-bottom: 1px solid var(--line);
      background: rgba(14,17,24,.86);
      position: sticky;
      top: 0;
      z-index: 10;
      backdrop-filter: blur(10px);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      font-weight: 740;
    }}
    .subtitle {{
      color: var(--muted);
      max-width: 1120px;
      line-height: 1.45;
    }}
    .wrap {{ padding: 20px 28px 40px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(6, minmax(130px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .card, .panel {{
      background: linear-gradient(180deg, rgba(255,255,255,.035), rgba(255,255,255,.01)), var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 14px 34px rgba(0,0,0,.18);
    }}
    .card {{ padding: 14px; min-height: 86px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .value {{ font-size: 26px; font-weight: 760; margin-top: 8px; }}
    .core {{ color: var(--violet); }}
    .gold {{ color: var(--gold); }}
    .silver {{ color: var(--silver); }}
    .bronze {{ color: var(--bronze); }}
    .red {{ color: var(--red); }}
    .green {{ color: var(--green); }}
    .toolbar {{
      display: grid;
      grid-template-columns: 1fr 210px 190px 190px;
      gap: 10px;
      margin-bottom: 14px;
    }}
    input, select, button {{
      background: var(--panel2);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px 11px;
      font: inherit;
      min-width: 0;
    }}
    button {{ cursor: pointer; }}
    button.active {{ border-color: var(--blue); box-shadow: 0 0 0 1px var(--blue) inset; }}
    .rank-buttons {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 14px;
    }}
    .rank-buttons button {{ padding: 8px 10px; }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(380px, 1fr) 420px;
      gap: 14px;
      align-items: start;
    }}
    .panel {{ padding: 14px; }}
    .panel h2 {{ margin: 0 0 12px; font-size: 17px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(18px, 1fr));
      gap: 5px;
      margin-bottom: 16px;
    }}
    .cell {{
      height: 22px;
      border-radius: 5px;
      border: 1px solid rgba(255,255,255,.12);
      cursor: pointer;
      opacity: .92;
    }}
    .cell:hover {{ transform: translateY(-1px); filter: brightness(1.2); }}
    .r-Gold {{ background: linear-gradient(180deg, #ffe28a, #b98017); }}
    .r-CoreMemoryCandidate {{ background: linear-gradient(180deg, #d7b6ff, #6847c7); }}
    .r-Silver {{ background: linear-gradient(180deg, #eef4ff, #6d7f9d); }}
    .r-Bronze {{ background: linear-gradient(180deg, #d99c63, #7f4f2a); }}
    .r-Deprecated {{ background: linear-gradient(180deg, #777, #333); }}
    .r-RedFlag {{ background: linear-gradient(180deg, #ff7890, #8e1931); }}
    .r-DiamondCandidate {{ background: linear-gradient(180deg, #cef8ff, #49adff); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 9px 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; position: sticky; top: 104px; background: var(--panel); }}
    tr {{ cursor: pointer; }}
    tr:hover {{ background: rgba(255,255,255,.035); }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 12px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,.035);
      white-space: nowrap;
    }}
    .bar {{
      width: 100%;
      height: 8px;
      background: #232b3b;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 6px;
    }}
    .bar > span {{
      display: block;
      height: 100%;
      background: linear-gradient(90deg, var(--blue), var(--green));
      width: 0;
    }}
    .detail-title {{ font-size: 20px; font-weight: 760; margin-bottom: 4px; }}
    .detail-id {{ color: var(--muted); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; overflow-wrap: anywhere; }}
    .kv {{
      display: grid;
      grid-template-columns: 160px 1fr;
      gap: 8px 12px;
      margin-top: 14px;
      font-size: 13px;
    }}
    .kv div:nth-child(odd) {{ color: var(--muted); }}
    .note {{
      margin-top: 14px;
      padding: 10px;
      border-radius: 7px;
      background: rgba(77,163,255,.08);
      border: 1px solid rgba(77,163,255,.28);
      color: #cfe4ff;
      line-height: 1.45;
    }}
    .table-wrap {{ max-height: 68vh; overflow: auto; border: 1px solid var(--line); border-radius: 8px; }}
    @media (max-width: 1180px) {{
      .cards {{ grid-template-columns: repeat(3, 1fr); }}
      .toolbar {{ grid-template-columns: 1fr 1fr; }}
      .layout {{ grid-template-columns: 1fr; }}
      th {{ top: 0; }}
    }}
    @media (max-width: 700px) {{
      header, .wrap {{ padding-left: 14px; padding-right: 14px; }}
      .cards {{ grid-template-columns: repeat(2, 1fr); }}
      .toolbar {{ grid-template-columns: 1fr; }}
      .kv {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>VRAXION Operator Rank Dashboard</h1>
    <div class="subtitle">{latest} rank view. Gold/Silver/Bronze are scoped ranks, not Core memory. Use this page to watch which operators are ready for Diamond/Core probation and which still need evidence.</div>
  </header>
  <div class="wrap">
    <section class="cards" id="cards"></section>

    <div class="toolbar">
      <input id="search" placeholder="Search operator, scope, family..." />
      <select id="rankFilter"></select>
      <select id="scopeFilter"></select>
      <select id="sortBy">
        <option value="rank">Sort by rank</option>
        <option value="activation">Sort by activation</option>
        <option value="remaining">Sort by Diamond remaining</option>
        <option value="value">Sort by counterfactual value</option>
        <option value="scope">Sort by scope</option>
      </select>
    </div>
    <div class="rank-buttons" id="rankButtons"></div>

    <section class="layout">
      <div class="panel">
        <h2>Rank Map</h2>
        <div class="grid" id="rankGrid"></div>
        <h2>Operators</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Operator</th>
                <th>Scope</th>
                <th>Activation</th>
                <th>Next target</th>
                <th>No-harm</th>
              </tr>
            </thead>
            <tbody id="rows"></tbody>
          </table>
        </div>
      </div>
      <aside class="panel" id="detail"></aside>
    </section>
  </div>
  <script>
    const DATA = {escaped_data};
    const rows = DATA.rows;
    const rankOrder = {{CoreMemoryCandidate: 0, DiamondCandidate: 1, Gold: 2, Silver: 3, Bronze: 4, Deprecated: 5, RedFlag: 6}};
    const nextTargets = {{
      Bronze: {{name: "Silver", value: 300}},
      Silver: {{name: "Gold", value: 3000}},
      Gold: {{name: "Diamond", value: 30000}},
      DiamondCandidate: {{name: "CoreCandidate", value: 100000}},
      CoreMemoryCandidate: {{name: "PermaCore probation", value: 300000}},
      Deprecated: {{name: "Stopped", value: 0}},
      RedFlag: {{name: "Stopped", value: 0}}
    }};
    let state = {{rank: "All", scope: "All", q: "", sort: "rank", selected: null}};

    function fmt(n) {{
      if (typeof n !== "number") return n ?? "";
      return n.toLocaleString();
    }}
    function pct(n) {{
      if (typeof n !== "number") return "";
      return (n * 100).toFixed(4) + "%";
    }}
    function rankClass(rank) {{ return "r-" + String(rank || "Bronze").replaceAll(" ", ""); }}
    function nextTarget(row) {{
      const target = nextTargets[row.rank] || nextTargets.Bronze;
      const remain = Math.max(0, target.value - (row.qualified_activation || 0));
      const progress = target.value > 0 ? Math.min(1, (row.qualified_activation || 0) / target.value) : 1;
      return {{...target, remain, progress}};
    }}
    function filtered() {{
      const q = state.q.toLowerCase().trim();
      let out = rows.filter(row => {{
        if (state.rank !== "All" && row.rank !== state.rank) return false;
        if (state.scope !== "All" && row.scope !== state.scope) return false;
        if (!q) return true;
        return [row.operator_id, row.display_name, row.scope, row.family, row.group_id, row.rank, row.e108_status]
          .join(" ").toLowerCase().includes(q);
      }});
      out.sort((a,b) => {{
        if (state.sort === "activation") return (b.qualified_activation||0) - (a.qualified_activation||0);
        if (state.sort === "remaining") return nextTarget(a).remain - nextTarget(b).remain;
        if (state.sort === "value") return (b.counterfactual_value||0) - (a.counterfactual_value||0);
        if (state.sort === "scope") return String(a.scope).localeCompare(String(b.scope)) || (rankOrder[a.rank] - rankOrder[b.rank]);
        return (rankOrder[a.rank] - rankOrder[b.rank]) || String(a.operator_id).localeCompare(String(b.operator_id));
      }});
      return out;
    }}
    function renderCards() {{
      const agg = DATA.aggregate;
      const cards = [
        ["CoreCandidate", agg.core_memory_candidate_count, "core"],
        ["Gold", agg.gold_count, "gold"],
        ["Silver", agg.silver_count, "silver"],
        ["Bronze", agg.bronze_count, "bronze"],
        ["Hard negative", agg.hard_negative_total, agg.hard_negative_total ? "red" : "green"],
        ["Qualified activations", fmt(agg.qualified_activation_total), "green"],
        ["E114 reaches target", agg.e114_projected_reach_permacore_count ?? "n/a", "green"],
        ["E114 targeted needed", agg.e114_projected_need_targeted_data_count ?? "n/a", agg.e114_projected_need_targeted_data_count ? "gold" : "green"]
      ];
      document.getElementById("cards").innerHTML = cards.map(([label,value,cls]) =>
        `<div class="card"><div class="label">${{label}}</div><div class="value ${{cls}}">${{value}}</div></div>`
      ).join("");
    }}
    function renderFilters() {{
      const ranks = ["All", ...Array.from(new Set(rows.map(r => r.rank))).sort((a,b) => rankOrder[a] - rankOrder[b])];
      const scopes = ["All", ...Array.from(new Set(rows.map(r => r.scope))).sort()];
      document.getElementById("rankFilter").innerHTML = ranks.map(r => `<option value="${{htmlEscape(r)}}">${{htmlEscape(r)}}</option>`).join("");
      document.getElementById("scopeFilter").innerHTML = scopes.map(s => `<option value="${{htmlEscape(s)}}">${{htmlEscape(s)}}</option>`).join("");
      document.getElementById("rankFilter").value = ranks.includes(state.rank) ? state.rank : "All";
      document.getElementById("scopeFilter").value = scopes.includes(state.scope) ? state.scope : "All";
      document.getElementById("rankButtons").innerHTML = ranks.map(r => {{
        const count = r === "All" ? rows.length : rows.filter(x => x.rank === r).length;
        return `<button data-rank="${{htmlEscape(r)}}" class="${{state.rank === r ? "active" : ""}}">${{htmlEscape(r)}} <span class="pill">${{count}}</span></button>`;
      }}).join("");
      document.querySelectorAll("#rankButtons button").forEach(btn => btn.onclick = () => {{
        state.rank = btn.dataset.rank;
        document.getElementById("rankFilter").value = state.rank;
        render();
      }});
    }}
    function renderGrid(items) {{
      document.getElementById("rankGrid").innerHTML = items.map(row =>
        `<div class="cell ${{rankClass(row.rank)}}" title="${{htmlEscape(row.rank)}} · ${{htmlEscape(row.operator_id)}} · ${{htmlEscape(row.scope)}}" data-id="${{htmlEscape(row.operator_id)}}"></div>`
      ).join("");
      document.querySelectorAll(".cell").forEach(cell => cell.onclick = () => select(cell.dataset.id));
    }}
    function renderRows(items) {{
      document.getElementById("rows").innerHTML = items.map(row => {{
        const target = nextTarget(row);
        const noharm = row.hard_negative === 0 ? "clean" : "flag";
        return `<tr data-id="${{htmlEscape(row.operator_id)}}">
          <td><span class="pill ${{String(row.rank).toLowerCase()}}">${{htmlEscape(row.rank)}}</span></td>
          <td><strong>${{htmlEscape(row.display_name || row.operator_id)}}</strong><br><span class="detail-id">${{htmlEscape(row.operator_id)}}</span></td>
          <td>${{htmlEscape(row.scope)}}<br><span class="detail-id">${{htmlEscape(row.group_id)}} · ${{htmlEscape(row.family)}}</span></td>
          <td>${{fmt(row.qualified_activation)}}<div class="bar"><span style="width:${{(target.progress*100).toFixed(1)}}%"></span></div></td>
          <td>${{target.name}}<br><span class="detail-id">${{fmt(target.remain)}} remaining</span></td>
          <td><span class="pill ${{noharm === "clean" ? "green" : "red"}}">${{noharm}}</span></td>
        </tr>`;
      }}).join("");
      document.querySelectorAll("tbody tr").forEach(tr => tr.onclick = () => select(tr.dataset.id));
    }}
    function htmlEscape(value) {{
      return String(value ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
    }}
    function select(id) {{
      state.selected = rows.find(r => r.operator_id === id) || null;
      renderDetail();
    }}
    function renderDetail(items = filtered()) {{
      if (state.selected && !items.some(row => row.operator_id === state.selected.operator_id)) {{
        state.selected = null;
      }}
      const row = state.selected || items[0] || rows[0];
      state.selected = row;
      const target = nextTarget(row);
      document.getElementById("detail").innerHTML = `
        <div class="detail-title">${{htmlEscape(row.display_name || row.operator_id)}}</div>
        <div class="detail-id">${{htmlEscape(row.operator_id)}}</div>
        <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">
          <span class="pill ${{String(row.rank).toLowerCase()}}">${{htmlEscape(row.rank)}}</span>
          <span class="pill">${{htmlEscape(row.scope)}}</span>
          <span class="pill">${{htmlEscape(row.watch_state)}}</span>
        </div>
        <div class="kv">
          <div>Qualified activation</div><div>${{fmt(row.qualified_activation)}}</div>
          <div>Next target</div><div>${{target.name}} · ${{fmt(target.remain)}} remaining<div class="bar"><span style="width:${{(target.progress*100).toFixed(1)}}%"></span></div></div>
          <div>Positive</div><div>${{fmt(row.positive)}}</div>
          <div>Neutral valid</div><div>${{fmt(row.neutral_valid)}}</div>
          <div>Neutral waste</div><div>${{fmt(row.neutral_waste)}} (${{pct(row.neutral_waste_rate)}})</div>
          <div>Hard negative</div><div>${{fmt(row.hard_negative)}}</div>
          <div>95% upper fail bound</div><div>${{pct(row.rule_of_three_upper_failure_bound)}}</div>
          <div>Family coverage</div><div>E107 ${{fmt(row.e107_family_coverage)}} + E108 ${{fmt(row.e108_family_coverage)}} = ${{fmt(row.combined_family_coverage)}}</div>
          <div>Campaign count</div><div>${{fmt(row.campaign_count)}}</div>
          <div>Counterfactual value</div><div>${{fmt(row.counterfactual_value)}} · gain ${{fmt(row.activated_gain)}} · ablation ${{fmt(row.ablation_loss)}}</div>
          <div>Reload / Challenger / Prune</div><div>${{row.reload_shadow_pass ? "reload pass" : "reload no"}} · ${{row.challenger_pass ? "challenger pass" : "challenger no"}} · ${{row.prune_pass ? "prune pass" : "prune no"}}</div>
          <div>Status source</div><div>E107 ${{htmlEscape(row.e107_status)}} · E108 ${{htmlEscape(row.e108_status)}}${{row.e110_wave1_outcome ? " · E110 " + htmlEscape(row.e110_wave1_outcome) : ""}}${{row.e111_wave2_outcome ? " · E111 " + htmlEscape(row.e111_wave2_outcome) : ""}}${{row.e112_wave3_outcome ? " · E112 " + htmlEscape(row.e112_wave3_outcome) : ""}}</div>
          <div>Latest activation add</div><div>${{fmt(row.qualified_activation_add || 0)}}</div>
          <div>Selected variant</div><div>${{htmlEscape(row.selected_variant_type || "")}}${{typeof row.selected_prune_ratio === "number" ? " · prune " + (row.selected_prune_ratio * 100).toFixed(1) + "%" : ""}}<br><span class="detail-id">${{htmlEscape(row.selected_variant_id || "")}}</span></div>
          <div>Mutation budget</div><div>${{fmt(row.mutation_attempts || 0)}} attempts · ${{fmt(row.accepted_mutations || 0)}} accepted · ${{fmt(row.rollback_count || 0)}} rollback</div>
          <div>Core no-harm</div><div>${{row.long_horizon_no_harm_pass ? "long-horizon pass" : ""}}${{row.negative_scope_pass ? " · negative-scope pass" : ""}}</div>
          <div>E114 FineWeb calls</div><div>${{fmt(row.e114_current_run_calls || 0)}} in 1M · projected full ${{fmt(row.e114_projected_full_fineweb_calls || 0)}}</div>
          <div>E114 PermaCore projection</div><div>${{row.e114_projected_reaches_permacore_probation ? "reaches 300k with full FineWeb" : "targeted pressure data needed"}} · remaining after full ${{fmt(row.e114_projected_remaining_after_full_fineweb || 0)}}</div>
          <div>E114 selected policy</div><div>${{htmlEscape(row.e114_selected_variant || "")}}</div>
        </div>
        <div class="note">${{row.rank === "CoreMemoryCandidate" ? "Interpretation: this operator passed scoped CoreMemoryCandidate probation. It is still not PermaCore or TrueGolden without a later larger no-harm grind." : "Interpretation: rank is scoped. This operator is not Core memory unless a later Core probation grind passes the much higher qualified-activation and no-harm gates."}}</div>
      `;
    }}
    function render() {{
      renderCards();
      renderFilters();
      const items = filtered();
      renderGrid(items);
      renderRows(items);
      renderDetail(items);
    }}
    document.getElementById("search").oninput = e => {{ state.q = e.target.value; render(); }};
    document.getElementById("rankFilter").onchange = e => {{ state.rank = e.target.value; render(); }};
    document.getElementById("scopeFilter").onchange = e => {{ state.scope = e.target.value; render(); }};
    document.getElementById("sortBy").onchange = e => {{ state.sort = e.target.value; render(); }};
    render();
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e109", default=str(DEFAULT_E109))
    parser.add_argument("--e110", default=str(DEFAULT_E110))
    parser.add_argument("--e111", default=str(DEFAULT_E111))
    parser.add_argument("--e112", default=str(DEFAULT_E112))
    parser.add_argument("--e114", default=str(DEFAULT_E114))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    e109 = existing_artifact_path(Path(args.e109), SAMPLE_E109, "rank_results.json")
    e110_requested = Path(args.e110)
    e111_requested = Path(args.e111)
    e112_requested = Path(args.e112)
    e114_requested = Path(args.e114)
    e110 = e110_requested if (e110_requested / "wave_results.json").exists() else SAMPLE_E110 if (SAMPLE_E110 / "wave_results.json").exists() else None
    e111 = e111_requested if (e111_requested / "wave_results.json").exists() else SAMPLE_E111 if (SAMPLE_E111 / "wave_results.json").exists() else None
    e112 = e112_requested if (e112_requested / "wave_results.json").exists() else SAMPLE_E112 if (SAMPLE_E112 / "wave_results.json").exists() else None
    e114 = e114_requested if (e114_requested / "operator_projection_report.json").exists() else None
    out = Path(args.out)
    payload = build_payload(e109, e110, e111, e112, e114)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload), encoding="utf-8")
    print(json.dumps({"out": str(out), "operator_count": len(payload["rows"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
