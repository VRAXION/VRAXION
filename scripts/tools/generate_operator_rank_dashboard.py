#!/usr/bin/env python3
"""Generate a self-contained Operator rank dashboard from E109 artifacts."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


DEFAULT_E109 = Path("target/pilot_wave/e109_operator_rank_ladder_and_golden_watch_probation_mode")
DEFAULT_OUT = Path("target/pilot_wave/operator_rank_dashboard/index.html")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    ]
    return [{key: row.get(key) for key in keep} for row in rows]


def build_payload(e109: Path) -> dict[str, Any]:
    rank_results = read_json(e109 / "rank_results.json")
    return {
        "summary": read_json(e109 / "summary.json"),
        "aggregate": read_json(e109 / "aggregate_metrics.json"),
        "policy": read_json(e109 / "rank_policy_manifest.json"),
        "watch": read_json(e109 / "golden_watch_report.json"),
        "challenger": read_json(e109 / "challenger_prune_report.json"),
        "rows": compact_rows(rank_results["rows"]),
    }


def render_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    escaped_data = data.replace("</", "<\\/")
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
    <div class="subtitle">E109 rank ladder view. Gold/Silver/Bronze are scoped ranks, not Core memory. Use this page to watch which operators are ready for Diamond/Core probation and which still need evidence.</div>
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
    const rankOrder = {{DiamondCandidate: 0, Gold: 1, Silver: 2, Bronze: 3, Deprecated: 4, RedFlag: 5}};
    const nextTargets = {{
      Bronze: {{name: "Silver", value: 300}},
      Silver: {{name: "Gold", value: 3000}},
      Gold: {{name: "Diamond", value: 30000}},
      DiamondCandidate: {{name: "CoreCandidate", value: 100000}},
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
        ["Gold", agg.gold_count, "gold"],
        ["Silver", agg.silver_count, "silver"],
        ["Bronze", agg.bronze_count, "bronze"],
        ["Diamond", agg.diamond_candidate_count, "green"],
        ["Hard negative", agg.hard_negative_total, agg.hard_negative_total ? "red" : "green"],
        ["Qualified activations", fmt(agg.qualified_activation_total), "green"]
      ];
      document.getElementById("cards").innerHTML = cards.map(([label,value,cls]) =>
        `<div class="card"><div class="label">${{label}}</div><div class="value ${{cls}}">${{value}}</div></div>`
      ).join("");
    }}
    function renderFilters() {{
      const ranks = ["All", ...Array.from(new Set(rows.map(r => r.rank))).sort((a,b) => rankOrder[a] - rankOrder[b])];
      const scopes = ["All", ...Array.from(new Set(rows.map(r => r.scope))).sort()];
      document.getElementById("rankFilter").innerHTML = ranks.map(r => `<option>${{r}}</option>`).join("");
      document.getElementById("scopeFilter").innerHTML = scopes.map(s => `<option>${{s}}</option>`).join("");
      document.getElementById("rankButtons").innerHTML = ranks.map(r => {{
        const count = r === "All" ? rows.length : rows.filter(x => x.rank === r).length;
        return `<button data-rank="${{r}}" class="${{state.rank === r ? "active" : ""}}">${{r}} <span class="pill">${{count}}</span></button>`;
      }}).join("");
      document.querySelectorAll("#rankButtons button").forEach(btn => btn.onclick = () => {{
        state.rank = btn.dataset.rank;
        document.getElementById("rankFilter").value = state.rank;
        render();
      }});
    }}
    function renderGrid(items) {{
      document.getElementById("rankGrid").innerHTML = items.map(row =>
        `<div class="cell ${{rankClass(row.rank)}}" title="${{row.rank}} · ${{row.operator_id}} · ${{row.scope}}" data-id="${{row.operator_id}}"></div>`
      ).join("");
      document.querySelectorAll(".cell").forEach(cell => cell.onclick = () => select(cell.dataset.id));
    }}
    function renderRows(items) {{
      document.getElementById("rows").innerHTML = items.map(row => {{
        const target = nextTarget(row);
        const noharm = row.hard_negative === 0 ? "clean" : "flag";
        return `<tr data-id="${{row.operator_id}}">
          <td><span class="pill ${{row.rank.toLowerCase()}}">${{row.rank}}</span></td>
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
      state.selected = rows.find(r => r.operator_id === id) || state.selected;
      renderDetail();
    }}
    function renderDetail() {{
      const row = state.selected || filtered()[0] || rows[0];
      state.selected = row;
      const target = nextTarget(row);
      document.getElementById("detail").innerHTML = `
        <div class="detail-title">${{htmlEscape(row.display_name || row.operator_id)}}</div>
        <div class="detail-id">${{htmlEscape(row.operator_id)}}</div>
        <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">
          <span class="pill ${{row.rank.toLowerCase()}}">${{row.rank}}</span>
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
          <div>Status source</div><div>E107 ${{htmlEscape(row.e107_status)}} · E108 ${{htmlEscape(row.e108_status)}}</div>
        </div>
        <div class="note">Interpretation: rank is scoped. This operator is not Core memory unless a later Core probation grind passes the much higher qualified-activation and no-harm gates.</div>
      `;
    }}
    function render() {{
      renderCards();
      renderFilters();
      const items = filtered();
      renderGrid(items);
      renderRows(items);
      renderDetail();
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
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    e109 = Path(args.e109)
    out = Path(args.out)
    payload = build_payload(e109)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload), encoding="utf-8")
    print(json.dumps({"out": str(out), "operator_count": len(payload["rows"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
