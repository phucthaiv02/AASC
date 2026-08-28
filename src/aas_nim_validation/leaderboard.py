from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass, replace
from io import StringIO
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LeaderboardRow:
    run: str
    score: float | None
    raw_mean: float | None
    public_lb_score: float | None
    gemma_score: float | None
    gpt_oss_score: float | None
    baseline_run: str | None = None
    score_delta: float | None = None
    public_lb_delta: float | None = None
    gemma_delta: float | None = None
    gpt_oss_delta: float | None = None


def discover_rows(validation_dir: Path) -> list[LeaderboardRow]:
    root = validation_dir.expanduser().resolve()
    if not root.is_dir():
        return []

    rows = [_read_run(run_dir) for run_dir in sorted(path for path in root.iterdir() if path.is_dir())]
    if not rows:
        return []

    baseline = next((row for row in rows if row.run == "baseline"), None)
    if baseline is None or baseline.score is None:
        raise ValueError(f"Baseline summaries not found under {root / 'baseline'}")

    compared = [_compare(row, baseline) for row in rows]
    return sorted(
        compared,
        key=lambda row: (
            row.run == "baseline",
            row.score is None,
            -(row.score or 0.0),
            -(row.raw_mean or 0.0),
            row.run,
        ),
    )


def _read_run(run_dir: Path) -> LeaderboardRow:
    models: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*_summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and _number(payload.get("score_normalized_0_to_1000")) is not None:
            models.append(payload)

    if not models:
        return LeaderboardRow(
            run=run_dir.name,
            score=None,
            raw_mean=None,
            public_lb_score=None,
            gemma_score=None,
            gpt_oss_score=None,
        )

    scores = [_required_number(item["score_normalized_0_to_1000"]) for item in models]
    raw_scores = [
        value for item in models if (value := _number(item.get("score_raw"))) is not None
    ]
    model_scores = {
        str(item.get("model")): _number(item.get("score_normalized_0_to_1000"))
        for item in models
    }
    return LeaderboardRow(
        run=run_dir.name,
        score=sum(scores) / len(scores),
        raw_mean=sum(raw_scores) / len(raw_scores) if raw_scores else 0.0,
        public_lb_score=_read_public_lb_score(run_dir / "public_lb.csv"),
        gemma_score=model_scores.get("gemma"),
        gpt_oss_score=model_scores.get("gpt_oss"),
    )


def _compare(row: LeaderboardRow, baseline: LeaderboardRow) -> LeaderboardRow:
    if row.run == "baseline":
        return replace(row, baseline_run=baseline.run)
    if row.score is None or baseline.score is None:
        return replace(row, baseline_run=baseline.run)
    score_delta = row.score - baseline.score
    public_lb_delta = (
        row.public_lb_score - baseline.public_lb_score
        if row.public_lb_score is not None and baseline.public_lb_score is not None
        else None
    )
    gemma_delta = _difference(row.gemma_score, baseline.gemma_score)
    gpt_oss_delta = _difference(row.gpt_oss_score, baseline.gpt_oss_score)
    return replace(
        row,
        baseline_run=baseline.run,
        score_delta=score_delta,
        public_lb_delta=public_lb_delta,
        gemma_delta=gemma_delta,
        gpt_oss_delta=gpt_oss_delta,
    )


def render_terminal(rows: list[LeaderboardRow]) -> str:
    headers = (
        "#",
        "Run",
        "Score",
        "Gemma",
        "GPT-OSS",
        "Public LB",
        "Raw mean",
    )
    values = [
        (
            "" if row.run == "baseline" else str(rank),
            row.run,
            _format_score_comparison(row),
            _format_value_comparison(row.gemma_score, row.gemma_delta),
            _format_value_comparison(row.gpt_oss_score, row.gpt_oss_delta),
            _format_public_lb_comparison(row),
            _format_optional_number(row.raw_mean),
        )
        for rank, row in enumerate(rows, 1)
    ]
    widths = (
        [max(len(headers[index]), *(len(row[index]) for row in values)) for index in range(len(headers))]
        if values
        else [len(header) for header in headers]
    )

    def line(items: tuple[str, ...]) -> str:
        return "  ".join(item.ljust(widths[index]) for index, item in enumerate(items)).rstrip()

    separator = tuple("-" * width for width in widths)
    return "\n".join([line(headers), line(separator), *(line(row) for row in values)])


def write_html(rows: list[LeaderboardRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    body_rows = []
    for rank, row in enumerate(rows, 1):
        body_rows.append(
            f'<tr class="{"baseline" if row.run == "baseline" else ""}">'
            f'<td data-sort="{rank if row.run != "baseline" else ""}">'
            f"{'' if row.run == 'baseline' else rank}</td>"
            f'<td data-sort="{html.escape(row.run)}">{html.escape(row.run)}</td>'
            f'<td class="score" data-sort="{_sort_value(row.score)}">'
            f"{_format_optional_number(row.score)} "
            f"{_format_html_change(row.score_delta)}"
            "</td>"
            f'<td data-sort="{_sort_value(row.gemma_score)}">'
            f"{_format_optional_number(row.gemma_score)} "
            f"{_format_html_change(row.gemma_delta)}</td>"
            f'<td data-sort="{_sort_value(row.gpt_oss_score)}">'
            f"{_format_optional_number(row.gpt_oss_score)} "
            f"{_format_html_change(row.gpt_oss_delta)}</td>"
            f'<td data-sort="{_sort_value(row.public_lb_score)}">'
            f"{_format_optional_number(row.public_lb_score)} "
            f"{_format_html_change(row.public_lb_delta)}</td>"
            f'<td data-sort="{_sort_value(row.raw_mean)}">'
            f"{_format_optional_number(row.raw_mean)}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>AAS validation leaderboard</title><style>
:root {{ font-family:Inter,ui-sans-serif,system-ui,sans-serif; font-weight:400; }}
body {{ max-width:1500px; margin:0 auto; padding:48px 24px 72px; background:#f5f2ed; color:#000; min-height:100vh; }}
h1 {{ margin:0 0 8px; color:#000; font-weight:400; }}
p {{ margin:0 0 28px; color:#000; }}
table {{ width:100%; border-collapse:separate; border-spacing:0; background:#fff; border:1px solid #ddd6cf; border-radius:10px; overflow:hidden; }}
th,td {{ padding:13px 15px; border-bottom:1px solid #e7e1dc; text-align:left; white-space:nowrap; font-weight:400; }}
th {{ background:#eee8e2; color:#000; }}
th button {{ padding:0; border:0; background:transparent; color:#000; font:inherit; font-weight:400; cursor:pointer; }}
th button::after {{ content:' ↕'; color:#666; }}
th button[data-direction='asc']::after {{ content:' ▲'; color:#16823b; }}
th button[data-direction='desc']::after {{ content:' ▼'; color:#d12f2f; }}
tbody tr:nth-child(even) {{ background:#faf8f5; }}
tbody tr:last-child td {{ border-bottom:0; }}
tr.baseline {{ background:#fff7df; color:#000; }}
.score {{ position:relative; min-width:150px; color:#000; }}
.positive {{ color:#16823b; }}
.negative {{ color:#d12f2f; }}
.neutral {{ color:#000; }}
.indicator {{ display:inline-block; margin:0 3px 0 5px; font-size:1.3rem; line-height:.7; vertical-align:-.08em; }}
</style></head><body><h1>AAS validation leaderboard</h1>
<p>{len(rows)} validation run(s), compared with baseline.</p>
<table><thead><tr><th><button data-column="0">#</button></th>
<th><button data-column="1">Run</button></th><th><button data-column="2">Score</button></th>
<th><button data-column="3">Gemma</button></th><th><button data-column="4">GPT-OSS</button></th>
<th><button data-column="5">Public LB</button></th><th><button data-column="6">Raw mean</button></th>
</tr></thead>
<tbody id="leaderboard-body">{''.join(body_rows)}</tbody></table>
<script>
const body=document.getElementById('leaderboard-body');
document.querySelectorAll('th button').forEach(button=>button.addEventListener('click',()=>{{
  const column=Number(button.dataset.column);
  const direction=button.dataset.direction==='desc'?'asc':'desc';
  document.querySelectorAll('th button').forEach(item=>delete item.dataset.direction);
  button.dataset.direction=direction;
  const rows=[...body.rows];
  rows.sort((left,right)=>{{
    const a=left.cells[column].dataset.sort;
    const b=right.cells[column].dataset.sort;
    if(a===''||b==='') return a===''?(b===''?0:1):-1;
    const an=Number(a),bn=Number(b);
    const result=Number.isNaN(an)||Number.isNaN(b)?a.localeCompare(b):an-bn;
    return direction==='asc'?result:-result;
  }});
  rows.forEach(row=>body.appendChild(row));
}}));
</script></body></html>"""
    output.write_text(document, encoding="utf-8")


def render_csv(rows: list[LeaderboardRow]) -> str:
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        (
            "rank",
            "run",
            "score",
            "baseline_run",
            "score_delta",
            "gemma_score",
            "gemma_delta",
            "gpt_oss_score",
            "gpt_oss_delta",
            "public_lb_score",
            "public_lb_delta",
            "raw_mean",
        )
    )
    for rank, row in enumerate(rows, 1):
        writer.writerow(
            (
                "" if row.run == "baseline" else rank,
                row.run,
                row.score,
                row.baseline_run,
                row.score_delta,
                row.gemma_score,
                row.gemma_delta,
                row.gpt_oss_score,
                row.gpt_oss_delta,
                row.public_lb_score,
                row.public_lb_delta,
                row.raw_mean,
            )
        )
    return output.getvalue()


def _read_public_lb_score(path: Path) -> float | None:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            scores = [
                score
                for row in csv.DictReader(handle)
                if (score := _number_from_text(row.get("score"))) is not None
            ]
    except OSError:
        return None
    return sum(scores) / len(scores) if scores else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _required_number(value: Any) -> float:
    number = _number(value)
    if number is None:
        raise TypeError("Expected a numeric score")
    return number


def _difference(value: float | None, baseline: float | None) -> float | None:
    return value - baseline if value is not None and baseline is not None else None


def _number_from_text(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: float) -> str:
    return f"{value:.2f}"


def _format_optional_number(value: float | None) -> str:
    return "N/A" if value is None else _format_number(value)


def _sort_value(value: float | None) -> str:
    return "" if value is None else str(value)


def _format_delta(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.2f}"


def _format_score_comparison(row: LeaderboardRow) -> str:
    if row.score is None:
        return "N/A"
    change = _format_change(row.score_delta)
    return f"{_format_number(row.score)} {change}"


def _format_public_lb_comparison(row: LeaderboardRow) -> str:
    value = _format_optional_number(row.public_lb_score)
    return f"{value} {_format_change(row.public_lb_delta)}" if row.public_lb_score is not None else value


def _format_value_comparison(value: float | None, delta: float | None) -> str:
    formatted = _format_optional_number(value)
    return f"{formatted} {_format_change(delta)}" if value is not None else formatted


def _format_change(value: float | None) -> str:
    if value is None:
        return ""
    indicator = "▲" if value > 0 else "▼" if value < 0 else "—"
    return f"{indicator} {_format_delta(value)}"


def _format_html_change(value: float | None) -> str:
    if value is None:
        return ""
    indicator = "▲" if value > 0 else "▼" if value < 0 else "—"
    color = _delta_class(value)
    return (
        f'<span class="indicator {color}">{indicator}</span>'
        f'<span class="{color}">{_format_delta(value)}</span>'
    )


def _delta_class(value: float | None) -> str:
    if value is None or value == 0:
        return "neutral"
    return "positive" if value > 0 else "negative"
