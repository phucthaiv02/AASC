from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LeaderboardRow:
    run: str
    attack: str
    score: float
    raw_mean: float
    findings: int
    unique_cells: int
    budget_s: float
    models: str
    summary_path: Path


def discover_rows(artifacts_dir: Path) -> list[LeaderboardRow]:
    root = artifacts_dir.expanduser().resolve()
    if not root.is_dir():
        return []

    rows: list[LeaderboardRow] = []
    for path in sorted(root.rglob("summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        models = payload.get("models")
        if not isinstance(models, list) or not models:
            continue
        valid_models = [item for item in models if isinstance(item, dict)]
        if not valid_models:
            continue

        relative_parent = path.parent.relative_to(root)
        run = str(relative_parent) if str(relative_parent) != "." else root.name
        score = _number(payload.get("local_public_mean"))
        if score is None:
            model_scores = [
                value
                for item in valid_models
                if (value := _number(item.get("score_normalized_0_to_1000"))) is not None
            ]
            if not model_scores:
                continue
            score = sum(model_scores) / len(model_scores)

        raw_scores = [
            value
            for item in valid_models
            if (value := _number(item.get("score_raw"))) is not None
        ]
        budget_values = [
            value
            for item in valid_models
            if (value := _number(item.get("budget_s"))) is not None
        ]
        attack = payload.get("attack")
        if not isinstance(attack, str) or not attack:
            attack = run
        rows.append(
            LeaderboardRow(
                run=run,
                attack=attack,
                score=score,
                raw_mean=sum(raw_scores) / len(raw_scores) if raw_scores else 0.0,
                findings=sum(_integer(item.get("findings_count")) for item in valid_models),
                unique_cells=sum(_integer(item.get("unique_cells")) for item in valid_models),
                budget_s=max(budget_values, default=0.0),
                models=", ".join(str(item.get("model", "unknown")) for item in valid_models),
                summary_path=path,
            )
        )
    return sorted(rows, key=lambda row: (-row.score, -row.raw_mean, row.run))


def render_terminal(rows: list[LeaderboardRow]) -> str:
    headers = ("#", "Run", "Attack", "Score", "Raw mean", "Findings", "Cells", "Budget")
    values = [
        (
            str(rank),
            row.run,
            row.attack,
            _format_number(row.score),
            _format_number(row.raw_mean),
            str(row.findings),
            str(row.unique_cells),
            f"{_format_number(row.budget_s)}s",
        )
        for rank, row in enumerate(rows, 1)
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        for index in range(len(headers))
    ] if values else [len(header) for header in headers]

    def line(items: tuple[str, ...]) -> str:
        return "  ".join(item.ljust(widths[index]) for index, item in enumerate(items)).rstrip()

    separator = tuple("-" * width for width in widths)
    return "\n".join([line(headers), line(separator), *(line(row) for row in values)])


def write_html(rows: list[LeaderboardRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    body_rows = []
    max_score = max((row.score for row in rows), default=1.0) or 1.0
    for rank, row in enumerate(rows, 1):
        width = max(1.0, row.score / max_score * 100.0)
        body_rows.append(
            "<tr>"
            f"<td>{rank}</td><td>{html.escape(row.run)}</td>"
            f"<td><code>{html.escape(row.attack)}</code></td>"
            f'<td class="score">{_format_number(row.score)}'
            f'<span class="bar" style="width:{width:.2f}%"></span></td>'
            f"<td>{_format_number(row.raw_mean)}</td><td>{row.findings}</td>"
            f"<td>{row.unique_cells}</td><td>{_format_number(row.budget_s)}s</td>"
            f"<td>{html.escape(row.models)}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>AAS local leaderboard</title><style>
:root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
body {{ max-width: 1400px; margin: 40px auto; padding: 0 20px; background:#0b1020; color:#e8ecf6; }}
h1 {{ margin-bottom:6px; }} p {{ color:#9da9c6; }}
table {{ width:100%; border-collapse:collapse; background:#131a2c; border-radius:12px; overflow:hidden; }}
th,td {{ padding:12px 14px; border-bottom:1px solid #27314c; text-align:left; white-space:nowrap; }}
th {{ background:#19233b; }} tr:first-child td:first-child {{ color:#ffd166; font-weight:700; }}
.score {{ position:relative; min-width:150px; font-weight:700; }}
.bar {{ display:block; height:4px; margin-top:6px; background:#4cc9f0; border-radius:4px; }}
code {{ color:#b9fbc0; }}
</style></head><body><h1>AAS local leaderboard</h1>
<p>{len(rows)} validation run(s), ranked by local public mean.</p>
<table><thead><tr><th>#</th><th>Run</th><th>Attack</th><th>Score</th><th>Raw mean</th>
<th>Findings</th><th>Cells</th><th>Budget/model</th><th>Models</th></tr></thead>
<tbody>{''.join(body_rows)}</tbody></table></body></html>"""
    output.write_text(document, encoding="utf-8")


def render_csv(rows: list[LeaderboardRow]) -> str:
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(("rank", "run", "attack", "score", "raw_mean", "findings", "unique_cells", "budget_s", "models"))
    for rank, row in enumerate(rows, 1):
        writer.writerow((rank, row.run, row.attack, row.score, row.raw_mean, row.findings, row.unique_cells, row.budget_s, row.models))
    return output.getvalue()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _format_number(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")

