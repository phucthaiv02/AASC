import csv
import json
from io import StringIO

import pytest

from aas_nim_validation.leaderboard import discover_rows, render_csv, render_terminal, write_html


def _validation_run(root, name, scores, public_scores=()):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    for model, score in scores.items():
        (run_dir / f"{model}_summary.json").write_text(
            json.dumps(
                {
                    "model": model,
                    "score_normalized_0_to_1000": score,
                    "score_raw": score * 200,
                    "findings_count": 2,
                    "unique_cells": 2,
                }
            )
        )
    if public_scores:
        content = "try,score\n" + "".join(
            f"{index},{score}\n" for index, score in enumerate(public_scores, 1)
        )
        (run_dir / "public_lb.csv").write_text(content)


def test_discovers_new_validation_structure_and_compares_baseline(tmp_path):
    validation = tmp_path / "validation"
    _validation_run(validation, "baseline", {"gemma": 190, "gpt_oss": 224}, (88.38, 88.56))
    _validation_run(validation, "adaptive", {"gemma": 220, "gpt_oss": 202}, (76.59, 76.185))
    _validation_run(validation, "bandit", {"gemma": 230, "gpt_oss": 240})
    (validation / "pending").mkdir()

    rows = discover_rows(validation)

    assert [row.run for row in rows] == ["bandit", "adaptive", "pending", "baseline"]
    pending = next(row for row in rows if row.run == "pending")
    assert pending.score is None
    assert pending.gemma_score is None
    assert pending.gpt_oss_score is None
    assert pending.public_lb_score is None
    assert pending.raw_mean is None
    adaptive = next(row for row in rows if row.run == "adaptive")
    assert adaptive.baseline_run == "baseline"
    assert adaptive.score == 211
    assert adaptive.score_delta == 4
    assert adaptive.public_lb_score == pytest.approx(76.3875)
    assert adaptive.public_lb_delta == pytest.approx(-12.0825)

    terminal = render_terminal(rows)
    assert "211.00 ▲ +4.00" in terminal
    assert "76.39 ▼ -12.08" in terminal
    assert "pending" in terminal
    assert "N/A" in terminal

    output = tmp_path / "leaderboard.html"
    write_html(rows, output)
    document = output.read_text()
    assert "AAS validation leaderboard" in document
    assert "adaptive" in document
    assert 'data-column="2">Score</button>' in document
    assert "leaderboard-body" in document

    csv_rows = list(csv.DictReader(StringIO(render_csv(rows))))
    assert csv_rows[1]["baseline_run"] == "baseline"
    assert csv_rows[1]["score_delta"] == "4.0"


def test_requires_baseline_directory(tmp_path):
    validation = tmp_path / "validation"
    _validation_run(validation, "adaptive", {"gemma": 220})

    with pytest.raises(ValueError, match="Baseline summaries not found"):
        discover_rows(validation)
