import json

from aas_nim_validation.leaderboard import discover_rows, render_terminal, write_html


def _summary(path, attack, score):
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "attack": attack,
                "local_public_mean": score,
                "models": [
                    {
                        "model": "model/a",
                        "score_normalized_0_to_1000": score,
                        "score_raw": score * 200,
                        "findings_count": 2,
                        "unique_cells": 2,
                        "budget_s": 300,
                    }
                ],
            }
        )
    )


def test_discovers_ranks_and_renders(tmp_path):
    _summary(tmp_path / "runs" / "low" / "summary.json", "attacks/low.py", 0.1)
    _summary(tmp_path / "runs" / "high" / "summary.json", "attacks/high.py", 0.4)
    (tmp_path / "broken" / "summary.json").parent.mkdir()
    (tmp_path / "broken" / "summary.json").write_text("not-json")

    rows = discover_rows(tmp_path)

    assert [row.run for row in rows] == ["runs/high", "runs/low"]
    assert "attacks/high.py" in render_terminal(rows)
    output = tmp_path / "leaderboard.html"
    write_html(rows, output)
    assert "AAS local leaderboard" in output.read_text()

