"""`distil retention` CLI surface — exit codes are the contract.

These are the paths a CI job depends on, so what matters is that a bad outcome exits
non-zero and a good one exits zero. No network: the dataset loader is stubbed.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from distil import datasets as ds
from distil.cli import main


class _Case:
    def __init__(self, docs: list[tuple[str, str]], answer: str, support: list[str]) -> None:
        self.id = "c1"
        self.question = "q?"
        self.docs = docs
        self.answer = answer
        self.support = support
        self.answerable = True
        self.dataset = "stub"


def _compressible(answer: str = "Tim Burton") -> list[_Case]:
    docs = [
        (f"d{i}", f"Paragraph {i} about films and directors. id={i} path=/srv/x{i}.txt")
        for i in range(12)
    ]
    docs.append(("gold", f"The director was {answer}."))
    return [_Case(docs, answer, [f"The director was {answer}."])]


def test_corpus_mode_passes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["retention"]) == 0
    assert "fact-level retention" in capsys.readouterr().out


def test_corpus_gate_passes_at_zero_lost(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["retention", "--max-lost", "0"]) == 0


def test_corpus_gate_fails_when_budget_is_impossible(capsys: pytest.CaptureFixture[str]) -> None:
    """A negative budget can never be met — proves the gate can actually fail."""
    assert main(["retention", "--max-lost", "-1"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_corpus_json_is_machine_readable(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["retention", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall"]["lost"] == 0
    assert set(payload["by_dimension"]) == {"numerics", "artifacts", "errors"}


def test_dataset_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["retention", "--dataset", "list"]) == 0
    out = capsys.readouterr().out
    assert "hotpotqa" in out and "squad" in out


def test_dataset_mode_scores_and_passes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ds, "load", lambda name, n, offline=False: _compressible())
    assert main(["retention", "--dataset", "hotpotqa", "-n", "1", "--min-recall", "0.9"]) == 0
    assert "public benchmark" in capsys.readouterr().out


def test_dataset_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(ds, "load", lambda name, n, offline=False: _compressible())
    assert main(["retention", "--dataset", "hotpotqa", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset"] == "hotpotqa"
    assert "baseline" in payload and "distil" in payload


def test_fetch_failure_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The failure that must never be silent: a gate that measured nothing."""

    def boom(name: str, n: Any, offline: bool = False) -> Any:
        raise ds.DatasetUnavailable("datasets-server unreachable after 3 attempts")

    monkeypatch.setattr(ds, "load", boom)
    assert main(["retention", "--dataset", "hotpotqa", "--min-recall", "0.99"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_min_recall_rejects_unengaged_compression(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Short prose does not compress; a recall gate must NOT pass on an identity
    transform, because 100% there is arithmetic rather than evidence."""
    monkeypatch.setattr(
        ds, "load", lambda name, n, offline=False: [_Case([("d", "Tiny.")], "Tiny", [])]
    )
    code = main(["retention", "--dataset", "squad", "--shape", "prose", "--min-recall", "0.99"])
    assert code == 1
    assert "compression did not engage" in capsys.readouterr().out


def test_min_recall_fails_on_a_real_regression(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Force genuine loss by making the answer absent from the compressed text AND
    from any restore entry: the gate must fail rather than round up."""
    import distil.retention as r

    monkeypatch.setattr(ds, "load", lambda name, n, offline=False: _compressible())

    def lost_report(cases: Any, dataset: str = "", *, shape: str = "json") -> r.DatasetReport:
        # Engaged compression (savings well above the floor) that lost the answer with
        # no recovery path — the one outcome the gate exists to catch.
        rep = r.DatasetReport(dataset=dataset, shape=shape)
        rep.scores.append(
            r.CaseScore(case_id="c1", savings=0.4, answer_retained=False, answer_recoverable=False)
        )
        rep.baseline.append(
            r.CaseScore(case_id="c1", savings=0.4, answer_retained=False, answer_recoverable=False)
        )
        return rep

    monkeypatch.setattr(r, "score_dataset", lost_report)
    code = main(["retention", "--dataset", "hotpotqa", "--min-recall", "0.99"])
    assert code == 1
    out = capsys.readouterr().out
    assert "answer recall 0.0% < --min-recall" in out


def test_live_mode_reads_the_meter(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    assert main(["retention", "--live"]) == 0
    assert "no live retention samples yet" in capsys.readouterr().out

    from distil.retention import LiveMeter

    original = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t",
                    "content": "latency_ms=42 path=/var/log/app/a.log",
                }
            ],
        }
    ]
    LiveMeter(1.0).observe(original, original, None)
    assert main(["retention", "--live"]) == 0
    assert "live fact retention" in capsys.readouterr().out


def test_worker_config_carries_retention_rate() -> None:
    """wrap's default path runs the proxy in a hot-swap WORKER, so a flag that is not
    in WorkerConfig silently never takes effect there."""
    from distil.hotswap import WorkerConfig

    cfg = WorkerConfig(upstream="http://x", retention_rate=0.25)
    assert json.loads(cfg.to_env())["retention_rate"] == 0.25


def test_worker_config_tolerates_unknown_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mid-upgrade, a newer supervisor can spawn an older worker. An unrecognised
    config key must not kill the session."""
    from distil.hotswap import WorkerConfig

    monkeypatch.setenv(
        "DISTIL_WORKER_CONFIG",
        json.dumps({"upstream": "http://x", "retention_rate": 0.5, "from_the_future": True}),
    )
    cfg = WorkerConfig.from_env()
    assert cfg.upstream == "http://x" and cfg.retention_rate == 0.5
