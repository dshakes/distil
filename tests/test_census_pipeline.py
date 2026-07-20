"""Server-side census pipeline: validator (CI twin of the worker) + rollup."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # census_rollup imports census_validate by name
    spec.loader.exec_module(mod)
    return mod


census_validate = _load("census_validate")
census_rollup = _load("census_rollup")


def _row(**kw) -> dict:
    base = {
        "schema": 1,
        "install_id": "a" * 32,
        "version": "1.21.0",
        "os": "Darwin",
        "arch": "arm64",
        "python": "3.12.13",
        "runs": 10,
        "tokens_saved": 1000,
        "dollars_saved": 1.5,
        "ts": int(time.time()),
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# Validator — must agree with distil/census.py's frozen schema
# ---------------------------------------------------------------------------


def test_validator_accepts_real_client_payload(tmp_path, monkeypatch):
    """The payload distil/census.py actually builds passes the CI validator —
    the two frozen schemas cannot drift apart without failing here."""
    monkeypatch.setenv("DISTIL_HOME", str(tmp_path))
    from distil import census

    census.opt_in()
    assert census_validate.validate(census.build_payload()) is None


def test_validator_rejects(tmp_path):
    assert census_validate.validate(_row(extra=1)) is not None
    assert census_validate.validate(_row(install_id="zz")) is not None
    assert census_validate.validate(_row(tokens_saved=1e14)) is not None
    assert census_validate.validate(_row(runs=True)) is not None  # bool is not a count
    assert census_validate.validate(_row(os="../etc")) is not None
    assert census_validate.validate("not a dict") is not None


# ---------------------------------------------------------------------------
# Rollup
# ---------------------------------------------------------------------------


def _write_metrics(tmp_path: Path, census_rows: list[dict], adoption_rows: list[dict]) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    (data / "census.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in census_rows), encoding="utf-8"
    )
    (data / "adoption.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in adoption_rows), encoding="utf-8"
    )
    return tmp_path


def test_rollup_dedupes_by_install_id_latest_wins(tmp_path):
    now = int(time.time())
    rows = [
        _row(install_id="a" * 32, tokens_saved=100, ts=now - 100),
        _row(install_id="a" * 32, tokens_saved=500, ts=now),  # same machine, later
        _row(install_id="b" * 32, tokens_saved=50, ts=now, version="1.20.0"),
    ]
    agg = census_rollup.rollup(_write_metrics(tmp_path, rows, []), now=now)
    assert agg["installs"]["census_total"] == 2
    assert agg["savings"]["tokens"] == 550  # 500 (latest a) + 50 (b), not 650
    assert agg["installs"]["by_version"] == {"1.21.0": 1, "1.20.0": 1}


def test_rollup_active_windows(tmp_path):
    now = int(time.time())
    rows = [
        _row(install_id="a" * 32, ts=now - 86400),  # 1 day → active in both
        _row(install_id="b" * 32, ts=now - 86400 * 20),  # 20d → 30d only
        _row(install_id="c" * 32, ts=now - 86400 * 90),  # stale
    ]
    agg = census_rollup.rollup(_write_metrics(tmp_path, rows, []), now=now)
    assert agg["installs"]["active_7d"] == 1
    assert agg["installs"]["active_30d"] == 2
    assert agg["installs"]["census_total"] == 3


def test_rollup_drops_invalid_rows_and_survives_corruption(tmp_path):
    now = int(time.time())
    valid = _row(install_id="a" * 32, ts=now)
    metrics = _write_metrics(tmp_path, [valid, _row(tokens_saved=1e14, install_id="b" * 32)], [])
    with (metrics / "data" / "census.jsonl").open("a") as f:
        f.write("{corrupt json\n")
    agg = census_rollup.rollup(metrics, now=now)
    assert agg["installs"]["census_total"] == 1  # skew row + corrupt line dropped


def test_rollup_merges_passive_channels_and_badges(tmp_path):
    now = int(time.time())
    adoption = [
        {
            "ts": now,
            "pypi_downloads": {"month": 11943, "week": 1632, "day": 46},
            "github": {"stars": 5, "forks": 1},
            "clones": {"uniques_14d": 335, "count_14d": 2610},
            "docker": {"skipped": "no image configured"},
        }
    ]
    rows = [_row(install_id="a" * 32, tokens_saved=1_500_000_000, ts=now)]
    agg = census_rollup.rollup(_write_metrics(tmp_path, rows, adoption), now=now)
    assert agg["channels"]["pypi_downloads_month"] == 11943
    assert agg["channels"]["clones_uniques_14d"] == 335
    badges = census_rollup.badges(agg)
    assert badges["savings-tokens"]["message"] == "1.5B"
    assert badges["downloads-month"]["message"] == "11.9k"
    assert all(b["schemaVersion"] == 1 for b in badges.values())


# ---------------------------------------------------------------------------
# Schema 2 (usage dimensions)
# ---------------------------------------------------------------------------


def _row2(**kw) -> dict:
    base = _row(
        billing="metered",
        by_model={"claude-opus-4-8": 500, "claude-haiku-4-5": 100},
        agents=["claude"],
    )
    base["schema"] = 2
    base.update(kw)
    return base


def test_validator_accepts_both_schemas():
    assert census_validate.validate(_row()) is None  # v1 clients stay valid
    assert census_validate.validate(_row2()) is None


def test_validator_rejects_bad_v2_fields():
    assert census_validate.validate(_row2(billing="free")) is not None
    assert census_validate.validate(_row2(agents=["hacker-tool"])) is not None
    assert census_validate.validate(_row2(by_model={"m/../etc": 1})) is not None
    assert census_validate.validate(_row2(by_model={f"m{i}": 1 for i in range(9)})) is not None
    assert census_validate.validate(_row2(by_model={"claude": 1e14})) is not None
    bad = _row2()
    del bad["agents"]
    assert census_validate.validate(bad) is not None  # v2 keys are all-or-nothing


def test_rollup_aggregates_usage(tmp_path):
    now = int(time.time())
    rows = [
        _row2(install_id="a" * 32, ts=now, billing="metered", agents=["claude", "other"]),
        _row2(
            install_id="b" * 32,
            ts=now,
            billing="subscription",
            by_model={"claude-opus-4-8": 200},
            agents=["claude"],
        ),
        _row(install_id="c" * 32, ts=now),  # v1 row: counted, no usage contribution
    ]
    agg = census_rollup.rollup(_write_metrics(tmp_path, rows, []), now=now)
    assert agg["installs"]["census_total"] == 3
    assert agg["usage"]["by_model"]["claude-opus-4-8"] == 700
    assert agg["usage"]["billing"] == {"metered": 1, "subscription": 1}
    assert agg["usage"]["agents"]["claude"] == 2


def test_rollup_buckets_dollars_by_billing(tmp_path):
    """Metered $ = real; subscription $ = notional; v1 rows (no billing) = notional."""
    now = int(time.time())
    rows = [
        _row2(install_id="a" * 32, ts=now, billing="metered", dollars_saved=10.0),
        _row2(install_id="b" * 32, ts=now, billing="subscription", dollars_saved=99.0),
        _row(install_id="c" * 32, ts=now, dollars_saved=5.0),  # v1: no billing
    ]
    agg = census_rollup.rollup(_write_metrics(tmp_path, rows, []), now=now)
    assert agg["savings"]["dollars"] == 10.0
    assert agg["savings"]["dollars_notional"] == 104.0
