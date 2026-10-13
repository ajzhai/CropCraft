"""Pre-flight checks, against the failures that actually happened.

Each of these is a real incident on this project, reproduced. That is the point
of the module: none of them were interesting, all of them were knowable in
seconds, and every one was found somewhere between forty minutes and eight hours
into a run instead.
"""

from __future__ import annotations

from ggssvt.eval.preflight import DEGRADED, FATAL, Finding, _proxies, report


def test_a_doubled_scheme_is_fatal(monkeypatch):
    """The exact value that broke git on the lab machine."""
    monkeypatch.setenv("http_proxy", "http://http://www.up.ac.za/proxy.pac:8080/")
    found = {f.name: f for f in _proxies()}
    assert found["http_proxy"].level == FATAL
    assert "two schemes" in found["http_proxy"].detail


def test_a_pac_file_used_as_a_proxy_is_fatal(monkeypatch):
    """A .pac file chooses a proxy; it is not one, and no tool can use it."""
    monkeypatch.setenv("https_proxy", "http://www.up.ac.za/proxyc1.pac:8080/")
    found = {f.name: f for f in _proxies()}
    assert found["https_proxy"].level == FATAL
    assert "PAC" in found["https_proxy"].detail


def test_a_well_formed_proxy_passes(monkeypatch):
    monkeypatch.setenv("https_proxy", "http://proxy.up.ac.za:8080")
    assert all(f.level == "ok" for f in _proxies())


def test_no_proxy_at_all_passes(monkeypatch):
    for var in ("http_proxy", "https_proxy", "all_proxy",
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    assert all(f.level == "ok" for f in _proxies())


def test_conflict_markers_are_caught_before_the_run(tmp_path, monkeypatch):
    """The ground truth failure that took down three CLI entry points."""
    import ggssvt.eval.preflight as mod

    path = tmp_path / "ground_truth.csv"
    path.write_text(
        "plant_id,date,species_breed,total_fresh_weight_with_pot_g,"
        "pot_weight_g,net_weight_g,pot_weight_source,notes\n"
        "A001,2026-01-01,Eucalyptus,1000.0,400.0,600.0,measured,\n"
        "<<<<<<< HEAD\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "DATASET_DIR", tmp_path)
    found = mod._ground_truth()
    assert found[0].level == FATAL
    assert "fields against" in found[0].detail


def test_a_clean_ground_truth_reports_its_range(tmp_path, monkeypatch):
    import ggssvt.eval.preflight as mod

    path = tmp_path / "ground_truth.csv"
    path.write_text(
        "plant_id,date,species_breed,total_fresh_weight_with_pot_g,"
        "pot_weight_g,net_weight_g,pot_weight_source,notes\n"
        "A001,2026-01-01,Eucalyptus,1000.0,400.0,600.0,measured,\n"
        "A002,2026-01-01,Eucalyptus,2000.0,900.0,1100.0,estimated,\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "DATASET_DIR", tmp_path)
    found = mod._ground_truth()
    assert found[0].level == "ok"
    assert "0.60 to 1.10 kg" in found[0].detail
    assert "1 with measured" in found[0].detail


def test_report_exits_non_zero_only_on_fatal(capsys):
    ok = [Finding("a", "ok", "fine")]
    assert report(ok) == 0

    degraded = [Finding("a", DEGRADED, "missing", "install it")]
    assert report(degraded) == 0, "a degraded check must not block a run"

    fatal = [Finding("a", FATAL, "broken", "fix it")]
    assert report(fatal) == 1


def test_report_prints_the_fix_for_problems_only(capsys):
    report([Finding("bad", FATAL, "broken", "DO THIS"),
            Finding("good", "ok", "fine", "NOT THIS")])
    printed = capsys.readouterr().out
    assert "DO THIS" in printed
    assert "NOT THIS" not in printed
