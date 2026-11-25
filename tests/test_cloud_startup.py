import sys

import pytest

from catering_visual.__main__ import main as gallery_main
from pipeline import main as video_main
from vfs import runtime


@pytest.mark.parametrize("platform, directories, allowed", [
    ("linux", ("library-files", "scratch", "shared"), True),
    ("linux", ("scratch", "shared"), False),
    ("win32", ("library-files", "scratch", "shared"), False),
    ("darwin", ("library-files", "scratch", "shared"), False),
])
def test_managed_workspace_recognition(monkeypatch, tmp_path, platform, directories, allowed):
    for name in directories:
        (tmp_path / name).mkdir()
    with monkeypatch.context() as env:
        env.setattr(runtime, "CLOUD_WORKSPACE", tmp_path)
        env.setattr(sys, "platform", platform)
        assert runtime.is_codex_cloud() is allowed


@pytest.mark.parametrize("entry, arguments", [
    (video_main, ["--prepare", "--input", "missing.mp4", "--output"]),
    (video_main, ["--input", "missing.mp4", "--resume", "--output"]),
    (video_main, ["--qc-only", "--output"]),
    (gallery_main, ["analyze", "missing-photos", "--out"]),
    (gallery_main, ["render", "missing.json", "--out"]),
])
def test_unknown_workspace_stops_before_work(monkeypatch, tmp_path, capsys, entry, arguments):
    monkeypatch.setattr(runtime, "CLOUD_WORKSPACE", tmp_path / "workspace")
    output = tmp_path / "blocked-output"
    assert entry([*arguments, str(output)]) == 1
    assert "Codex Cloud workspace could not be verified" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("entry", [video_main, gallery_main])
def test_unknown_workspace_reports_guard_on_default_startup(monkeypatch, tmp_path, capsys, entry):
    monkeypatch.setattr(runtime, "CLOUD_WORKSPACE", tmp_path / "workspace")
    assert entry([]) == 1
    assert "Codex Cloud workspace could not be verified" in capsys.readouterr().err
