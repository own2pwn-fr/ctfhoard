"""Tests for the Hugging Face publisher — fully offline, no network access."""

from __future__ import annotations

from pathlib import Path

import pytest

from ctfhoard import hf


def _make_corpus(root: Path) -> None:
    """Build a small corpus tree with mixed extensions and known sizes."""
    chal = root / "hackropole" / "evt" / "2026" / "baby__deadbeef"
    (chal / "writeups").mkdir(parents=True, exist_ok=True)
    (chal / "chal.py").write_text("print('pwn')\n")  # 13 bytes, .py
    (chal / "firmware.bin").write_bytes(b"\x00\x01\x02\x03")  # 4 bytes, .bin
    (chal / "writeups" / "writeup_00.html").write_text("<html></html>")  # 13 bytes, .html
    (chal / "notes.py").write_text("x = 1\n")  # 6 bytes, .py


def test_corpus_stats_counts_files_bytes_and_extensions(tmp_path: Path) -> None:
    _make_corpus(tmp_path)
    stats = hf.corpus_stats(tmp_path)

    assert stats["files"] == 4
    assert stats["total_bytes"] == 13 + 4 + 13 + 6
    ext_tally = dict(stats["by_extension_top"])
    assert ext_tally[".py"] == 2
    assert ext_tally[".bin"] == 1
    assert ext_tally[".html"] == 1


def test_corpus_stats_empty_dir_is_zeroed(tmp_path: Path) -> None:
    stats = hf.corpus_stats(tmp_path / "does-not-exist")
    assert stats == {"files": 0, "total_bytes": 0, "by_extension_top": []}


def test_publish_corpus_dry_run_reports_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_corpus(tmp_path)

    # Guard: dry-run must never construct an HfApi / hit the network.
    def _boom(*_args, **_kwargs):  # pragma: no cover - only fires on regression
        raise AssertionError("dry_run must not touch huggingface_hub")

    monkeypatch.setattr("huggingface_hub.HfApi", _boom, raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    report = hf.publish_corpus(tmp_path, dry_run=True)

    assert "dry-run" in report
    assert "4 files" in report
    assert str(13 + 4 + 13 + 6) in report


def test_publish_corpus_empty_dir_does_not_crash(tmp_path: Path) -> None:
    report = hf.publish_corpus(tmp_path / "empty", dry_run=True)
    assert "empty" in report


def test_publish_corpus_missing_token_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_corpus(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(hf.HFTokenError):
        hf.publish_corpus(tmp_path, dry_run=False)


def test_publish_catalog_dry_run_reports_shards(tmp_path: Path) -> None:
    (tmp_path / "hackropole").mkdir()
    (tmp_path / "hackropole" / "challenges.jsonl").write_text('{"id": "x"}\n')
    report = hf.publish_catalog(tmp_path, dry_run=True)
    assert "dry-run" in report
    assert "1 catalog shard" in report
