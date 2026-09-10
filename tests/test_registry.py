import pytest

from forecast_scarce.data import registry


@pytest.fixture(autouse=True)
def temp_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "data_registry.json")


def test_first_hash_is_recorded_then_verified():
    registry.check("m5", "raw", "abc123")
    registry.check("m5", "raw", "abc123")  # must not raise


def test_mismatch_raises():
    registry.check("m5", "raw", "abc123")
    with pytest.raises(ValueError, match="hash mismatch"):
        registry.check("m5", "raw", "def456")


def test_update_flag_overwrites():
    registry.check("m5", "raw", "abc123")
    registry.check("m5", "raw", "def456", update=True)
    registry.check("m5", "raw", "def456")


def test_kinds_are_tracked_separately():
    registry.check("m5", "raw", "abc123")
    registry.check("m5", "normalized", "zzz999")
    registry.check("m5", "raw", "abc123")


def test_file_and_dir_hashes_are_stable(tmp_path):
    (tmp_path / "a.parquet").write_bytes(b"hello")
    first = registry.sha256_dir(tmp_path)
    (tmp_path / "b.parquet").write_bytes(b"world")
    assert registry.sha256_dir(tmp_path) != first
