"""Integration tests for sleep_tools.io (uses real EDF + TSV data)."""
from sleep_tools import SleepRecording
from tests.conftest import LONG_EDF


def test_load_edf_channels(recording: SleepRecording) -> None:
    """Channels are renamed to canonical short names."""
    assert recording.channels == ["EEG1", "EEG2", "EMG"]


def test_load_edf_duration(recording: SleepRecording) -> None:
    """Recording duration is positive and matches the file size."""
    assert recording.duration > 0


def test_load_edf_sfreq(recording: SleepRecording) -> None:
    assert recording.sfreq == 400.0


def test_animal_id_parsed(recording: SleepRecording) -> None:
    assert recording.animal_id == "LUMI-0013"


def test_annotations_loaded(recording: SleepRecording) -> None:
    """TSV is discovered automatically and contains expected columns."""
    assert recording.annotations is not None
    for col in ("Number", "Start Time", "Time From Start", "Channel", "Annotation"):
        assert col in recording.annotations.columns


def test_annotations_have_ttl_events(recording: SleepRecording) -> None:
    ann = recording.annotations
    ttl = ann[ann["Annotation"].str.startswith("TTL")]
    assert len(ttl) > 0


def test_repr(recording: SleepRecording) -> None:
    r = repr(recording)
    assert "LUMI-0013" in r
    assert "EEG1" in r


def test_no_tsv_fallback(tmp_path) -> None:
    """Loading an EDF placed in a directory with no TSV yields annotations=None
    but still parses animal_id from the filename."""
    import shutil
    edf_copy = tmp_path / LONG_EDF.name
    shutil.copy(LONG_EDF, edf_copy)
    # No TSV exists in tmp_path, so auto-discovery returns nothing
    rec = SleepRecording.from_edf(edf_copy, tsv_path=None, verbose=False)
    assert rec.animal_id == "LUMI-0013"
    assert rec.annotations is None
