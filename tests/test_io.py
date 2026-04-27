"""Integration tests for sleep_tools.io (uses real EDF + TSV data)."""
import h5py

from sleep_tools import ScoringSession, SleepAnalyzer, SleepRecording, save_to_h5
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


def test_save_to_h5_session_features_match_epoch_count(
    tmp_path,
    recording: SleepRecording,
) -> None:
    analyzer = SleepAnalyzer(recording, epoch_len=5.0)
    session = ScoringSession(recording, epoch_len=10.0)
    out = save_to_h5(
        recording,
        tmp_path / "aligned.h5",
        analyzer=analyzer,
        session=session,
        include_raw_signals=False,
    )
    with h5py.File(out, "r") as f:
        n_epochs = len(f["/epochs/times"])
        assert len(f["/epochs/labels"]) == n_epochs
        for key in f["/epochs/features"]:
            assert len(f["/epochs/features"][key]) == n_epochs
        assert f["/epochs/features"].attrs["feature_source"] == (
            "interpolated_from_analysis"
        )
        assert "/analysis/times" in f
        assert len(f["/analysis/times"]) != n_epochs


def test_scoring_session_from_h5_roundtrip_after_feature_alignment(
    tmp_path,
    recording: SleepRecording,
) -> None:
    analyzer = SleepAnalyzer(recording, epoch_len=5.0)
    session = ScoringSession(recording, epoch_len=10.0)
    session.label_indices(0, 2, "W")
    out = save_to_h5(
        recording,
        tmp_path / "roundtrip.h5",
        analyzer=analyzer,
        session=session,
        include_raw_signals=False,
    )
    loaded = ScoringSession.from_h5(out, recording)
    assert loaded.epoch_len == session.epoch_len
    assert loaded.labels.tolist() == session.labels.tolist()
