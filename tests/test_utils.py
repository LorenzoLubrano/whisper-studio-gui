import os

import pytest

import trascrivi_locale as ws


@pytest.mark.parametrize("seconds, expected", [
    (0, "00:00:00,000"),
    (1.5, "00:00:01,500"),
    (1.9996, "00:00:02,000"),      # prima diventava 00:00:01,1000 (SRT non valido)
    (59.9999, "00:01:00,000"),
    (3661.25, "01:01:01,250"),
    (-0.2, "00:00:00,000"),
])
def test_format_timestamp(seconds, expected):
    assert ws.format_timestamp(seconds) == expected


SEGS = [
    {"start": 0.0, "end": 2.08, "text": " Buongiorno a tutti."},
    {"start": 2.08, "end": 4.9996, "text": " Seconda frase. "},
]


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_write_srt(tmp_path):
    p = tmp_path / "a.srt"
    ws.write_srt(SEGS, str(p))
    assert read(p) == (
        "1\n00:00:00,000 --> 00:00:02,080\nBuongiorno a tutti.\n\n"
        "2\n00:00:02,080 --> 00:00:05,000\nSeconda frase.\n\n"
    )


def test_write_vtt(tmp_path):
    p = tmp_path / "a.vtt"
    ws.write_vtt(SEGS, str(p))
    assert read(p) == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.080\nBuongiorno a tutti.\n\n"
        "00:00:02.080 --> 00:00:05.000\nSeconda frase.\n\n"
    )


def test_write_txt_segmented(tmp_path):
    p = tmp_path / "a.segments.txt"
    ws.write_txt_segmented(SEGS, str(p))
    assert read(p) == (
        "[00:00:00,000–00:00:02,080] Buongiorno a tutti.\n"
        "[00:00:02,080–00:00:05,000] Seconda frase.\n"
    )


def test_output_paths_follow_selected_formats():
    cfg = {"save_txt": True, "save_srt": True, "save_vtt": False, "save_txt_seg": True}
    base = os.path.join("C:\\", "media", "lezione")
    assert ws.output_paths(base + ".mp4", cfg) == [
        base + ".txt", base + ".segments.txt", base + ".srt"]


@pytest.mark.parametrize("elapsed, processed, total, expected", [
    (10.0, 20.0, 60.0, 20.0),   # 0.5 s di calcolo per secondo di audio, restano 40 s di audio
    (10.0, 0.0, 60.0, None),    # nessun segmento ancora: stima impossibile
    (10.0, 70.0, 60.0, 0.0),    # mai negativa
])
def test_estimate_eta(elapsed, processed, total, expected):
    assert ws.estimate_eta(elapsed, processed, total) == expected


# ---------- scelta del dispositivo ----------

@pytest.mark.parametrize("requested, probe, expected", [
    ("auto", ("cuda", "GPU NVIDIA (CUDA)"), "cuda"),
    ("auto", ("cpu", "GPU NVIDIA trovata, ma mancano le librerie CUDA 12 (cuBLAS)"), "cpu"),
    ("cpu", ("cuda", "GPU NVIDIA (CUDA)"), "cpu"),
    ("cuda", ("cuda", "GPU NVIDIA (CUDA)"), "cuda"),
])
def test_resolve_device(requested, probe, expected):
    assert ws.resolve_device(requested, lambda: probe) == expected


def test_resolve_device_explicit_gpu_unavailable_raises_with_reason():
    reason = "GPU NVIDIA trovata, ma mancano le librerie CUDA 12 (cuBLAS)"
    with pytest.raises(RuntimeError, match="cuBLAS"):
        ws.resolve_device("cuda", lambda: ("cpu", reason))


def test_resolve_device_cpu_does_not_probe():
    def probe():
        raise AssertionError("la sonda GPU non deve girare se l'utente sceglie la CPU")
    assert ws.resolve_device("cpu", probe) == "cpu"


@pytest.mark.parametrize("error, expected", [
    (RuntimeError("Library cublas64_12.dll is not found or cannot be loaded"), True),
    (RuntimeError("CUDA failed with error out of memory"), True),
    (RuntimeError("cuDNN failed with status CUDNN_STATUS_NOT_INITIALIZED"), True),
    (RuntimeError("parallel_for failed: cudaErrorNoKernelImageForDevice"), True),
    (ValueError("Invalid data found when processing input"), False),
])
def test_is_cuda_error(error, expected):
    assert ws.is_cuda_error(error) is expected


def test_cuda_library_dirs_include_cuda_path_bin(tmp_path, monkeypatch):
    (tmp_path / "bin").mkdir()
    monkeypatch.setenv("CUDA_PATH", str(tmp_path))
    assert str(tmp_path / "bin") in ws.cuda_library_dirs()


def test_cuda_library_dirs_skip_missing_folders(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_PATH", str(tmp_path / "non-esiste"))
    assert all(os.path.isdir(d) for d in ws.cuda_library_dirs())


@pytest.mark.parametrize("device, requested, expected", [
    ("cpu", "float16", "auto"),   # float16 non esiste su CPU: CTranslate2 darebbe ValueError
    ("cpu", "int8", "int8"),
    ("cpu", "auto", "auto"),
    ("cuda", "float16", "float16"),
])
def test_compute_type_for_device(device, requested, expected):
    assert ws.compute_type_for(device, requested) == expected
