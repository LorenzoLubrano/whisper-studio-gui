import os
import time
import types

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
    (ValueError("Invalid data found when processing input: 'D:/video/barracuda.mp3'"), False),
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



def test_failed_write_keeps_the_previous_file(tmp_path):
    media = tmp_path / "a.mp3"
    media.write_bytes(b"")
    srt = tmp_path / "a.srt"
    srt.write_text("vecchio", encoding="utf-8")
    broken = [{"start": 0.0, "end": 1.0, "text": "ok"}, {"start": 1.0, "end": 2.0, "text": None}]
    cfg = {"save_txt": False, "save_srt": True, "save_vtt": False, "save_txt_seg": False}
    with pytest.raises(AttributeError):
        ws.write_outputs(str(media), broken, cfg)
    assert srt.read_text(encoding="utf-8") == "vecchio"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.mp3", "a.srt"]


# ---------- ricerca di cuBLAS: solo percorsi completi da cartelle note, mai la cartella corrente ----------

needs_windows = pytest.mark.skipif(os.name != "nt", reason="nomi delle DLL di Windows")


@pytest.fixture
def no_cuda_env(tmp_path, monkeypatch):
    empty = tmp_path / "programma"
    empty.mkdir()
    monkeypatch.setattr(ws, "APP_DIR", str(empty))
    monkeypatch.setenv("PATH", "")
    # anche la cartella del pulsante "Attiva GPU NVIDIA": sul PC di sviluppo puo' contenere cuBLAS vero
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata-vuota"))
    for k in list(os.environ):
        if k.upper().startswith("CUDA_PATH"):
            monkeypatch.delenv(k)
    return tmp_path


def fake_dll(folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "cublas64_12.dll").write_bytes(b"MZ")
    return str(folder / "cublas64_12.dll")


@needs_windows
def test_find_cublas_in_versioned_cuda_path(no_cuda_env, monkeypatch):
    dll = fake_dll(no_cuda_env / "CUDA" / "v12.8" / "bin")
    monkeypatch.setenv("CUDA_PATH_V12_8", str(no_cuda_env / "CUDA" / "v12.8"))
    assert ws.find_cublas() == dll


@needs_windows
def test_find_cublas_next_to_the_program(no_cuda_env, monkeypatch):
    dll = fake_dll(no_cuda_env / "programma")
    assert ws.find_cublas() == dll


@needs_windows
def test_find_cublas_in_path(no_cuda_env, monkeypatch):
    dll = fake_dll(no_cuda_env / "libs")
    monkeypatch.setenv("PATH", str(no_cuda_env / "libs"))
    assert ws.find_cublas() == dll


@needs_windows
def test_find_cublas_ignores_the_current_directory(no_cuda_env, monkeypatch):
    fake_dll(no_cuda_env / "download")
    monkeypatch.chdir(no_cuda_env / "download")
    assert ws.find_cublas() is None


def test_cuda_status_without_nvidia_gpu(monkeypatch):
    import ctranslate2
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 0)
    device, description = ws.cuda_status()
    assert device == "cpu"
    assert "nessuna GPU NVIDIA" in description


# ---------- modelli: prima dal disco, download solo se mancano ----------

class FakeWhisperModel:
    cached = set()
    calls = []

    def __init__(self, name, device="auto", compute_type="default", local_files_only=False, **kw):
        from huggingface_hub.utils import LocalEntryNotFoundError
        FakeWhisperModel.calls.append((name, local_files_only))
        if local_files_only and name not in FakeWhisperModel.cached:
            raise LocalEntryNotFoundError("non in cache")


def fake_snapshot_download(repo_id, allow_patterns=None, tqdm_class=None, **kw):
    """Usa tqdm_class come huggingface_hub 1.x con un repository Xet (misurato su faster-whisper-base):
    i file si registrano uno alla volta (prima i piccoli), i byte dalla rete avanzano con regolarita'
    mentre i byte scritti ("Reconstructing") arrivano tutti alla fine."""
    fake_snapshot_download.calls.append((repo_id, list(allow_patterns or [])))
    transfer = tqdm_class(desc="Downloading bytes", total=0, unit="B", unit_scale=True)
    rebuild = tqdm_class(desc="Reconstructing (incomplete total...)", total=0, unit="B", unit_scale=True)
    rebuild.total = transfer.total = 10          # prima solo i file piccoli
    transfer.update(10)
    rebuild.total += 190                         # poi model.bin
    transfer.total += 190
    for _ in range(4):
        transfer.update(45)
    rebuild.update(200)
    transfer.close()
    rebuild.close()
    return "C:/cache/models--Systran--faster-whisper-" + repo_id.split("-")[-1]


class FakeHfApi:
    fail = False

    def model_info(self, repo_id, files_metadata=False):
        if FakeHfApi.fail:
            raise OSError("rete assente")
        sizes = {".gitattributes": 1, "README.md": 2, "config.json": 1, "model.bin": 190, "tokenizer.json": 6,
                 "vocabulary.txt": 3}
        return types.SimpleNamespace(siblings=[types.SimpleNamespace(rfilename=k, size=v) for k, v in sizes.items()])


class TickingClock:
    """Ogni lettura dell'ora avanza di un secondo: nessun aggiornamento viene scartato dal limite di frequenza."""
    def __init__(self):
        self.now = 1000.0

    def time(self):
        self.now += 1.0
        return self.now


@pytest.fixture
def fake_whisper(monkeypatch):
    import faster_whisper
    import huggingface_hub
    FakeWhisperModel.cached = {"small"}
    FakeWhisperModel.calls = []
    fake_snapshot_download.calls = []
    FakeHfApi.fail = False
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    monkeypatch.setattr(ws, "time", TickingClock())
    return FakeWhisperModel


def test_cached_model_is_loaded_without_network(fake_whisper):
    progress = []
    ws.load_model("small", "cpu", "auto", on_download=lambda d, t: progress.append((d, t)))
    assert fake_whisper.calls == [("small", True)]
    assert progress == []
    assert fake_snapshot_download.calls == []


def test_missing_model_is_downloaded_with_progress(fake_whisper):
    progress = []
    ws.load_model("medium", "cpu", "auto", on_download=lambda d, t: progress.append((d, t)))
    assert fake_snapshot_download.calls[0][0] == "Systran/faster-whisper-medium"
    assert "model.bin" in fake_snapshot_download.calls[0][1]
    # il modello viene poi aperto dalla cartella scaricata, senza un secondo download
    assert fake_whisper.calls == [("medium", True), ("C:/cache/models--Systran--faster-whisper-medium", False)]
    assert progress[0] == (0, 0)                    # "download iniziato"
    assert progress[-1] == (200, 200)
    during = progress[1:-1]
    assert all(t == 200 for _, t in during)         # totale vero dall'inizio (config+model.bin+tokenizer+vocabulary)
    assert all(d < t for d, t in during)            # niente 100% prima della fine
    assert (100, 200) in during                     # avanza con i byte dalla rete, anche se quelli scritti sono fermi
    assert [d for d, _ in progress] == sorted(d for d, _ in progress)


def test_download_progress_without_size_information(fake_whisper):
    FakeHfApi.fail = True
    progress = []
    ws.load_model("medium", "cpu", "auto", on_download=lambda d, t: progress.append((d, t)))
    assert progress[0] == (0, 0) and progress[-1][0] == progress[-1][1] > 0


@needs_windows
def test_cuda_status_tries_the_next_cublas_if_one_does_not_load(no_cuda_env, monkeypatch):
    import ctranslate2
    first = fake_dll(no_cuda_env / "programma")          # es. copiata senza cublasLt accanto
    second = fake_dll(no_cuda_env / "CUDA" / "v12.9" / "bin")
    monkeypatch.setenv("CUDA_PATH_V12_9", str(no_cuda_env / "CUDA" / "v12.9"))
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 1)
    tried = []
    monkeypatch.setattr(ws, "_load_cublas", lambda p: tried.append(p) or p == second)
    assert ws.cuda_status()[0] == "cuda"
    assert tried == [first, second]


def test_temporary_file_never_touches_a_user_file_with_the_same_name(tmp_path):
    media = tmp_path / "a.mp3"
    media.write_bytes(b"")
    mine = tmp_path / "a.srt.tmp"
    mine.write_text("mio", encoding="utf-8")
    segs = [{"start": 0.0, "end": 1.0, "text": "ok"}]
    cfg = {"save_txt": False, "save_srt": True, "save_vtt": False, "save_txt_seg": False}
    ws.write_outputs(str(media), segs, cfg)
    assert mine.read_text(encoding="utf-8") == "mio"
    assert (tmp_path / "a.srt").exists()


def test_download_can_be_cancelled(fake_whisper):
    with pytest.raises(ws.DownloadCancelled):
        ws.load_model("medium", "cpu", "auto", on_download=lambda d, t: None, should_stop=lambda: True)
    assert fake_whisper.calls == [("medium", True)]  # il modello non viene aperto


# ---------- pulsante "Attiva GPU NVIDIA": download verificato delle librerie cuBLAS ----------

import hashlib  # noqa: E402
import io  # noqa: E402
import zipfile  # noqa: E402


def make_fake_wheel(extra=None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("nvidia/cublas/bin/cublas64_12.dll", b"MZ cublas")
        z.writestr("nvidia/cublas/bin/cublasLt64_12.dll", b"MZ cublasLt")
        z.writestr("nvidia/cublas/bin/nvblas64_12.dll", b"MZ nvblas")
        for name, data in (extra or {}).items():
            z.writestr(name, data)
    return buf.getvalue()


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture
def fake_pypi(tmp_path, monkeypatch):
    """Finto download da PyPI: serve un finto pacchetto con l'impronta attesa, in una LOCALAPPDATA vuota."""
    import urllib.request
    wheel = make_fake_wheel()
    state = {"body": wheel, "error": None, "requests": []}

    def urlopen(req, timeout=None):
        state["requests"].append(req.full_url)
        if state["error"]:
            raise state["error"]
        return FakeResponse(state["body"])

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(ws, "CUBLAS_WHEEL_SHA256", hashlib.sha256(wheel).hexdigest())
    monkeypatch.setattr(ws, "CUBLAS_WHEEL_SIZE", len(wheel))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    return state


def make_old(folder, seconds=600):
    old = time.time() - seconds
    for p in [folder, *folder.rglob("*")]:
        os.utime(p, (old, old))


def installed_files(tmp_path):
    root = tmp_path / "appdata"
    return sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file())


def test_gpu_libraries_are_downloaded_verified_and_installed(fake_pypi, tmp_path):
    progress = []
    dest = ws.install_cuda_libraries(report=lambda d, t: progress.append((d, t)))
    assert fake_pypi["requests"] == [ws.CUBLAS_WHEEL_URL]
    assert ws.CUBLAS_WHEEL_URL.startswith("https://files.pythonhosted.org/")
    assert dest == str(tmp_path / "appdata" / "WhisperStudio" / "cuda12")
    # solo le due DLL, niente pacchetto scaricato ne' cartelle provvisorie
    assert installed_files(tmp_path) == ["WhisperStudio/cuda12/cublas64_12.dll",
                                         "WhisperStudio/cuda12/cublasLt64_12.dll"]
    assert (tmp_path / "appdata/WhisperStudio/cuda12/cublasLt64_12.dll").read_bytes() == b"MZ cublasLt"
    assert progress[-1] == (ws.CUBLAS_WHEEL_SIZE, ws.CUBLAS_WHEEL_SIZE)


def test_gpu_libraries_with_the_wrong_fingerprint_are_not_installed(fake_pypi, tmp_path):
    body = bytearray(fake_pypi["body"])
    body[-1] ^= 0xFF                                       # stessa dimensione, un byte diverso
    fake_pypi["body"] = bytes(body)
    with pytest.raises(ws.GpuSetupError, match="non corrisponde"):
        ws.install_cuda_libraries()
    assert installed_files(tmp_path) == []


def test_gpu_libraries_download_can_be_cancelled(fake_pypi, tmp_path):
    with pytest.raises(ws.DownloadCancelled):
        ws.install_cuda_libraries(should_stop=lambda: True)
    assert installed_files(tmp_path) == []


def test_gpu_libraries_network_error_is_explained(fake_pypi, tmp_path):
    import urllib.error
    fake_pypi["error"] = urllib.error.URLError("getaddrinfo failed")
    with pytest.raises(ws.GpuSetupError, match="connessione"):
        ws.install_cuda_libraries()
    assert installed_files(tmp_path) == []


def test_installed_gpu_libraries_are_found(no_cuda_env, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(no_cuda_env / "appdata"))
    dll = fake_dll(no_cuda_env / "appdata" / "WhisperStudio" / "cuda12")
    assert ws.find_cublas() == dll


def test_leftovers_of_an_interrupted_install_are_cleaned(fake_pypi, tmp_path):
    leftover = tmp_path / "appdata" / "WhisperStudio" / "cuda12-vecchio"
    leftover.mkdir(parents=True)
    (leftover / "cublas.whl").write_bytes(b"x" * 1000)  # es. finestra chiusa durante il download
    make_old(leftover)
    ws.install_cuda_libraries()
    assert installed_files(tmp_path) == ["WhisperStudio/cuda12/cublas64_12.dll",
                                         "WhisperStudio/cuda12/cublasLt64_12.dll"]


@pytest.mark.parametrize("done, total, expected", [
    (0, 553162896, "0% (0 di 527 MB)"),
    (276581448, 553162896, "50% (263 di 527 MB)"),
    (551000000, 553162896, "99% (525 di 527 MB)"),   # mai 100% prima della fine
    (553162896, 553162896, "100% (527 di 527 MB)"),
])
def test_download_progress_text(done, total, expected):
    assert ws.progress_text(done, total) == expected


# ---------- dalla revisione del pulsante GPU ----------

def test_download_bigger_than_expected_is_stopped(fake_pypi, tmp_path):
    fake_pypi["body"] = fake_pypi["body"] + b"x" * 5000   # il server manda piu' del previsto
    with pytest.raises(ws.GpuSetupError, match="dimensione"):
        ws.install_cuda_libraries()
    assert installed_files(tmp_path) == []


def test_truncated_download_is_reported_as_incomplete(fake_pypi, tmp_path):
    fake_pypi["body"] = fake_pypi["body"][:100]            # connessione chiusa a meta'
    with pytest.raises(ws.GpuSetupError, match="incomplet"):
        ws.install_cuda_libraries()
    assert installed_files(tmp_path) == []


def test_package_no_longer_available_is_explained(fake_pypi, tmp_path):
    import urllib.error
    fake_pypi["error"] = urllib.error.HTTPError(ws.CUBLAS_WHEEL_URL, 404, "Not Found", {}, None)
    with pytest.raises(ws.GpuSetupError, match="non è più disponibile"):
        ws.install_cuda_libraries()


def test_protocol_errors_are_explained_in_italian(fake_pypi, tmp_path, monkeypatch):
    import http.client
    import urllib.request

    class Broken(FakeResponse):
        def read(self, n=-1):
            raise http.client.IncompleteRead(b"", 10)

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: Broken(b""))
    with pytest.raises(ws.GpuSetupError, match="connessione"):
        ws.install_cuda_libraries()
    assert installed_files(tmp_path) == []


def test_not_enough_disk_space_is_checked_first(fake_pypi, tmp_path, monkeypatch):
    import collections
    import shutil
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage(10**12, 10**12 - 10**6, 10**6))
    with pytest.raises(ws.GpuSetupError, match="spazio"):
        ws.install_cuda_libraries()
    assert fake_pypi["requests"] == []                    # nessun download inutile


def test_cancel_right_before_the_final_move_installs_nothing(fake_pypi, tmp_path):
    # chiamate allo stop: 2 nel download, 1 dopo la verifica, 1 per ciascuna delle 2 DLL, poi quella finale
    asked = []
    with pytest.raises(ws.DownloadCancelled):
        ws.install_cuda_libraries(should_stop=lambda: asked.append(1) or len(asked) >= 6)
    assert installed_files(tmp_path) == []


def test_cancel_right_after_verification_installs_nothing(fake_pypi, tmp_path):
    # il finto pacchetto arriva in un solo blocco: il download chiede lo stop 2 volte (prima del blocco e
    # prima della lettura finale vuota); "Interrompi" premuto dopo deve fermare verifica ed estrazione
    asked = []
    with pytest.raises(ws.DownloadCancelled):
        ws.install_cuda_libraries(should_stop=lambda: asked.append(1) or len(asked) >= 3)
    assert installed_files(tmp_path) == []


def test_existing_installation_is_replaced_whole(fake_pypi, tmp_path):
    old = tmp_path / "appdata" / "WhisperStudio" / "cuda12"
    old.mkdir(parents=True)
    (old / "cublas64_12.dll").write_bytes(b"vecchia")
    (old / "avanzo.dll").write_bytes(b"vecchio")
    ws.install_cuda_libraries()
    assert installed_files(tmp_path) == ["WhisperStudio/cuda12/cublas64_12.dll",
                                         "WhisperStudio/cuda12/cublasLt64_12.dll"]


def test_package_without_the_expected_libraries_installs_nothing(fake_pypi, tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("nvidia/cublas/bin/cublas64_12.dll", b"MZ solo una")
    body = buf.getvalue()
    fake_pypi["body"] = body
    monkeypatch.setattr(ws, "CUBLAS_WHEEL_SHA256", hashlib.sha256(body).hexdigest())
    monkeypatch.setattr(ws, "CUBLAS_WHEEL_SIZE", len(body))
    with pytest.raises(ws.GpuSetupError):
        ws.install_cuda_libraries()
    assert installed_files(tmp_path) == []


def test_filesystem_errors_are_explained(fake_pypi, tmp_path):
    blocker = tmp_path / "appdata" / "WhisperStudio" / "cuda12"
    blocker.parent.mkdir(parents=True)
    blocker.write_text("un file al posto della cartella", encoding="utf-8")
    with pytest.raises(ws.GpuSetupError):
        ws.install_cuda_libraries()


def test_final_move_is_retried_when_the_antivirus_holds_the_files(fake_pypi, tmp_path, monkeypatch):
    real_replace = ws.os.replace
    attempts = []

    def flaky(src, dst):
        attempts.append(dst)
        if len(attempts) < 3:
            raise PermissionError(5, "Accesso negato")
        return real_replace(src, dst)

    monkeypatch.setattr(ws.os, "replace", flaky)
    monkeypatch.setattr(ws.time, "sleep", lambda s: None)
    ws.install_cuda_libraries()
    assert len(attempts) == 3
    assert installed_files(tmp_path)[0] == "WhisperStudio/cuda12/cublas64_12.dll"


def test_relative_data_folder_is_refused(fake_pypi, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", "relativo")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ws.GpuSetupError):
        ws.install_cuda_libraries()
    assert not (tmp_path / "relativo").exists()


def test_leftovers_can_be_cleaned_without_installing(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    leftover = tmp_path / "WhisperStudio" / "cuda12-abc"
    leftover.mkdir(parents=True)
    (leftover / "cublas.whl").write_bytes(b"x")
    make_old(leftover)
    fresh = tmp_path / "WhisperStudio" / "cuda12-altra-finestra"   # download in corso in un'altra finestra
    fresh.mkdir()
    (fresh / "cublas.whl").write_bytes(b"y")
    (tmp_path / "WhisperStudio" / "cuda12").mkdir()
    ws.clean_gpu_install_leftovers()
    assert sorted(p.name for p in (tmp_path / "WhisperStudio").iterdir()) == ["cuda12", "cuda12-altra-finestra"]
