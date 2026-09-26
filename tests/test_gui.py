"""Test della finestra vera (Tk) con un modello finto al posto di Whisper."""
import os
import threading
import tkinter as tk

import pytest

from conftest import FakeModel, Loader, is_idle, make_media, make_segment, pump_for, pump_until

TWO_SEGS = [make_segment(0.0, 2.0, " Prima frase.", 1), make_segment(2.0, 4.0, " Seconda frase.", 2)]


def start_with(app, files):
    app.files_selected = list(files)
    app._refresh_listbox()
    app.btn_start.invoke()


def test_transcription_error_is_shown_and_ui_recovers(make_app, dialogs, media):
    err = ValueError("Invalid data found when processing input")
    app = make_app(Loader({"cpu": FakeModel("cpu", error=err)}))
    start_with(app, [media])
    pump_until(app, lambda: is_idle(app), timeout=15)
    errors = [c for c in dialogs.calls if c[0] == "showerror"]
    assert len(errors) == 1
    assert "Invalid data found when processing input" in errors[0][2]
    assert "intervista.mp3" in errors[0][2]


def test_model_load_error_is_shown(make_app, dialogs, media):
    app = make_app(Loader({"cpu": OSError("Impossibile scaricare il modello: rete assente")}))
    start_with(app, [media])
    pump_until(app, lambda: is_idle(app), timeout=15)
    errors = [c for c in dialogs.calls if c[0] == "showerror"]
    assert len(errors) == 1 and "rete assente" in errors[0][2]


def test_auto_falls_back_to_cpu_when_gpu_fails_during_transcription(make_app, dialogs, media):
    gpu = FakeModel("cuda", error=RuntimeError("Library cublas64_12.dll is not found or cannot be loaded"))
    cpu = FakeModel("cpu", segments=TWO_SEGS, duration=4.0)
    loader = Loader({"cuda": gpu, "cpu": cpu})
    app = make_app(loader, probe=("cuda", "GPU NVIDIA (CUDA)"))
    start_with(app, [media])
    pump_until(app, lambda: is_idle(app), timeout=15)
    assert [c[1] for c in loader.calls] == ["cuda", "cpu"]
    assert dialogs.kinds() == ["showinfo"]
    assert "CPU" in dialogs.calls[-1][2]  # l'utente deve sapere che la GPU non ha funzionato
    with open(os.path.splitext(media)[0] + ".txt", encoding="utf-8") as f:
        assert f.read() == "Prima frase. Seconda frase.\n"


def test_auto_uses_cpu_when_gpu_libraries_are_missing(make_app, dialogs, media):
    loader = Loader({"cpu": FakeModel("cpu", segments=TWO_SEGS, duration=4.0)})
    app = make_app(loader, probe=("cpu", "GPU NVIDIA trovata, ma mancano le librerie CUDA 12 (cuBLAS)"))
    start_with(app, [media])
    pump_until(app, lambda: is_idle(app), timeout=15)
    assert [c[1] for c in loader.calls] == ["cpu"]
    assert dialogs.kinds() == ["showinfo"]


def test_explicit_gpu_without_cuda_libraries_is_an_error(make_app, dialogs, media):
    loader = Loader({"cpu": FakeModel("cpu", segments=TWO_SEGS)})
    app = make_app(loader, probe=("cpu", "GPU NVIDIA trovata, ma mancano le librerie CUDA 12 (cuBLAS)"))
    app.device_choice.set("GPU (CUDA)")
    start_with(app, [media])
    pump_until(app, lambda: is_idle(app), timeout=15)
    assert loader.calls == []
    errors = [c for c in dialogs.calls if c[0] == "showerror"]
    assert len(errors) == 1 and "cuBLAS" in errors[0][2]


def test_progress_bar_is_full_and_still_after_success(make_app, dialogs, media):
    app = make_app(Loader({"cpu": FakeModel("cpu", segments=TWO_SEGS, duration=4.0)}))
    start_with(app, [media])
    pump_until(app, lambda: is_idle(app), timeout=15)
    first = float(app.progress.cget("value"))
    pump_for(app, 0.4)
    assert first == 100.0
    assert float(app.progress.cget("value")) == 100.0


def test_cancel_is_reported_as_cancellation_not_error(make_app, dialogs, media):
    gate = threading.Event()
    model = FakeModel("cpu", segments=TWO_SEGS, duration=4.0, gate=gate)
    app = make_app(Loader({"cpu": model}))
    start_with(app, [media])
    pump_until(app, lambda: str(app.btn_stop.cget("state")) == "normal" and model.calls, timeout=15)
    app.btn_stop.invoke()
    gate.set()
    pump_until(app, lambda: is_idle(app), timeout=15)
    assert "showerror" not in dialogs.kinds()
    infos = [c for c in dialogs.calls if c[0] == "showinfo"]
    assert len(infos) == 1 and "annullat" in infos[0][2].lower()
    assert not os.path.exists(os.path.splitext(media)[0] + ".txt")


def test_file_controls_are_locked_while_running(make_app, dialogs, media):
    gate = threading.Event()
    model = FakeModel("cpu", segments=TWO_SEGS, duration=4.0, gate=gate)
    app = make_app(Loader({"cpu": model}))
    start_with(app, [media])
    pump_until(app, lambda: model.calls, timeout=15)
    locked = [str(b.cget("state")) for b in (app.btn_add, app.btn_remove, app.btn_clear)]
    gate.set()
    pump_until(app, lambda: is_idle(app), timeout=15)
    unlocked = [str(b.cget("state")) for b in (app.btn_add, app.btn_remove, app.btn_clear)]
    assert locked == ["disabled"] * 3
    assert unlocked == ["normal"] * 3


def test_existing_outputs_need_confirmation(make_app, dialogs, media):
    txt = os.path.splitext(media)[0] + ".txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.write("appunti da non perdere\n")
    loader = Loader({"cpu": FakeModel("cpu", segments=TWO_SEGS, duration=4.0)})
    app = make_app(loader)
    dialogs.answer = False
    start_with(app, [media])
    pump_for(app, 0.3)
    assert dialogs.kinds() == ["askyesno"]
    assert loader.calls == []
    with open(txt, encoding="utf-8") as f:
        assert f.read() == "appunti da non perdere\n"

    dialogs.answer = True
    app.btn_start.invoke()
    pump_until(app, lambda: is_idle(app) and len(loader.calls) == 1, timeout=15)
    with open(txt, encoding="utf-8") as f:
        assert f.read() == "Prima frase. Seconda frase.\n"


def test_no_output_format_selected_warns_and_does_not_start(make_app, dialogs, media):
    loader = Loader({"cpu": FakeModel("cpu", segments=TWO_SEGS)})
    app = make_app(loader)
    for var in (app.save_txt, app.save_srt, app.save_vtt, app.save_txt_seg):
        var.set(False)
    start_with(app, [media])
    pump_for(app, 0.3)
    assert dialogs.kinds() == ["showwarning"]
    assert loader.calls == []


def test_fallback_to_cpu_keeps_a_precision_the_cpu_supports(make_app, dialogs, media):
    gpu = FakeModel("cuda", error=RuntimeError("Library cublas64_12.dll is not found or cannot be loaded"))
    loader = Loader({"cuda": gpu, "cpu": FakeModel("cpu", segments=TWO_SEGS, duration=4.0)})
    app = make_app(loader, probe=("cuda", "GPU NVIDIA (CUDA)"))
    app.model_name.set("small")
    app.compute_type.set("float16")
    start_with(app, [media])
    pump_until(app, lambda: is_idle(app), timeout=15)
    assert loader.calls == [("small", "cuda", "float16"), ("small", "cpu", "auto")]
    assert dialogs.kinds() == ["showinfo"]


def test_closing_the_window_cancels_the_event_poll(make_app):
    app = make_app(Loader({}))
    pump_for(app, 0.3)
    app.destroy()
    scripts = [str(app.tk.call("after", "info", i)) for i in app.tk.call("after", "info")]
    assert not any("_poll_events" in s for s in scripts)



@pytest.mark.parametrize("preset, beam", [("Fast", 1), ("Balanced", 3), ("Accurate", 5)])
def test_presets_decode_deterministically(make_app, dialogs, media, preset, beam):
    # in faster-whisper una temperatura fissa > 0 vuol dire parole scelte a caso e beam_size ignorato
    model = FakeModel("cpu", segments=TWO_SEGS, duration=4.0)
    app = make_app(Loader({"cpu": model}))
    app.speed_preset.set(preset)
    start_with(app, [media])
    pump_until(app, lambda: is_idle(app), timeout=15)
    kwargs = model.calls[0][1]
    temps = kwargs.get("temperature", [0.0])
    first = temps[0] if isinstance(temps, (list, tuple)) else temps
    assert kwargs["beam_size"] == beam
    assert first == 0.0


def test_event_poll_survives_a_failing_handler(make_app):
    app = make_app(Loader({}))
    app.report_callback_exception = lambda *exc: None  # l'errore viene segnalato, non deve fermare la coda
    app._post("_set_status", 123, "argomento in piu'")  # TypeError dentro il gestore
    app._post("_set_status", "evento successivo")
    pump_until(app, lambda: app.lbl_status.cget("text") == "evento successivo", timeout=3)
    app._post("_set_status", "ancora vivo")
    pump_until(app, lambda: app.lbl_status.cget("text") == "ancora vivo", timeout=3)


CUBLAS_MISSING = RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")


def test_batch_continues_after_a_broken_file_and_reports_it(make_app, dialogs, tmp_path):
    files = [make_media(tmp_path, n) for n in ("a.mp3", "b.mp3", "c.mp3")]
    model = FakeModel("cpu", segments=TWO_SEGS, duration=4.0,
                      error_for={"b.mp3": ValueError("Invalid data found when processing input")})
    app = make_app(Loader({"cpu": model}))
    start_with(app, files)
    pump_until(app, lambda: is_idle(app), timeout=15)
    assert (tmp_path / "a.txt").exists() and (tmp_path / "c.txt").exists()
    assert not (tmp_path / "b.txt").exists()
    assert dialogs.kinds() == ["showwarning"]
    assert "b.mp3" in dialogs.calls[0][2] and "Invalid data found" in dialogs.calls[0][2]


def test_missing_file_is_reported_while_the_others_are_done(make_app, dialogs, tmp_path):
    ok = make_media(tmp_path, "ok.mp3")
    gone = str(tmp_path / "sparito.mp3")
    app = make_app(Loader({"cpu": FakeModel("cpu", segments=TWO_SEGS, duration=4.0)}))
    start_with(app, [ok, gone])
    pump_until(app, lambda: is_idle(app), timeout=15)
    assert (tmp_path / "ok.txt").exists()
    assert dialogs.kinds() == ["showwarning"]
    assert "sparito.mp3" in dialogs.calls[0][2]


def test_files_that_would_write_the_same_outputs_are_refused(make_app, dialogs, tmp_path):
    files = [make_media(tmp_path, "lezione.mp4"), make_media(tmp_path, "lezione.m4a")]
    loader = Loader({"cpu": FakeModel("cpu", segments=TWO_SEGS)})
    app = make_app(loader)
    start_with(app, files)
    pump_for(app, 0.3)
    assert dialogs.kinds() == ["showwarning"]
    assert "lezione.mp4" in dialogs.calls[0][2] and "lezione.m4a" in dialogs.calls[0][2]
    assert loader.calls == []


def test_after_a_gpu_failure_auto_stays_on_cpu_for_the_session(make_app, dialogs, media):
    gpu = FakeModel("cuda", error=CUBLAS_MISSING)
    loader = Loader({"cuda": gpu, "cpu": FakeModel("cpu", segments=TWO_SEGS, duration=4.0)})
    app = make_app(loader, probe=("cuda", "GPU NVIDIA (CUDA)"))
    start_with(app, [media])
    pump_until(app, lambda: is_idle(app), timeout=15)
    app.btn_start.invoke()  # i file di output ora esistono: la conferma risponde si'
    pump_until(app, lambda: is_idle(app) and len(loader.calls) == 3, timeout=15)
    assert [c[1] for c in loader.calls] == ["cuda", "cpu", "cpu"]


def test_after_a_gpu_failure_explicit_gpu_asks_to_restart(make_app, dialogs, media):
    gpu = FakeModel("cuda", error=CUBLAS_MISSING)
    loader = Loader({"cuda": gpu, "cpu": FakeModel("cpu", segments=TWO_SEGS, duration=4.0)})
    app = make_app(loader, probe=("cuda", "GPU NVIDIA (CUDA)"))
    start_with(app, [media])
    pump_until(app, lambda: is_idle(app), timeout=15)
    app.device_choice.set("GPU (CUDA)")
    app.btn_start.invoke()
    pump_until(app, lambda: is_idle(app) and dialogs.kinds()[-1] == "showerror", timeout=15)
    assert "riavvia" in dialogs.calls[-1][2].lower()
    assert [c[1] for c in loader.calls] == ["cuda", "cpu"]


def test_auto_falls_back_to_cpu_when_the_gpu_model_cannot_load(make_app, dialogs, media):
    loader = Loader({"cuda": RuntimeError("CUDA failed with error out of memory"),
                     "cpu": FakeModel("cpu", segments=TWO_SEGS, duration=4.0)})
    app = make_app(loader, probe=("cuda", "GPU NVIDIA (CUDA)"))
    start_with(app, [media])
    pump_until(app, lambda: is_idle(app), timeout=15)
    assert [c[1] for c in loader.calls] == ["cuda", "cpu"]
    assert dialogs.kinds() == ["showinfo"]
    assert "CPU" in dialogs.calls[0][2]
    assert os.path.exists(os.path.splitext(media)[0] + ".txt")


def test_cancel_before_the_gpu_fallback_does_not_load_the_cpu_model(make_app, dialogs, media):
    gate = threading.Event()
    gpu = FakeModel("cuda", error=CUBLAS_MISSING, gate=gate)
    loader = Loader({"cuda": gpu, "cpu": FakeModel("cpu", segments=TWO_SEGS)})
    app = make_app(loader, probe=("cuda", "GPU NVIDIA (CUDA)"))
    start_with(app, [media])
    pump_until(app, lambda: gpu.calls, timeout=15)
    app.btn_stop.invoke()
    gate.set()
    pump_until(app, lambda: is_idle(app), timeout=15)
    assert [c[1] for c in loader.calls] == ["cuda"]
    assert dialogs.kinds() == ["showinfo"] and "annullat" in dialogs.calls[0][2].lower()


def test_unexpected_error_in_the_window_is_shown(make_app, dialogs):
    app = make_app(Loader({}))
    app._post("_set_status", 1, 2, 3)  # TypeError dentro il gestore
    pump_until(app, lambda: "showerror" in dialogs.kinds(), timeout=3)


# il thread di lavoro termina con SystemExit apposta: pytest lo segnalerebbe come avviso
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_worker_crash_still_ends_the_job(make_app, dialogs, media):
    app = make_app(Loader({"cpu": SystemExit()}))
    start_with(app, [media])
    pump_until(app, lambda: is_idle(app), timeout=15)
    assert dialogs.kinds() == ["showerror"]


def test_closing_while_running_asks_and_stops(make_app, dialogs, media):
    gate = threading.Event()
    model = FakeModel("cpu", segments=TWO_SEGS, duration=4.0, gate=gate)
    app = make_app(Loader({"cpu": model}))
    start_with(app, [media])
    pump_until(app, lambda: model.calls, timeout=15)
    dialogs.answer = False
    app._on_close()
    assert app.winfo_exists()
    assert not app.stop_requested.is_set()
    dialogs.answer = True
    app._on_close()
    gate.set()
    assert app.stop_requested.is_set()
    with pytest.raises(tk.TclError):
        app.winfo_exists()
    assert dialogs.kinds() == ["askyesno", "askyesno"]


def test_language_code_is_normalised(make_app, dialogs, media):
    model = FakeModel("cpu", segments=TWO_SEGS, duration=4.0)
    app = make_app(Loader({"cpu": model}))
    app.language.set(" IT ")
    start_with(app, [media])
    pump_until(app, lambda: is_idle(app), timeout=15)
    assert model.calls[0][1]["language"] == "it"


def test_explicit_gpu_failure_does_not_reuse_the_broken_gpu(make_app, dialogs, tmp_path):
    files = [make_media(tmp_path, n) for n in ("uno.mp3", "due.mp3")]
    gpu = FakeModel("cuda", error=CUBLAS_MISSING)
    loader = Loader({"cuda": gpu, "cpu": FakeModel("cpu", segments=TWO_SEGS)})
    app = make_app(loader, probe=("cuda", "GPU NVIDIA (CUDA)"))
    app.device_choice.set("GPU (CUDA)")
    start_with(app, files)
    pump_until(app, lambda: is_idle(app), timeout=15)
    assert [c[0] for c in gpu.calls] == [files[0]]  # il secondo file non tocca piu' la GPU guasta
    assert dialogs.kinds() == ["showerror"]
    msg = dialogs.calls[0][2]
    assert "uno.mp3" in msg and "due.mp3" in msg and "riavvia" in msg.lower()
    app.btn_start.invoke()
    pump_until(app, lambda: is_idle(app) and len(dialogs.calls) == 2, timeout=15)
    assert "riavvia" in dialogs.calls[1][2].lower()
    assert [c[1] for c in loader.calls] == ["cuda"]


def test_auto_gpu_error_without_cuda_words_still_falls_back(make_app, dialogs, media):
    gpu = FakeModel("cuda", error=RuntimeError("std::bad_alloc"))
    loader = Loader({"cuda": gpu, "cpu": FakeModel("cpu", segments=TWO_SEGS, duration=4.0)})
    app = make_app(loader, probe=("cuda", "GPU NVIDIA (CUDA)"))
    start_with(app, [media])
    pump_until(app, lambda: is_idle(app), timeout=15)
    assert [c[1] for c in loader.calls] == ["cuda", "cpu"]
    assert dialogs.kinds() == ["showinfo"] and "CPU" in dialogs.calls[0][2]


def test_broken_file_on_gpu_is_not_a_gpu_failure(make_app, dialogs, tmp_path):
    import av
    files = [make_media(tmp_path, n) for n in ("a.mp3", "b.mp3", "c.mp3")]
    broken = av.error.InvalidDataError(1094995529, "Invalid data found when processing input", files[1])
    gpu = FakeModel("cuda", segments=TWO_SEGS, duration=4.0, error_for={"b.mp3": broken})
    loader = Loader({"cuda": gpu, "cpu": FakeModel("cpu", segments=TWO_SEGS)})
    app = make_app(loader, probe=("cuda", "GPU NVIDIA (CUDA)"))
    start_with(app, files)
    pump_until(app, lambda: is_idle(app), timeout=15)
    assert [c[1] for c in loader.calls] == ["cuda"]
    assert (tmp_path / "a.txt").exists() and (tmp_path / "c.txt").exists()
    assert dialogs.kinds() == ["showwarning"] and "b.mp3" in dialogs.calls[0][2]


def test_cancel_summary_keeps_earlier_failures(make_app, dialogs, tmp_path):
    files = [make_media(tmp_path, n) for n in ("rotto.mp3", "lungo.mp3")]
    gate = threading.Event()
    model = FakeModel("cpu", segments=TWO_SEGS, duration=4.0, gate=gate, wait_before_error=False,
                      error_for={"rotto.mp3": ValueError("Invalid data found when processing input")})
    app = make_app(Loader({"cpu": model}))
    start_with(app, files)
    pump_until(app, lambda: len(model.calls) == 2, timeout=15)
    app.btn_stop.invoke()
    gate.set()
    pump_until(app, lambda: is_idle(app), timeout=15)
    assert dialogs.kinds() == ["showinfo"]
    assert "annullat" in dialogs.calls[0][2].lower() and "rotto.mp3" in dialogs.calls[0][2]


def test_cpu_model_failing_after_the_fallback_keeps_the_summary(make_app, dialogs, tmp_path):
    files = [make_media(tmp_path, n) for n in ("a.mp3", "b.mp3", "c.mp3")]
    gpu = FakeModel("cuda", segments=TWO_SEGS, duration=4.0, error_for={"b.mp3": CUBLAS_MISSING})
    loader = Loader({"cuda": gpu, "cpu": OSError("modello CPU non caricabile")})
    app = make_app(loader, probe=("cuda", "GPU NVIDIA (CUDA)"))
    start_with(app, files)
    pump_until(app, lambda: is_idle(app), timeout=15)
    assert (tmp_path / "a.txt").exists()
    assert dialogs.kinds() == ["showwarning"]
    msg = dialogs.calls[0][2]
    assert "b.mp3" in msg and "c.mp3" in msg and "modello CPU non caricabile" in msg


def test_closing_while_running_stops_the_progress_animation(make_app, dialogs, media):
    gate = threading.Event()
    model = FakeModel("cpu", segments=TWO_SEGS, duration=0.0, gate=gate)  # durata 0: barra animata
    app = make_app(Loader({"cpu": model}))
    start_with(app, [media])
    pump_until(app, lambda: model.calls, timeout=15)
    app._on_close()
    gate.set()
    scripts = [str(app.tk.call("after", "info", i)) for i in app.tk.call("after", "info")]
    assert not any("progressbar" in s.lower() for s in scripts)


def test_model_download_shows_the_percentage(make_app, dialogs, media):
    gate = threading.Event()
    seen = []

    def loader(name, device, compute_type, on_download=None, should_stop=None):
        on_download(0, 0)
        on_download(50 * 2**20, 200 * 2**20)
        gate.wait(10)
        return FakeModel("cpu", segments=TWO_SEGS, duration=4.0)

    app = make_app(loader)
    start_with(app, [media])
    pump_until(app, lambda: "25%" in app.lbl_status.cget("text"), timeout=15)
    seen.append((app.lbl_status.cget("text"), str(app.progress.cget("mode")), float(app.progress.cget("value"))))
    gate.set()
    pump_until(app, lambda: is_idle(app), timeout=15)
    text, mode, value = seen[0]
    assert "small" in text and "50" in text and "200" in text
    assert mode == "determinate" and value == 25.0
    assert dialogs.kinds() == ["showinfo"]


def test_stop_during_model_download_cancels(make_app, dialogs, media):
    import time
    import trascrivi_locale as ws

    def loader(name, device, compute_type, on_download=None, should_stop=None):
        on_download(0, 0)
        for i in range(400):
            if should_stop():
                raise ws.DownloadCancelled()
            on_download(i, 400)
            time.sleep(0.02)
        return FakeModel("cpu", segments=TWO_SEGS, duration=4.0)

    app = make_app(loader)
    start_with(app, [media])
    pump_until(app, lambda: "Download" in app.lbl_status.cget("text"), timeout=15)
    app.btn_stop.invoke()
    pump_until(app, lambda: is_idle(app), timeout=15)
    assert dialogs.kinds() == ["showinfo"]
    msg = dialogs.calls[0][2].lower()
    assert "download" in msg and "annullat" in msg and "da capo" in msg
    assert not os.path.exists(os.path.splitext(media)[0] + ".txt")


def model_combobox(app):
    stack = [app]
    while stack:
        w = stack.pop()
        if w.winfo_class() == "TCombobox" and str(w.cget("textvariable")) == str(app.model_name):
            return w
        stack.extend(w.winfo_children())


def test_turbo_model_is_offered(make_app):
    app = make_app(Loader({}))
    assert "turbo" in model_combobox(app).cget("values")


def test_turbo_cannot_translate_and_says_so(make_app, dialogs, media):
    loader = Loader({"cpu": FakeModel("cpu", segments=TWO_SEGS)})
    app = make_app(loader)
    app.model_name.set("turbo")
    app.task.set("translate")
    start_with(app, [media])
    pump_for(app, 0.3)
    assert dialogs.kinds() == ["showwarning"]
    assert "turbo" in dialogs.calls[0][2]
    assert loader.calls == []


# ---------- pulsante "Attiva GPU NVIDIA" ----------

def gpu_button_shown(app):
    pump_for(app, 0.3)  # il rilevamento della GPU arriva dalla coda
    return app.btn_gpu.winfo_manager() != ""


@pytest.mark.parametrize("probe, shown", [
    (("cpu", "CPU (GPU NVIDIA trovata, ma mancano le librerie CUDA 12 cuBLAS)"), True),
    (("cuda", "GPU NVIDIA (CUDA)"), False),
    (("cpu", "CPU (nessuna GPU NVIDIA trovata)"), False),
])
def test_gpu_button_only_when_the_nvidia_libraries_are_missing(make_app, probe, shown):
    import trascrivi_locale as ws
    assert probe[1] == ws.MISSING_CUBLAS or not shown
    app = make_app(Loader({}), probe=probe)
    assert gpu_button_shown(app) is shown


def test_gpu_install_asks_before_downloading(make_app, dialogs):
    import trascrivi_locale as ws
    calls = []
    app = make_app(Loader({}), probe=("cpu", ws.MISSING_CUBLAS), installer=lambda **kw: calls.append(kw))
    assert gpu_button_shown(app)
    dialogs.answer = False
    app.btn_gpu.invoke()
    pump_for(app, 0.3)
    assert dialogs.kinds() == ["askyesno"]
    assert "527" in dialogs.calls[0][2] and "NVIDIA" in dialogs.calls[0][2]
    assert calls == []


def test_gpu_install_shows_progress_and_activates_the_gpu(make_app, dialogs):
    import trascrivi_locale as ws
    state = {"probe": ("cpu", ws.MISSING_CUBLAS)}
    gate = threading.Event()

    def installer(report=None, should_stop=None):
        report(0, 553162896)
        report(276581448, 553162896)
        gate.wait(10)
        state["probe"] = ("cuda", "GPU NVIDIA (CUDA)")
        return "C:/qualcosa/cuda12"

    app = make_app(Loader({}), probe=lambda: state["probe"], installer=installer)
    assert gpu_button_shown(app)
    app.btn_gpu.invoke()
    pump_until(app, lambda: "50%" in app.lbl_status.cget("text"), timeout=15)
    assert str(app.btn_start.cget("state")) == "disabled"  # niente trascrizioni durante l'installazione
    gate.set()
    pump_until(app, lambda: is_idle(app), timeout=15)
    assert dialogs.kinds() == ["askyesno", "showinfo"]
    assert "GPU NVIDIA (CUDA)" in app.accel_label_var.get()
    assert not gpu_button_shown(app)


def test_gpu_install_error_is_shown(make_app, dialogs):
    import trascrivi_locale as ws

    def installer(report=None, should_stop=None):
        raise ws.GpuSetupError("Download non riuscito: controlla la connessione e riprova.")

    app = make_app(Loader({}), probe=("cpu", ws.MISSING_CUBLAS), installer=installer)
    app.btn_gpu.invoke()
    pump_until(app, lambda: is_idle(app) and len(dialogs.calls) == 2, timeout=15)
    assert dialogs.kinds() == ["askyesno", "showerror"]
    assert "connessione" in dialogs.calls[1][2]
    assert gpu_button_shown(app)  # si puo' riprovare


def test_gpu_install_can_be_cancelled(make_app, dialogs):
    import time
    import trascrivi_locale as ws

    def installer(report=None, should_stop=None):
        for i in range(500):
            if should_stop():
                raise ws.DownloadCancelled()
            report(i, 500)
            time.sleep(0.02)

    app = make_app(Loader({}), probe=("cpu", ws.MISSING_CUBLAS), installer=installer)
    app.btn_gpu.invoke()
    pump_until(app, lambda: "%" in app.lbl_status.cget("text"), timeout=15)
    app.btn_stop.invoke()
    pump_until(app, lambda: is_idle(app) and len(dialogs.calls) == 2, timeout=15)
    assert dialogs.kinds() == ["askyesno", "showinfo"]
    assert "annullat" in dialogs.calls[1][2].lower()


def gated_installer(gate):
    def installer(report=None, should_stop=None):
        report(10, 553162896)
        gate.wait(10)
        return "C:/x/cuda12"
    return installer


def test_gpu_button_is_offered_only_on_64_bit_windows(make_app, monkeypatch):
    import trascrivi_locale as ws
    monkeypatch.setattr(ws.platform, "machine", lambda: "ARM64")
    app = make_app(Loader({}), probe=("cpu", ws.MISSING_CUBLAS))
    assert not gpu_button_shown(app)


def test_closing_during_the_gpu_install_talks_about_the_download(make_app, dialogs):
    import trascrivi_locale as ws
    gate = threading.Event()
    app = make_app(Loader({}), probe=("cpu", ws.MISSING_CUBLAS), installer=gated_installer(gate))
    app.btn_gpu.invoke()
    pump_until(app, lambda: "%" in app.lbl_status.cget("text"), timeout=15)
    dialogs.answer = False
    app._on_close()
    gate.set()
    assert "NVIDIA" in dialogs.calls[-1][2]
    assert "file in corso" not in dialogs.calls[-1][2]


def test_leftovers_are_cleaned_at_startup(make_app, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    leftover = tmp_path / "WhisperStudio" / "cuda12-interrotto"
    leftover.mkdir(parents=True)
    (leftover / "cublas.whl").write_bytes(b"x" * 1000)
    old = __import__("time").time() - 600
    for p in (leftover, leftover / "cublas.whl"):
        os.utime(p, (old, old))
    app = make_app(Loader({}))
    pump_until(app, lambda: not leftover.exists(), timeout=5)


def test_no_stale_eta_during_the_gpu_install(make_app, dialogs):
    import time
    import trascrivi_locale as ws
    gate = threading.Event()
    app = make_app(Loader({}), probe=("cpu", ws.MISSING_CUBLAS), installer=gated_installer(gate))
    app.file_start, app.file_total_sec, app.file_done_sec = time.time() - 10, 60.0, 5.0  # lavoro precedente
    app.btn_gpu.invoke()
    pump_for(app, 1.2)
    eta = app.lbl_eta.cget("text")
    gate.set()
    pump_until(app, lambda: is_idle(app), timeout=15)
    assert eta == "--:--:--"


def test_open_output_folder_stays_available_after_the_gpu_install(make_app, dialogs, tmp_path):
    import trascrivi_locale as ws
    gate = threading.Event()
    app = make_app(Loader({}), probe=("cpu", ws.MISSING_CUBLAS), installer=gated_installer(gate))
    app.output_dir = str(tmp_path)          # c'e' gia' stato un lavoro
    app.btn_open.config(state="normal")
    app.btn_gpu.invoke()
    gate.set()
    pump_until(app, lambda: is_idle(app) and len(dialogs.calls) == 2, timeout=15)
    assert str(app.btn_open.cget("state")) == "normal"


def test_libraries_installed_but_gpu_still_unusable_is_a_warning(make_app, dialogs):
    import trascrivi_locale as ws
    app = make_app(Loader({}), probe=("cpu", ws.MISSING_CUBLAS), installer=lambda **kw: "C:/x/cuda12")
    app.btn_gpu.invoke()
    pump_until(app, lambda: is_idle(app) and len(dialogs.calls) == 2, timeout=15)
    assert dialogs.kinds() == ["askyesno", "showwarning"]
    assert gpu_button_shown(app)


def test_gpu_button_hides_during_a_transcription_and_returns_after(make_app, dialogs, media):
    import trascrivi_locale as ws
    gate = threading.Event()
    model = FakeModel("cpu", segments=TWO_SEGS, duration=4.0, gate=gate)
    app = make_app(Loader({"cpu": model}), probe=("cpu", ws.MISSING_CUBLAS))
    assert gpu_button_shown(app)
    start_with(app, [media])
    pump_until(app, lambda: model.calls, timeout=15)
    hidden = app.btn_gpu.winfo_manager() == ""
    gate.set()
    pump_until(app, lambda: is_idle(app), timeout=15)
    assert hidden and gpu_button_shown(app)
