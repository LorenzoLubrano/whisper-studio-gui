"""Test della finestra vera (Tk) con un modello finto al posto di Whisper."""
import os
import threading

from conftest import FakeModel, Loader, is_idle, make_segment, pump_for, pump_until

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


import pytest  # noqa: E402


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
