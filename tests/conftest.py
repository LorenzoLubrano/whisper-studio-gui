import gc
import os
import sys
import time
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import trascrivi_locale as ws  # noqa: E402
from faster_whisper.transcribe import Segment  # noqa: E402


def make_segment(start, end, text, idx=1):
    return Segment(id=idx, seek=0, start=start, end=end, text=text, tokens=[],
                   avg_logprob=-0.2, compression_ratio=1.2, no_speech_prob=0.01,
                   words=None, temperature=0.0)


def make_info(duration):
    return types.SimpleNamespace(language="it", language_probability=0.99,
                                 duration=duration, duration_after_vad=duration,
                                 all_language_probs=None, transcription_options=None,
                                 vad_options=None)


class FakeModel:
    """Sostituto di WhisperModel: stessa interfaccia usata dall'app (transcribe + model.device)."""

    def __init__(self, device, segments=(), duration=10.0, error=None, gate=None, error_for=None,
                 wait_before_error=True):
        self.model = types.SimpleNamespace(device=device, compute_type="float32")
        self._segments = list(segments)
        self._duration = duration
        self._error = error
        self._error_for = error_for or {}  # {nome file: eccezione}: fallisce solo su quei file
        self._gate = gate  # threading.Event: se c'e', si ferma (prima dell'errore o dopo il 1o segmento) finche' non viene impostato
        self._wait_before_error = wait_before_error
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        error = self._error_for.get(os.path.basename(path), self._error)

        def gen():
            if error is not None:
                if self._gate is not None and self._wait_before_error:
                    self._gate.wait(10)
                raise error
            for i, seg in enumerate(self._segments):
                if i == 1 and self._gate is not None:
                    self._gate.wait(10)
                yield seg

        return gen(), make_info(self._duration)


class Loader:
    """Registra le chiamate (nome, device, compute_type) e restituisce i modelli preparati."""

    def __init__(self, by_device):
        self.by_device = by_device
        self.calls = []

    def __call__(self, name, device, compute_type, on_download=None):
        self.calls.append((name, device, compute_type))
        model = self.by_device[device]
        if isinstance(model, BaseException):
            raise model
        return model


class Dialogs:
    def __init__(self):
        self.calls = []
        self.answer = True

    def kinds(self):
        return [c[0] for c in self.calls]


@pytest.fixture
def dialogs(monkeypatch):
    d = Dialogs()
    for kind in ("showinfo", "showerror", "showwarning"):
        monkeypatch.setattr(ws.messagebox, kind,
                            lambda title, msg, _k=kind, **kw: d.calls.append((_k, title, msg)))

    def ask(title, msg, **kw):
        d.calls.append(("askyesno", title, msg))
        return d.answer
    monkeypatch.setattr(ws.messagebox, "askyesno", ask)
    return d


@pytest.fixture
def make_app():
    apps = []

    def factory(loader, probe=("cpu", "CPU")):
        app = ws.WhisperGUI(model_loader=loader, device_probe=lambda: probe)
        app.withdraw()
        apps.append(app)
        return app

    yield factory
    for app in apps:
        try:
            app.destroy()
        except Exception:
            pass
    apps.clear()
    close_tk_garbage()


def close_tk_garbage():
    """Raccoglie qui, nel thread principale, le variabili Tk delle finestre chiuse.

    Altrimenti la raccolta puo' capitare nel thread di lavoro del test successivo:
    Variable.__del__ chiama Tcl da li' e aspetta un mainloop che nei test non c'e'.
    """
    gc.collect()


def pump_until(app, cond, timeout=30.0):
    t0 = time.time()
    while not cond():
        app.update()
        time.sleep(0.02)
        if time.time() - t0 > timeout:
            raise AssertionError(f"condizione non raggiunta in {timeout}s; stato: {app.lbl_status.cget('text')!r}")


def pump_for(app, seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.update()
        time.sleep(0.02)


def is_idle(app):
    return str(app.btn_start.cget("state")) == "normal"


@pytest.fixture
def media(tmp_path):
    """Un file 'media' qualsiasi: con FakeModel il contenuto non viene letto."""
    return make_media(tmp_path, "intervista.mp3")


def make_media(folder, name):
    p = folder / name
    p.write_bytes(b"\x00" * 16)
    return str(p)
