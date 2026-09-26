"""Prove complete con il motore vero (faster-whisper, modello tiny su CPU).

Lente: si attivano con  pytest -m slow . Il modello tiny viene scaricato al primo uso.
L'audio di prova e' generato con la voce italiana di Windows (System.Speech).
"""
import os
import re
import subprocess

import pytest

import trascrivi_locale as ws
from conftest import close_tk_garbage, is_idle, pump_until

pytestmark = pytest.mark.slow

TESTO = ("Buongiorno a tutti e benvenuti. Il sole splende sul golfo di Napoli "
         "e il Vesuvio si vede chiaramente all'orizzonte.")


@pytest.fixture(scope="session")
def voce_italiana(tmp_path_factory):
    if os.name != "nt":
        pytest.skip("serve la sintesi vocale di Windows")
    out = tmp_path_factory.mktemp("audio") / "voce.wav"
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$v = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -eq 'it-IT' } | Select-Object -First 1;"
        "if (-not $v) { exit 3 };"
        "$s.SelectVoice($v.VoiceInfo.Name);"
        f"$s.SetOutputToWaveFile('{out}');"
        f"$s.Speak('{TESTO.replace(chr(39), chr(39) * 2)}');"
        "$s.Dispose()"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True)
    if r.returncode != 0 or not out.exists():
        pytest.skip("nessuna voce italiana installata")
    return out


@pytest.fixture
def app_real():
    app = ws.WhisperGUI()
    app.withdraw()
    app.device_choice.set("CPU")
    app.model_name.set("tiny")
    yield app
    app.destroy()
    close_tk_garbage()


def test_real_transcription_writes_all_formats(app_real, dialogs, voce_italiana, tmp_path):
    media = tmp_path / "prova.wav"
    media.write_bytes(voce_italiana.read_bytes())
    for var in (app_real.save_txt, app_real.save_srt, app_real.save_vtt, app_real.save_txt_seg):
        var.set(True)
    app_real.files_selected = [str(media)]
    app_real.btn_start.invoke()
    pump_until(app_real, lambda: is_idle(app_real), timeout=300)

    assert dialogs.kinds() == ["showinfo"], dialogs.calls
    txt = (tmp_path / "prova.txt").read_text(encoding="utf-8").lower()
    for parola in ("buongiorno", "golfo", "napoli"):
        assert parola in txt
    srt = (tmp_path / "prova.srt").read_text(encoding="utf-8").splitlines()
    assert srt[0] == "1"
    assert re.fullmatch(r"\d\d:\d\d:\d\d,\d{3} --> \d\d:\d\d:\d\d,\d{3}", srt[1])
    assert (tmp_path / "prova.vtt").read_text(encoding="utf-8").startswith("WEBVTT\n\n00:00:")
    assert (tmp_path / "prova.segments.txt").read_text(encoding="utf-8").startswith("[00:00:")
    assert float(app_real.progress.cget("value")) == 100.0


def test_real_unreadable_file_shows_error(app_real, dialogs, tmp_path):
    media = tmp_path / "rotto.mp3"
    media.write_text("questo non e' un file audio", encoding="utf-8")
    app_real.files_selected = [str(media)]
    app_real.btn_start.invoke()
    pump_until(app_real, lambda: is_idle(app_real), timeout=120)
    errors = [c for c in dialogs.calls if c[0] == "showerror"]
    assert len(errors) == 1 and "rotto.mp3" in errors[0][2]
    assert not (tmp_path / "rotto.txt").exists()
