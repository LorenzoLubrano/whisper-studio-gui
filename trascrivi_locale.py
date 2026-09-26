import os
import platform
import re
import sys
import time
import queue
import tempfile
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# L'audio resta sul PC; la libreria dei modelli non invia statistiche d'uso
# ne' un eventuale token Hugging Face salvato sul PC (i modelli usati sono pubblici)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
# Download dei modelli via HTTP invece che Xet: solo cosi' "Interrompi" puo' fermarlo
# (il downloader Xet ignora le eccezioni del contatore di avanzamento e va avanti fino alla fine)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# =======================
#   UTILS
# =======================

AUDIO_EXT = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
VIDEO_EXT = (".mp4", ".mkv", ".mov", ".avi")

# Cartella del programma (accanto all'exe quando e' impacchettato con PyInstaller)
APP_DIR = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))

DEVICE_CHOICES = {"Automatico": "auto", "GPU (CUDA)": "cuda", "CPU": "cpu"}

def format_timestamp(seconds: float, sep: str = ",") -> str:
    # arrotonda una volta sola ai millisecondi: 1.9996 s -> 00:00:02,000 (non 00:00:01,1000)
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"

def write_srt(segments, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(seg['text'].strip() + "\n\n")

def write_vtt(segments, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            f.write(f"{format_timestamp(seg['start'], '.')} --> {format_timestamp(seg['end'], '.')}\n")
            f.write(seg['text'].strip() + "\n\n")

def write_txt_segmented(segments, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{format_timestamp(seg['start'])}–{format_timestamp(seg['end'])}] {seg['text'].strip()}\n")

def output_paths(media_path: str, cfg: dict) -> list:
    base = os.path.splitext(media_path)[0]
    paths = []
    if cfg["save_txt"]:
        paths.append(f"{base}.txt")
    if cfg["save_txt_seg"]:
        paths.append(f"{base}.segments.txt")
    if cfg["save_srt"]:
        paths.append(f"{base}.srt")
    if cfg["save_vtt"]:
        paths.append(f"{base}.vtt")
    return paths

def write_txt(segments, out_path):
    full_text = "".join(seg["text"] for seg in segments).strip()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text + "\n")

def write_outputs(media_path: str, segments: list, cfg: dict) -> list:
    outs = output_paths(media_path, cfg)
    for p in outs:
        if p.endswith(".segments.txt"):
            writer = write_txt_segmented
        elif p.endswith(".txt"):
            writer = write_txt
        elif p.endswith(".srt"):
            writer = write_srt
        else:
            writer = write_vtt
        # prima su un file temporaneo (nome unico, mai un file dell'utente), poi sostituzione
        # in un colpo solo: un errore o una chiusura a meta' non lasciano un file troncato
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(p) or ".", prefix=os.path.basename(p) + ".", suffix=".tmp")
        os.close(fd)
        try:
            writer(segments, tmp)
            os.replace(tmp, p)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    return outs

def hhmmss(secs: float) -> str:
    secs = max(0, int(round(secs)))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def progress_text(done: int, total: int) -> str:
    # per difetto: niente "100%" prima della fine, e MB coerenti con quelli annunciati
    mb = 2 ** 20
    return f"{int(done * 100 // total)}% ({done // mb} di {total // mb} MB)"

def estimate_eta(elapsed: float, processed: float, total: float):
    """Secondi rimanenti stimati dal ritmo reale; None finche' non c'e' ancora nessun segmento."""
    if processed <= 0:
        return None
    return max(0.0, (total - processed) * elapsed / processed)

# =======================
#   GPU / DISPOSITIVO
# =======================

CUBLAS_DLL = "cublas64_12.dll"
_cublas_module = None  # tenuto in vita: CTranslate2 riusa questo modulo gia' caricato
MISSING_CUBLAS = "CPU (GPU NVIDIA trovata, ma mancano le librerie CUDA 12 cuBLAS)"

# Pacchetto ufficiale NVIDIA su PyPI, versione provata su RTX serie 50. Versione e impronta sono
# fissate qui: si installa solo se il file scaricato e' esattamente questo.
CUBLAS_WHEEL_URL = ("https://files.pythonhosted.org/packages/20/e2/fc9a0e985249d873150276d5afb02e39a66817fedbf1"
                    "a385724393e505ed/nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl")
CUBLAS_WHEEL_SIZE = 553162896
CUBLAS_WHEEL_SHA256 = "623f43027d40d44ceadf0043f002bd25cf353e8f13ce90b9a87057019f560661"
CUBLAS_WHEEL_FILES = ("nvidia/cublas/bin/cublas64_12.dll", "nvidia/cublas/bin/cublasLt64_12.dll")

class GpuSetupError(RuntimeError):
    """Attivazione della GPU non riuscita, con un messaggio da mostrare all'utente."""

# spazio necessario durante l'installazione: pacchetto scaricato + le due DLL estratte + margine
GPU_INSTALL_PEAK = CUBLAS_WHEEL_SIZE + 771191808 + 64 * 2**20

def gpu_lib_dir() -> str:
    """Dove il pulsante "Attiva GPU NVIDIA" installa cuBLAS: nei dati dell'utente, non accanto all'exe
    (che puo' stare in Download, in una cartella di sola lettura o essere sostituito).
    Stringa vuota se la cartella dei dati non e' un percorso assoluto: meglio rinunciare che scrivere
    in un posto relativo alla cartella corrente."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    if not os.path.isabs(base):
        return ""
    return os.path.join(base, "WhisperStudio", "cuda12")

def gpu_button_supported() -> bool:
    # il pacchetto NVIDIA scaricato dal pulsante e' per Windows a 64 bit (x86-64)
    # (platform.machine() la prima volta interroga WMI: chiamarla fuori dal thread della finestra)
    return (os.name == "nt" and platform.machine().upper() in ("AMD64", "X86_64")
            and bool(gpu_lib_dir()))

def clean_gpu_install_leftovers():
    """Cancella le cartelle provvisorie di installazioni interrotte (finestra chiusa, crash).

    Salta quelle modificate negli ultimi 2 minuti: potrebbero essere il download in corso di un'altra
    finestra di Whisper Studio.
    """
    import shutil
    dest = gpu_lib_dir()
    if not dest:
        return
    parent = os.path.dirname(dest)
    try:
        names = [n for n in os.listdir(parent) if n.startswith("cuda12-")]
    except OSError:
        return
    now = time.time()
    for name in names:
        path = os.path.join(parent, name)
        try:
            latest = max([os.path.getmtime(path)] + [os.path.getmtime(os.path.join(path, f))
                                                     for f in os.listdir(path)])
        except OSError:
            continue
        if now - latest > 120:
            shutil.rmtree(path, ignore_errors=True)

def _replace_with_retry(src, dst, attempts=5):
    # un antivirus puo' tenere aperte per qualche istante le DLL appena scritte
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.5)

def cuda_library_dirs() -> list:
    """Cartelle note in cui cercare le librerie CUDA 12 (cuBLAS). Mai la cartella corrente."""
    dirs = [APP_DIR, gpu_lib_dir()]
    # CUDA Toolkit: CUDA_PATH e le varianti per versione (CUDA_PATH_V12_8, ...)
    for key, value in sorted(os.environ.items()):
        k = key.upper()
        if k == "CUDA_PATH" or k.startswith("CUDA_PATH_V12"):
            dirs.append(os.path.join(value, "bin"))
    # pacchetti pip nvidia-cublas-cu12 / nvidia-cudnn-cu12 (vedi requirements-gpu.txt)
    try:
        import importlib.util
        for pkg in ("nvidia.cublas", "nvidia.cudnn"):
            spec = importlib.util.find_spec(pkg)
            for loc in (spec.submodule_search_locations or []) if spec else []:
                dirs.append(os.path.join(loc, "bin"))
    except Exception:
        pass
    dirs += os.environ.get("PATH", "").split(os.pathsep)
    found = []
    for d in dirs:
        # solo percorsi assoluti: una voce "." nel PATH vorrebbe dire la cartella corrente
        if d and os.path.isabs(d) and os.path.isdir(d) and d not in found:
            found.append(d)
    return found

def cublas_candidates() -> list:
    """Percorsi completi di cublas64_12.dll nelle cartelle note, in ordine di preferenza."""
    paths = []
    for d in cuda_library_dirs():
        path = os.path.join(d, CUBLAS_DLL)
        if os.path.isfile(path):
            paths.append(path)
    return paths

def find_cublas():
    candidates = cublas_candidates()
    return candidates[0] if candidates else None

def _load_cublas(path: str) -> bool:
    global _cublas_module
    import ctypes
    try:
        # con il percorso completo Windows cerca le dipendenze (cublasLt) nella stessa cartella
        # e non nella cartella corrente; il PATH non viene toccato
        _cublas_module = ctypes.WinDLL(path)
        return True
    except OSError:
        return False

def cuda_status():
    """('cuda', descrizione) se la GPU NVIDIA e' davvero utilizzabile, altrimenti ('cpu', motivo)."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() == 0:
            return "cpu", "CPU (nessuna GPU NVIDIA trovata)"
    except Exception:
        return "cpu", "CPU"
    if os.name != "nt":
        import ctypes
        try:
            ctypes.CDLL("libcublas.so.12")
            return "cuda", "GPU NVIDIA (CUDA)"
        except OSError:
            return "cpu", "CPU (GPU NVIDIA trovata, ma mancano le librerie CUDA 12 cuBLAS)"
    # si prova la successiva se una copia non si carica (es. cublasLt64_12.dll dimenticata accanto)
    for path in cublas_candidates():
        if _load_cublas(path):
            return "cuda", "GPU NVIDIA (CUDA)"
    return "cpu", MISSING_CUBLAS

def install_cuda_libraries(report=None, should_stop=None) -> str:
    """Scarica il pacchetto cuBLAS di NVIDIA da PyPI, ne verifica l'impronta e installa le due DLL.

    Il file va su disco (non in memoria: 527 MB) in una cartella provvisoria accanto a quella finale;
    si estraggono solo le due DLL con nomi fissi, poi la cartella viene spostata al suo posto in un
    colpo solo. Qualunque errore o interruzione non lascia nulla di installato.
    """
    import hashlib
    import http.client
    import shutil
    import urllib.error
    import urllib.request
    import zipfile
    report = report or (lambda done, total: None)

    def check_stop():
        if should_stop and should_stop():
            raise DownloadCancelled("download annullato")

    def network_error(err):
        return GpuSetupError("Download delle librerie NVIDIA non riuscito: controlla la connessione e "
                             f"riprova.\n\n({describe(err)})")

    def disk_error(err):
        return GpuSetupError(f"Impossibile scrivere le librerie NVIDIA sul disco.\n\n({describe(err)})")

    dest = gpu_lib_dir()
    if not dest:
        raise GpuSetupError("Non trovo la cartella dei dati dell'utente (LOCALAPPDATA): le librerie "
                            "NVIDIA non possono essere installate.")
    parent = os.path.dirname(dest)
    try:
        os.makedirs(parent, exist_ok=True)
        clean_gpu_install_leftovers()
        free = shutil.disk_usage(parent).free
    except OSError as err:
        raise disk_error(err) from err
    if free < GPU_INSTALL_PEAK:
        raise GpuSetupError(f"Non c'è abbastanza spazio su disco: durante l'installazione servono circa "
                            f"{GPU_INSTALL_PEAK // 2**20} MB liberi (alla fine ne restano occupati circa "
                            f"740), ce ne sono {free // 2**20}.")
    try:
        staging = tempfile.mkdtemp(prefix="cuda12-", dir=parent)
    except OSError as err:
        raise disk_error(err) from err
    try:
        wheel = os.path.join(staging, "cublas.whl")
        digest = hashlib.sha256()
        done = 0
        request = urllib.request.Request(CUBLAS_WHEEL_URL, headers={"User-Agent": "WhisperStudio"})
        try:
            response = urllib.request.urlopen(request, timeout=30)
        except urllib.error.HTTPError as err:
            if err.code in (404, 410):
                raise GpuSetupError("Il pacchetto delle librerie NVIDIA non è più disponibile su PyPI: "
                                    "serve una versione aggiornata di Whisper Studio.") from err
            raise GpuSetupError(f"Il server dei pacchetti ha risposto con un errore ({err.code}): riprova "
                                "più tardi.") from err
        except (OSError, http.client.HTTPException) as err:
            raise network_error(err) from err
        try:
            out = open(wheel, "wb")
        except OSError as err:
            response.close()
            raise disk_error(err) from err
        with response, out:
            while True:
                check_stop()
                try:
                    # blocchi piccoli: "Interrompi" risponde presto anche su connessioni lente
                    chunk = response.read(256 * 1024)
                except (OSError, http.client.HTTPException) as err:
                    raise network_error(err) from err
                if not chunk:
                    break
                done += len(chunk)
                if done > CUBLAS_WHEEL_SIZE:
                    raise GpuSetupError("Il server ha inviato più dati della dimensione prevista: download "
                                        "interrotto, non è stato installato nulla.")
                try:
                    out.write(chunk)
                except OSError as err:
                    raise disk_error(err) from err
                digest.update(chunk)
                report(done, CUBLAS_WHEEL_SIZE)
        if done != CUBLAS_WHEEL_SIZE:
            raise GpuSetupError("Download incompleto: la connessione si è interrotta prima della fine. "
                                "Riprova.")
        if digest.hexdigest() != CUBLAS_WHEEL_SHA256:
            raise GpuSetupError("Il file scaricato non corrisponde a quello atteso: per sicurezza non è "
                                "stato installato nulla.")
        check_stop()
        out_dir = os.path.join(staging, "cuda12")
        try:
            os.makedirs(out_dir)
            with zipfile.ZipFile(wheel) as archive:
                for member in CUBLAS_WHEEL_FILES:
                    check_stop()
                    target = os.path.join(out_dir, member.rsplit("/", 1)[-1])
                    with archive.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
            os.remove(wheel)
            check_stop()
            if os.path.lexists(dest):
                shutil.rmtree(dest)  # sostituita per intero, mai un misto di vecchio e nuovo
            _replace_with_retry(out_dir, dest)
        except (KeyError, zipfile.BadZipFile, OSError) as err:
            raise GpuSetupError(f"Installazione delle librerie NVIDIA non riuscita.\n\n({describe(err)})") from err
        return dest
    finally:
        shutil.rmtree(staging, ignore_errors=True)

def resolve_device(requested: str, probe=cuda_status) -> str:
    if requested == "cpu":
        return "cpu"
    device, description = probe()
    if requested == "cuda" and device != "cuda":
        raise RuntimeError(f"GPU non utilizzabile: {description}.\n"
                           "Scegli «Automatico» o «CPU» in Dispositivo.")
    return device

_CUDA_ERROR = re.compile(r"\bcu(?:da|blas|dnn)", re.IGNORECASE)

def is_cuda_error(err: BaseException) -> bool:
    # \b: "barracuda.mp3" in un messaggio non e' un errore della GPU
    return bool(_CUDA_ERROR.search(str(err)))

def is_media_error(err: BaseException) -> bool:
    """Errore nel leggere il file (mancante, illeggibile, non multimediale), non della GPU."""
    try:
        import av
        if isinstance(err, av.error.FFmpegError):
            return True
    except Exception:
        pass
    return isinstance(err, OSError)

def describe(err: BaseException) -> str:
    return str(err) or type(err).__name__

def compute_type_for(device: str, requested: str) -> str:
    # float16 non e' disponibile su CPU: CTranslate2 rifiuterebbe il modello
    if device == "cpu" and requested == "float16":
        return "auto"
    return requested

# gli stessi file che scarica faster-whisper per un modello
MODEL_FILES = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]

class DownloadCancelled(Exception):
    """Download del modello interrotto dall'utente."""

def _download_progress_tqdm(report, known_total=0, should_stop=None):
    """Classe tqdm per snapshot_download: nessun output su console, byte scaricati a report(fatti, totali).

    huggingface_hub crea due barre in byte: "Downloading bytes" (dalla rete) e "Reconstructing ..."
    (scritti su disco). Con i repository Xet i byte scritti restano fermi fino alla fine, quelli
    dalla rete avanzano con regolarita': si usa il maggiore dei due. Il totale e' quello vero dei file
    (known_total); senza, quello che la libreria conosce finora.
    """
    import io
    from tqdm import tqdm as base_tqdm
    byte_bars = []
    last = [0.0]

    class ProgressTqdm(base_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["file"] = io.StringIO()  # nell'exe non c'e' una console
            kwargs["mininterval"] = 3600     # il disegno testuale della barra non serve
            super().__init__(*args, **kwargs)
            if self.unit == "B":
                byte_bars.append(self)

        def update(self, n=1):
            out = super().update(n)
            if should_stop and should_stop():
                # interrompe il download in corso (il file parziale viene scartato)
                raise DownloadCancelled("download annullato")
            if self.unit == "B":
                total = known_total or max(int(b.total or 0) for b in byte_bars)
                done = min(max(int(b.n) for b in byte_bars), total)
                now = time.time()
                # il 100% arriva solo a download finito (lo manda download_model_with_progress)
                if total and done < total and now - last[0] > 0.2:
                    last[0] = now
                    report(done, total)
            return out

    ProgressTqdm.byte_bars = byte_bars
    return ProgressTqdm

def _download_size(repo_id: str) -> int:
    """Byte totali dei file del modello, chiesti a Hugging Face prima di scaricare (0 se non disponibile)."""
    import fnmatch
    import huggingface_hub
    try:
        info = huggingface_hub.HfApi().model_info(repo_id, files_metadata=True)
        return sum(int(s.size or 0) for s in info.siblings
                   if any(fnmatch.fnmatch(s.rfilename, p) for p in MODEL_FILES))
    except Exception:
        return 0

def download_model_with_progress(name: str, report, should_stop=None) -> str:
    import huggingface_hub
    try:
        from faster_whisper.utils import _MODELS
        repo_id = name if "/" in name else _MODELS[name]
    except (ImportError, KeyError):
        repo_id = f"Systran/faster-whisper-{name}"
    total = _download_size(repo_id)
    tqdm_class = _download_progress_tqdm(report, total, should_stop)
    path = huggingface_hub.snapshot_download(repo_id, allow_patterns=MODEL_FILES, tqdm_class=tqdm_class)
    final = total or max((int(b.total or 0) for b in tqdm_class.byte_bars), default=0)
    if final:
        report(final, final)
    return path

def load_model(name: str, device: str, compute_type: str, on_download=None, should_stop=None):
    from faster_whisper import WhisperModel
    from huggingface_hub.utils import LocalEntryNotFoundError
    try:
        # prima dal disco: niente rete se il modello e' gia' stato scaricato
        return WhisperModel(name, device=device, compute_type=compute_type, local_files_only=True)
    except LocalEntryNotFoundError:
        report = on_download or (lambda done, total: None)
        report(0, 0)  # download iniziato, dimensione non ancora nota
        path = download_model_with_progress(name, report, should_stop)
        return WhisperModel(path, device=device, compute_type=compute_type)

# =======================
#   APP (FASTER-WHISPER)
# =======================

class WhisperGUI(tk.Tk):
    def __init__(self, model_loader=None, device_probe=None, gpu_installer=None):
        super().__init__()
        self.title("Whisper Studio")
        self.geometry("1000x700")
        self.minsize(950, 650)

        self._load_model_fn = model_loader or load_model
        self._device_probe = device_probe or cuda_status
        self._gpu_installer = gpu_installer or install_cuda_libraries
        self._gpu_missing = False  # GPU NVIDIA presente ma senza cuBLAS: si offre il pulsante
        self._gpu_installing = False
        self._gpu_supported = False  # Windows x64 con cartella dati valida (dal thread del rilevamento)

        # ---- Modern Palette (Slate & Blue) ----
        self.COL_BG_MAIN    = "#f1f5f9"  # Slate 100
        self.COL_BG_CARD    = "#ffffff"  # White
        self.COL_TEXT_MAIN  = "#0f172a"  # Slate 900
        self.COL_TEXT_MUTED = "#64748b"  # Slate 500

        self.COL_ACCENT     = "#2563eb"  # Blue 600
        self.COL_ACCENT_HVR = "#1d4ed8"  # Blue 700
        self.COL_ACCENT_TXT = "#ffffff"

        self.COL_BORDER     = "#cbd5e1"  # Slate 300
        self.COL_INPUT_BG   = "#f8fafc"  # Slate 50

        self.COL_SUCCESS    = "#10b981"  # Emerald 500
        self.COL_ERROR      = "#ef4444"  # Red 500

        # Font configuration
        self.FONT_MAIN = ("Segoe UI", 10)
        self.FONT_BOLD = ("Segoe UI", 10, "bold")
        self.FONT_HEAD = ("Segoe UI", 22, "bold")
        self.FONT_SUB  = ("Segoe UI", 11)
        self.FONT_SMALL= ("Segoe UI", 9)

        self.configure(bg=self.COL_BG_MAIN)
        self._setup_styles()

        # State vars
        self.files_selected = []
        self.model_name     = tk.StringVar(value="small")
        self.task           = tk.StringVar(value="transcribe")
        self.language       = tk.StringVar(value="it")
        self.save_txt       = tk.BooleanVar(value=True)
        self.save_srt       = tk.BooleanVar(value=True)
        self.save_vtt       = tk.BooleanVar(value=False)
        self.save_txt_seg   = tk.BooleanVar(value=False)
        self.speed_preset   = tk.StringVar(value="Balanced")
        self.compute_type   = tk.StringVar(value="auto")
        self.device_choice  = tk.StringVar(value="Automatico")

        # Progress/ETA (scritti solo dal thread della finestra)
        self.stop_requested = threading.Event()
        self.running        = False
        self.file_start     = None
        self.file_total_sec = 0.0
        self.file_done_sec  = 0.0
        self.output_dir     = None

        # Motivo dell'errore GPU, se c'e' stato: per il resto della sessione non si riprova la GPU,
        # perche' dopo un errore CUDA una nuova chiamata puo' bloccarsi dentro CTranslate2
        self._cuda_failed = None

        # Il lavoro in background non tocca mai Tk: manda eventi su questa coda
        self._events = queue.Queue()

        self.accel_label_var = tk.StringVar(value="Acceleratore: rilevamento...")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._detect_accelerator()
        self._poll_id = self.after(100, self._poll_events)

    # ---------- STYLING ----------
    def _setup_styles(self):
        style = ttk.Style()
        try: style.theme_use("clam")
        except Exception: pass

        # Global resets
        style.configure(".",
            background=self.COL_BG_MAIN,
            foreground=self.COL_TEXT_MAIN,
            font=self.FONT_MAIN
        )

        # Frame & Labelframes
        style.configure("Card.TFrame", background=self.COL_BG_CARD, relief="flat")

        # Modern LabelFrame: White bg, subtle border
        style.configure("Card.TLabelframe",
            background=self.COL_BG_CARD,
            foreground=self.COL_TEXT_MAIN,
            bordercolor=self.COL_BORDER,
            borderwidth=1,
            relief="solid"
        )
        style.configure("Card.TLabelframe.Label",
            background=self.COL_BG_CARD,
            foreground=self.COL_ACCENT,
            font=self.FONT_BOLD
        )

        # Labels
        style.configure("TLabel", background=self.COL_BG_CARD, foreground=self.COL_TEXT_MAIN)
        style.configure("Main.TLabel", background=self.COL_BG_MAIN, foreground=self.COL_TEXT_MAIN)
        style.configure("Header.TLabel", background=self.COL_BG_MAIN, foreground=self.COL_TEXT_MAIN, font=self.FONT_HEAD)
        style.configure("SubHeader.TLabel", background=self.COL_BG_MAIN, foreground=self.COL_TEXT_MUTED, font=self.FONT_SUB)
        style.configure("Muted.TLabel", background=self.COL_BG_CARD, foreground=self.COL_TEXT_MUTED, font=self.FONT_SMALL)
        style.configure("Status.TLabel", background=self.COL_BG_CARD, foreground=self.COL_TEXT_MAIN, font=("Segoe UI", 10))

        # Inputs (Entry, Combobox)
        style.configure("TEntry",
            fieldbackground=self.COL_INPUT_BG,
            bordercolor=self.COL_BORDER,
            insertcolor=self.COL_TEXT_MAIN,
            padding=5
        )
        style.map("TEntry", bordercolor=[("focus", self.COL_ACCENT)])

        style.configure("TCombobox",
            fieldbackground=self.COL_INPUT_BG,
            background=self.COL_BG_CARD,
            arrowcolor=self.COL_TEXT_MAIN,
            bordercolor=self.COL_BORDER,
            padding=5
        )
        style.map("TCombobox", fieldbackground=[("readonly", self.COL_INPUT_BG)], bordercolor=[("focus", self.COL_ACCENT)])

        # Buttons
        # Primary Action (Accent Color)
        style.configure("Accent.TButton",
            background=self.COL_ACCENT,
            foreground=self.COL_ACCENT_TXT,
            font=self.FONT_BOLD,
            borderwidth=0,
            focuscolor=self.COL_ACCENT_HVR,
            padding=(20, 10)
        )
        style.map("Accent.TButton",
            background=[("active", self.COL_ACCENT_HVR), ("disabled", self.COL_BORDER)],
            foreground=[("disabled", "#94a3b8")]
        )

        # Secondary/Ghost Button (White with border)
        style.configure("Ghost.TButton",
            background=self.COL_BG_CARD,
            foreground=self.COL_TEXT_MAIN,
            bordercolor=self.COL_BORDER,
            borderwidth=1,
            relief="solid",
            padding=(15, 6)
        )
        style.map("Ghost.TButton",
            background=[("active", "#f1f5f9")],
            bordercolor=[("active", self.COL_TEXT_MUTED)]
        )

        # Link-style Button (alto come una riga di testo)
        style.configure("Link.TButton",
            background=self.COL_BG_CARD,
            foreground=self.COL_ACCENT,
            borderwidth=0,
            focuscolor=self.COL_BG_CARD,
            padding=0,
            font=("Segoe UI", 9, "underline")
        )
        style.map("Link.TButton",
            background=[("active", self.COL_BG_CARD)],
            foreground=[("active", self.COL_ACCENT_HVR)]
        )

        # Progress Bars
        style.configure("Horizontal.TProgressbar",
            troughcolor="#e2e8f0",
            background=self.COL_ACCENT,
            bordercolor=self.COL_BG_CARD,
            thickness=6
        )

    # ---------- UI BUILD ----------
    def _build_ui(self):
        # --- HEADER SECTION ---
        header_frame = ttk.Frame(self, style="Main.TFrame")
        header_frame.pack(fill="x", padx=30, pady=(25, 20))

        ttk.Label(header_frame, text="Whisper Studio", style="Header.TLabel").pack(anchor="w")
        ttk.Label(header_frame, text="Trascrizione e traduzione basata su faster-whisper. Veloce, locale, accurato.", style="SubHeader.TLabel").pack(anchor="w", pady=(5, 0))

        # --- MAIN CONTENT GRID ---
        main_container = ttk.Frame(self, style="Main.TFrame")
        main_container.pack(fill="both", expand=True, padx=30, pady=0)

        # LEFT COLUMN: File List
        left_col = ttk.Frame(main_container, style="Main.TFrame")
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 15))

        file_card = ttk.Labelframe(left_col, text=" File di Origine ", style="Card.TLabelframe", padding=15)
        file_card.pack(fill="both", expand=True)

        # Custom Styled Listbox
        self.listbox = tk.Listbox(file_card,
            height=10,
            selectmode=tk.EXTENDED,
            font=self.FONT_MAIN,
            bg=self.COL_INPUT_BG,
            fg=self.COL_TEXT_MAIN,
            selectbackground=self.COL_ACCENT,
            selectforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=self.COL_BORDER,
            highlightcolor=self.COL_ACCENT
        )
        self.listbox.pack(fill="both", expand=True, pady=(0, 15))

        btn_row_files = ttk.Frame(file_card, style="Card.TFrame")
        btn_row_files.pack(fill="x")

        self.btn_add = ttk.Button(btn_row_files, text="+ Aggiungi Media", command=self.add_files, style="Ghost.TButton")
        self.btn_add.pack(side="left", padx=(0, 5))
        self.btn_remove = ttk.Button(btn_row_files, text="Rimuovi Selezionati", command=self.remove_selected, style="Ghost.TButton")
        self.btn_remove.pack(side="left", padx=5)
        self.btn_clear = ttk.Button(btn_row_files, text="Svuota Tutto", command=self.clear_list, style="Ghost.TButton")
        self.btn_clear.pack(side="right")

        # RIGHT COLUMN: Options
        right_col = ttk.Frame(main_container, style="Main.TFrame")
        right_col.pack(side="right", fill="both", expand=False, padx=(15, 0), ipadx=0)

        # Width constraint for right column
        right_col.columnconfigure(0, minsize=320)

        # -- AI Settings Card --
        opt_card = ttk.Labelframe(right_col, text=" Configurazione AI ", style="Card.TLabelframe", padding=15)
        opt_card.pack(fill="x", pady=(0, 15))

        # Grid layout for options
        opt_card.columnconfigure(1, weight=1)

        # Model
        ttk.Label(opt_card, text="Modello", style="Muted.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 2))
        cb_model = ttk.Combobox(opt_card, textvariable=self.model_name, state="readonly", values=["tiny", "base", "small", "medium", "large-v3", "turbo"])
        cb_model.grid(row=1, column=0, sticky="ew", pady=(0, 12), padx=(0, 5))

        # Compute Type
        ttk.Label(opt_card, text="Precisione", style="Muted.TLabel").grid(row=0, column=1, sticky="w", pady=(0, 2))
        cb_compute = ttk.Combobox(opt_card, textvariable=self.compute_type, state="readonly", values=["auto", "int8", "float16", "float32"])
        cb_compute.grid(row=1, column=1, sticky="ew", pady=(0, 12))

        # Language
        ttk.Label(opt_card, text="Lingua (es. it, en, fr)", style="Muted.TLabel").grid(row=2, column=0, sticky="w", pady=(0, 2))
        lang_entry = ttk.Entry(opt_card, textvariable=self.language)
        lang_entry.grid(row=3, column=0, sticky="ew", pady=(0, 12), padx=(0, 5))

        # Device
        ttk.Label(opt_card, text="Dispositivo", style="Muted.TLabel").grid(row=2, column=1, sticky="w", pady=(0, 2))
        cb_device = ttk.Combobox(opt_card, textvariable=self.device_choice, state="readonly", values=list(DEVICE_CHOICES))
        cb_device.grid(row=3, column=1, sticky="ew", pady=(0, 12))

        # Preset
        ttk.Label(opt_card, text="Velocità vs Qualità", style="Muted.TLabel").grid(row=4, column=0, columnspan=2, sticky="w", pady=(0, 2))
        cb_preset = ttk.Combobox(opt_card, textvariable=self.speed_preset, state="readonly", values=["Fast", "Balanced", "Accurate"])
        cb_preset.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 5))

        # -- Task & Output Card --
        out_card = ttk.Labelframe(right_col, text=" Task & Output ", style="Card.TLabelframe", padding=15)
        out_card.pack(fill="x")

        # Task Radio
        tk.Frame(out_card, height=1, bg=self.COL_BG_CARD).pack(pady=2) # Spacer
        task_frame = ttk.Frame(out_card, style="Card.TFrame")
        task_frame.pack(fill="x", pady=(0, 10))
        rb_transcribe = ttk.Radiobutton(task_frame, text="Trascrivi", value="transcribe", variable=self.task)
        rb_transcribe.pack(side="left", padx=(0, 15))
        rb_translate = ttk.Radiobutton(task_frame, text="Traduci (→ Inglese)", value="translate", variable=self.task)
        rb_translate.pack(side="left")

        tk.Frame(out_card, height=1, bg=self.COL_BORDER).pack(fill="x", pady=10) # Separator

        # Checkboxes grid
        chk_frame = ttk.Frame(out_card, style="Card.TFrame")
        chk_frame.pack(fill="x")
        chk_txt = ttk.Checkbutton(chk_frame, text="Salva .txt", variable=self.save_txt)
        chk_txt.grid(row=0, column=0, sticky="w", pady=4, padx=(0,10))
        chk_srt = ttk.Checkbutton(chk_frame, text="Salva .srt (Sottotitoli)", variable=self.save_srt)
        chk_srt.grid(row=0, column=1, sticky="w", pady=4)
        chk_vtt = ttk.Checkbutton(chk_frame, text="Salva .vtt (Web)", variable=self.save_vtt)
        chk_vtt.grid(row=1, column=0, sticky="w", pady=4, padx=(0,10))
        chk_seg = ttk.Checkbutton(chk_frame, text="Salva .txt (Segmentato)", variable=self.save_txt_seg)
        chk_seg.grid(row=1, column=1, sticky="w", pady=4)

        # Controlli bloccati durante l'elaborazione (widget, stato da ripristinare)
        self._locked_widgets = [
            (self.btn_add, "normal"), (self.btn_remove, "normal"), (self.btn_clear, "normal"),
            (cb_model, "readonly"), (cb_compute, "readonly"), (cb_device, "readonly"), (cb_preset, "readonly"),
            (lang_entry, "normal"), (rb_transcribe, "normal"), (rb_translate, "normal"),
            (chk_txt, "normal"), (chk_srt, "normal"), (chk_vtt, "normal"), (chk_seg, "normal"),
        ]

        # --- FOOTER / STATUS SECTION ---
        footer_frame = ttk.Frame(self, style="Card.TFrame")
        footer_frame.pack(side="bottom", fill="x", padx=0, pady=0)

        # Top footer border
        tk.Frame(footer_frame, height=1, bg=self.COL_BORDER).pack(fill="x")

        content_footer = ttk.Frame(footer_frame, style="Card.TFrame", padding=20)
        content_footer.pack(fill="x")

        # Accelerator Info (Bottom Left)
        info_frame = ttk.Frame(content_footer, style="Card.TFrame")
        info_frame.pack(side="left", fill="y")
        self.lbl_accel = ttk.Label(info_frame, textvariable=self.accel_label_var, style="Muted.TLabel")
        self.lbl_accel.pack(anchor="w")
        self.lbl_eta = ttk.Label(info_frame, text="--:--:--", style="Muted.TLabel")
        self.lbl_eta.pack(anchor="w")
        # al posto dell'ETA, quando non si lavora e mancano le librerie della GPU
        self.btn_gpu = ttk.Button(info_frame, text=f"Attiva GPU NVIDIA (download {int(CUBLAS_WHEEL_SIZE / 2**20)} MB)",
                                  command=self.install_gpu, style="Link.TButton", cursor="hand2")

        # Action Buttons (Bottom Right)
        btn_frame = ttk.Frame(content_footer, style="Card.TFrame")
        btn_frame.pack(side="right")

        self.btn_open = ttk.Button(btn_frame, text="Apri Cartella Output", command=self.open_folder, style="Ghost.TButton", state="disabled")
        self.btn_open.pack(side="left", padx=(0, 10))

        self.btn_stop = ttk.Button(btn_frame, text="Interrompi", command=self.request_stop, style="Ghost.TButton", state="disabled")
        self.btn_stop.pack(side="left", padx=(0, 10))

        self.btn_start = ttk.Button(btn_frame, text="Avvia Elaborazione", command=self.start, style="Accent.TButton")
        self.btn_start.pack(side="left")

        # Progress Bar & Status Text (Middle)
        # We put this above the footer or integrated. Let's put it just above the footer line.
        status_bar_container = ttk.Frame(self, style="Main.TFrame", padding=(30, 0, 30, 10))
        status_bar_container.pack(side="bottom", fill="x")

        self.lbl_status = ttk.Label(status_bar_container, text="Pronto.", style="Main.TLabel", font=("Segoe UI", 11))
        self.lbl_status.pack(anchor="w", pady=(0, 5))

        self.progress = ttk.Progressbar(status_bar_container, mode="determinate", maximum=100, value=0, style="Horizontal.TProgressbar")
        self.progress.pack(fill="x")

    # ---------- ACCEL DETECTION ----------
    def _detect_accelerator(self):
        def probe():
            try:
                clean_gpu_install_leftovers()  # avanzi di un'installazione interrotta (finestra chiusa, crash)
            except Exception:
                pass
            try:
                _, description = self._device_probe()
            except Exception:
                description = "CPU"
            self._post("_accel_detected", description, gpu_button_supported())
        threading.Thread(target=probe, daemon=True).start()

    def _accel_detected(self, description, supported=None):
        if supported is not None:  # calcolato nel thread del rilevamento
            self._gpu_supported = supported
        self.accel_label_var.set(f"Acceleratore: {description}")
        self._gpu_missing = description == MISSING_CUBLAS and self._gpu_supported
        self._refresh_gpu_button()

    def _refresh_gpu_button(self):
        if self._gpu_missing and not self.running:
            self.lbl_eta.pack_forget()
            self.btn_gpu.pack(anchor="w")
        else:
            self.btn_gpu.pack_forget()
            if not self.lbl_eta.winfo_manager():
                self.lbl_eta.pack(anchor="w")

    # ---------- ATTIVAZIONE GPU (librerie NVIDIA scaricate su richiesta) ----------
    def install_gpu(self):
        if self.running:
            return
        if not messagebox.askyesno(
                "Attivare la GPU NVIDIA?",
                "Per usare la scheda video NVIDIA servono le librerie CUDA di NVIDIA (cuBLAS).\n\n"
                f"• Download: {int(CUBLAS_WHEEL_SIZE / 2**20)} MB da pypi.org, il pacchetto ufficiale NVIDIA "
                "nvidia-cublas-cu12 12.9.2.10\n"
                f"• Spazio su disco: circa {f'{GPU_INSTALL_PEAK / 2**30:.1f}'.replace('.', ',')} GB liberi "
                f"durante l'installazione, poi 740 MB in\n  {gpu_lib_dir()}\n"
                "• Il file viene verificato prima di installarlo\n"
                "• Le librerie sono di NVIDIA e soggette alla sua licenza\n\n"
                "Scaricare adesso?"):
            return
        self.stop_requested.clear()
        self.file_start = None  # niente ETA di un lavoro precedente durante il download
        self._gpu_installing = True
        self.set_ui_running(True)
        self.lbl_status.config(text="Download delle librerie NVIDIA...")
        threading.Thread(target=self._run_gpu_install, daemon=True).start()

    def _run_gpu_install(self):
        ended = False
        try:
            self._gpu_installer(report=lambda done, total: self._post("_gpu_progress", done, total),
                                should_stop=self.stop_requested.is_set)
            try:
                device, description = self._device_probe()
            except Exception:
                device, description = "cpu", "CPU"
            self._post("_gpu_installed", device, description)
            ended = True
        except DownloadCancelled:
            self._post("_gpu_install_cancelled")
            ended = True
        except Exception as err:
            self._post("_finish_with_error", f"Attivazione della GPU non riuscita.\n\n{describe(err)}")
            ended = True
        finally:
            if not ended:
                self._post("_finish_with_error", "L'attivazione della GPU si è interrotta in modo imprevisto.")

    def _gpu_progress(self, done, total):
        if total <= 0:
            return
        self.progress.stop()
        self.progress.config(mode="determinate", maximum=100, value=min(100.0, done / total * 100.0))
        if done >= total:
            self.lbl_status.config(text="Verifica e installazione delle librerie NVIDIA...")
        else:
            self.lbl_status.config(text=f"Download delle librerie NVIDIA: {progress_text(done, total)}")

    def _gpu_installed(self, device, description):
        self.set_ui_running(False)
        self._accel_detected(description)
        if device == "cuda":
            self.progress.config(value=100)
            self.lbl_status.config(text="✅ GPU NVIDIA attivata.", foreground=self.COL_SUCCESS)
            messagebox.showinfo("Whisper Studio", "GPU NVIDIA attivata: da ora la trascrizione usa la scheda video.")
        else:
            self.progress.config(value=0)
            self.lbl_status.config(text="⚠ GPU non utilizzabile.", foreground=self.COL_ERROR)
            messagebox.showwarning("Whisper Studio", "Le librerie sono state installate, ma la GPU non risulta "
                                                     f"utilizzabile:\n{description}")

    def _gpu_install_cancelled(self):
        self.set_ui_running(False)
        self.progress.config(value=0)
        self.lbl_status.config(text="⏹ Download annullato.", foreground=self.COL_TEXT_MAIN)
        messagebox.showinfo("Whisper Studio", "Download delle librerie NVIDIA annullato: non è stato installato nulla.")

    # ---------- EVENTI DAL LAVORO IN BACKGROUND ----------
    def _post(self, handler: str, *args):
        """Chiamabile da qualsiasi thread: l'aggiornamento avviene nel thread della finestra."""
        self._events.put((handler, args))

    def _poll_events(self):
        try:
            while True:
                try:
                    handler, args = self._events.get_nowait()
                except queue.Empty:
                    break
                try:
                    getattr(self, handler)(*args)
                except Exception:
                    # segnala l'errore ma non fermare la coda: la finestra resterebbe bloccata
                    self.report_callback_exception(*sys.exc_info())
            if self.running:
                self._refresh_eta()
        finally:
            self._poll_id = self.after(100, self._poll_events)

    def destroy(self):
        # senza questo il controllo periodico e l'animazione della barra resterebbero programmati
        try:
            self.after_cancel(self._poll_id)
            self.progress.stop()
        except (AttributeError, tk.TclError):
            pass
        super().destroy()

    def _set_accel_label(self, text):
        self.accel_label_var.set(text)

    def _set_status(self, text):
        self.lbl_status.config(text=text)

    # ---------- FILE LIST ----------
    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Seleziona file multimediali",
            filetypes=[("Media Files", "*.mp4 *.mkv *.mov *.avi *.mp3 *.wav *.m4a *.flac *.ogg"), ("Tutti i file", "*.*")]
        )
        if not paths:
            return
        for p in paths:
            if p not in self.files_selected:
                ext = os.path.splitext(p.lower())[1]
                if ext in AUDIO_EXT or ext in VIDEO_EXT:
                    self.files_selected.append(p)
        self._refresh_listbox()

    def remove_selected(self):
        sel = list(self.listbox.curselection())[::-1]
        for idx in sel:
            try:
                del self.files_selected[idx]
            except Exception:
                pass
        self._refresh_listbox()

    def clear_list(self):
        self.files_selected = []
        self._refresh_listbox()

    def _refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for p in self.files_selected:
            # Clean display of filename
            self.listbox.insert(tk.END, f"  📄  {os.path.basename(p)}")

    # ---------- UI HELPERS ----------
    def set_ui_running(self, running: bool):
        self.running = running
        for widget, idle_state in self._locked_widgets:
            widget.config(state="disabled" if running else idle_state)
        if running:
            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="normal")
            self.btn_open.config(state="disabled")
            self.listbox.config(state="disabled")
            self.progress.config(mode="indeterminate", value=0)
            self.progress.start(10)
            self.lbl_status.config(foreground=self.COL_ACCENT)
        else:
            # ferma sempre l'animazione, in qualunque modo sia la barra
            self.progress.stop()
            self.progress.config(mode="determinate")
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            if self.output_dir:
                self.btn_open.config(state="normal")
            self.listbox.config(state="normal")
            self.lbl_eta.config(text="--:--:--")
            self._gpu_installing = False
        self._refresh_gpu_button()

    def _start_file_progress(self, duration):
        self.file_start = time.time()
        self.file_total_sec = float(duration or 0.0)
        self.file_done_sec = 0.0
        if self.file_total_sec > 0:
            self.progress.stop()
            self.progress.config(mode="determinate", maximum=100, value=0)
            self.lbl_eta.config(text="ETA: calcolo...")

    def _update_file_progress(self, done_sec):
        self.file_done_sec = done_sec
        if self.file_total_sec > 0:
            self.progress.config(value=min(100.0, done_sec / self.file_total_sec * 100.0))

    def _refresh_eta(self):
        if not self.file_start or self.file_total_sec <= 0:
            return
        eta = estimate_eta(time.time() - self.file_start, self.file_done_sec, self.file_total_sec)
        self.lbl_eta.config(text="ETA: calcolo..." if eta is None else f"ETA: {hhmmss(eta)}")

    def _download_progress(self, model_name, done, total):
        if total <= 0:
            self.lbl_status.config(text=f"Download del modello '{model_name}' da Internet (solo la prima volta)...")
            return
        self.progress.stop()
        self.progress.config(mode="determinate", maximum=100, value=min(100.0, done / total * 100.0))
        self.lbl_status.config(text=f"Download del modello '{model_name}': {progress_text(done, total)[:-1]}, "
                                    "solo la prima volta)")

    def _begin_file(self, status):
        # nuovo file: niente percentuale ne' ETA del file precedente finche' non parte la trascrizione
        self.lbl_status.config(text=status)
        self.file_start = None
        self.file_total_sec = 0.0
        self.progress.config(mode="indeterminate", value=0)
        self.progress.start(10)
        self.lbl_eta.config(text="--:--:--")

    def _file_done(self, idx, total, out_dir):
        self.output_dir = out_dir
        self.btn_open.config(state="normal")
        self.lbl_status.config(text=f"Completato file {idx} di {total}.")

    # ---------- ACTIONS ----------
    def start(self):
        if not self.files_selected:
            messagebox.showwarning("Nessun File", "Seleziona almeno un file audio o video per iniziare.")
            return

        # Capture settings in main thread
        cfg = {
            "files": list(self.files_selected),
            "model_name": self.model_name.get(),
            "task": self.task.get(),
            "language": (self.language.get().strip().lower() or None),
            "compute_type": self.compute_type.get() or "auto",
            "device": DEVICE_CHOICES.get(self.device_choice.get(), "auto"),
            "preset": self.speed_preset.get(),
            "save_txt": self.save_txt.get(),
            "save_srt": self.save_srt.get(),
            "save_vtt": self.save_vtt.get(),
            "save_txt_seg": self.save_txt_seg.get(),
        }

        if cfg["model_name"] == "turbo" and cfg["task"] == "translate":
            # verificato: con "translate" turbo restituisce il testo nella lingua originale
            messagebox.showwarning(
                "Traduzione non disponibile",
                "Il modello turbo non sa tradurre: restituirebbe il testo nella lingua originale.\n\n"
                "Per «Traduci» scegli medium o large-v3; turbo va benissimo per «Trascrivi».")
            return

        if not output_paths("x", cfg):
            messagebox.showwarning("Nessun formato", "Scegli almeno un formato di output da salvare.")
            return

        # es. lezione.mp4 e lezione.m4a scriverebbero entrambi lezione.txt: il secondo cancellerebbe il primo
        owners = {}
        for f in cfg["files"]:
            for p in output_paths(f, cfg):
                owners.setdefault(os.path.normcase(os.path.abspath(p)), []).append(os.path.basename(f))
        clashes = sorted({tuple(names) for names in owners.values() if len(names) > 1})
        if clashes:
            lines = "\n".join(" e ".join(c) for c in clashes)
            messagebox.showwarning(
                "Nomi in conflitto",
                f"Questi file salverebbero la trascrizione con lo stesso nome e uno cancellerebbe l'altro:\n\n"
                f"{lines}\n\nRinominane uno oppure elaborali in due volte.")
            return

        existing = [p for f in cfg["files"] for p in output_paths(f, cfg) if os.path.exists(p)]
        if existing:
            shown = "\n".join(os.path.basename(p) for p in existing[:5])
            more = f"\n... e altri {len(existing) - 5}" if len(existing) > 5 else ""
            if not messagebox.askyesno(
                    "File già presenti",
                    f"Questi file di output esistono già e verranno sovrascritti:\n\n{shown}{more}\n\nContinuare?"):
                return

        self.stop_requested.clear()
        self.file_start = None
        self.set_ui_running(True)
        self.lbl_status.config(text="Inizializzazione ambiente e modelli...")

        t = threading.Thread(target=self._run, args=(cfg,), daemon=True)
        t.start()

    def request_stop(self):
        self.stop_requested.set()
        self.lbl_status.config(text="Interruzione in corso...", foreground=self.COL_ERROR)

    def _on_close(self):
        if self.running:
            if self._gpu_installing:
                question = ("È in corso il download delle librerie NVIDIA: verrà annullato e non sarà installato "
                            "nulla.\n\nChiudere comunque?")
            else:
                question = "C'è un'elaborazione in corso: il file in corso non verrà salvato.\n\nChiudere comunque?"
            if not messagebox.askyesno("Chiudere Whisper Studio?", question):
                return
            self.stop_requested.set()
        self.destroy()

    def report_callback_exception(self, exc, val, tb):
        # nell'exe senza console stderr non esiste: l'errore va mostrato in una finestra
        import traceback
        if sys.stderr:
            traceback.print_exception(exc, val, tb)
        try:
            messagebox.showerror("Errore imprevisto", f"{exc.__name__}: {val}")
        except Exception:
            pass

    # ---------- CORE LOGIC (thread in background: comunica solo con _post) ----------
    def _run(self, cfg):
        ended = False
        try:
            self._process(cfg)
            ended = True
        except DownloadCancelled:
            self._post("_finish_download_cancelled", cfg["model_name"])
            ended = True
        except Exception as err:
            # il messaggio viene calcolato qui: `err` non esiste piu' fuori da questo blocco
            self._post("_finish_with_error", describe(err))
            ended = True
        finally:
            if not ended:  # anche un'uscita anomala del thread deve sbloccare la finestra
                self._post("_finish_with_error", "L'elaborazione si è interrotta in modo imprevisto.")

    def _gpu_failed(self, err) -> str:
        # da qui in poi la GPU non si usa piu' in questa sessione: riusarla puo' bloccarsi
        self._cuda_failed = describe(err).splitlines()[-1]
        self._post("_set_accel_label", "Acceleratore: CPU (la GPU ha dato errore: si riprova al prossimo avvio)")
        return f"la GPU ha dato errore ({self._cuda_failed}): il lavoro è stato fatto sulla CPU."

    def _gpu_restart_hint(self) -> str:
        return (f"la GPU ha dato errore ({self._cuda_failed}). Chiudi e riavvia Whisper Studio per "
                "riprovarla, oppure scegli «Automatico» o «CPU» in Dispositivo.")

    def _process(self, cfg):
        gpu_note = None
        if cfg["device"] != "cpu" and self._cuda_failed:
            if cfg["device"] == "cuda":
                raise RuntimeError(f"La GPU ha già dato errore in questa sessione:\n{self._cuda_failed}\n\n"
                                   "Chiudi e riavvia Whisper Studio per riprovarla, "
                                   "oppure scegli «Automatico» o «CPU» in Dispositivo.")
            device = "cpu"
        else:
            device = resolve_device(cfg["device"], self._device_probe)
        try:
            model = self._load(cfg, device)
        except Exception as err:
            if not (device == "cuda" and is_cuda_error(err)):
                raise
            gpu_note = self._gpu_failed(err)
            if cfg["device"] != "auto":
                raise RuntimeError(f"Errore caricamento modello sulla GPU:\n{self._gpu_restart_hint()}") from err
            device = "cpu"
            model = self._load(cfg, device)

        # preset decoding: temperatura predefinita di faster-whisper (parte da 0 e sale solo se un
        # pezzo viene male). Una temperatura fissa > 0 sceglierebbe le parole a caso ignorando beam_size
        preset = cfg["preset"]
        if preset == "Fast":
            decode = {"beam_size": 1}
        elif preset == "Accurate":
            decode = {"beam_size": 5}
        else:  # Balanced
            decode = {"beam_size": 3}

        files = cfg["files"]
        total_files = len(files)
        done, failed, missing = [], [], []

        for idx, path in enumerate(files, start=1):
            if self.stop_requested.is_set():
                break
            name = os.path.basename(path)
            if not os.path.isfile(path):
                missing.append(name)
                continue

            self._post("_begin_file", f"Elaborazione ({idx}/{total_files}): {name}")
            try:
                segments = self._transcribe(model, path, cfg, decode)
            except Exception as err:
                if self.stop_requested.is_set():
                    break
                # sulla GPU tutto cio' che non e' un errore di lettura del file e' un guasto della GPU
                if not (device == "cuda" and (is_cuda_error(err) or not is_media_error(err))):
                    failed.append((name, describe(err)))
                    continue
                gpu_note = self._gpu_failed(err)
                if cfg["device"] != "auto":
                    # GPU scelta esplicitamente: niente ripiego, e il modello guasto non si riusa
                    failed.extend((os.path.basename(p), self._gpu_restart_hint()) for p in files[idx - 1:])
                    gpu_note = None
                    break
                # la GPU c'e' ma non funziona: si riparte con un modello nuovo sulla CPU
                self._post("_begin_file", f"La GPU non risponde, passo alla CPU: {name}")
                device = "cpu"
                try:
                    model = self._load(cfg, device)
                except Exception as load_err:
                    if self.stop_requested.is_set():
                        break
                    failed.extend((os.path.basename(p), describe(load_err)) for p in files[idx - 1:])
                    break
                try:
                    segments = self._transcribe(model, path, cfg, decode)
                except Exception as err2:
                    if self.stop_requested.is_set():
                        break
                    failed.append((name, describe(err2)))
                    continue

            if segments is None:  # interrotto dall'utente
                break

            try:
                write_outputs(path, segments, cfg)
            except OSError as err:
                failed.append((name, f"impossibile salvare i file di output: {describe(err)}"))
                continue
            done.append(name)
            self._post("_file_done", idx, total_files, os.path.dirname(path))

        if self.stop_requested.is_set():
            self._post("_finish_cancelled", done, failed, missing, gpu_note)
        else:
            self._post("_finish_summary", done, failed, missing, gpu_note)

    def _load(self, cfg, device):
        model_name = cfg["model_name"]
        compute_type = compute_type_for(device, cfg["compute_type"])
        self._post("_set_status", f"Caricamento modello '{model_name}' in memoria...")

        def on_download(done=0, total=0):
            self._post("_download_progress", model_name, done, total)

        try:
            model = self._load_model_fn(model_name, device, compute_type, on_download=on_download,
                                        should_stop=self.stop_requested.is_set)
        except DownloadCancelled:
            raise
        except Exception as err:
            raise RuntimeError(f"Errore caricamento modello '{model_name}':\n{describe(err)}") from err
        inner = getattr(model, "model", None)
        used = "GPU NVIDIA (CUDA)" if getattr(inner, "device", device) == "cuda" else "CPU"
        precision = getattr(inner, "compute_type", compute_type)
        self._post("_set_accel_label", f"In uso: {used} · precisione {precision}")
        return model

    def _transcribe(self, model, path, cfg, decode):
        """Segmenti trascritti, oppure None se l'utente ha chiesto di interrompere."""
        task = cfg["task"]
        gen, info = model.transcribe(
            path,
            task="translate" if task == "translate" else "transcribe",
            language=None if task == "translate" else cfg["language"],
            vad_filter=True,
            **decode
        )
        self._post("_start_file_progress", float(getattr(info, "duration", 0.0) or 0.0))
        segments_out = []
        for seg in gen:
            if self.stop_requested.is_set():
                return None
            segments_out.append({"start": float(seg.start or 0.0),
                                 "end": float(seg.end or 0.0),
                                 "text": seg.text or ""})
            self._post("_update_file_progress", float(seg.end or 0.0))
        if self.stop_requested.is_set():
            return None
        return segments_out

    @staticmethod
    def _summary_details(failed, missing, gpu_note) -> str:
        parts = []
        if failed:
            parts.append("Non è stato possibile elaborare:\n" + "\n".join(f"• {n}: {why}" for n, why in failed))
        if missing:
            parts.append("File non trovati (saltati):\n" + "\n".join(f"• {n}" for n in missing))
        if gpu_note:
            parts.append(f"Nota: {gpu_note}")
        return "\n\n".join(parts)

    def _finish_summary(self, done, failed, missing, gpu_note):
        if not failed and not missing:
            self._finish_ok("Tutti i file sono stati elaborati con successo." + (f"\n\nNota: {gpu_note}" if gpu_note else ""))
            return
        details = self._summary_details(failed, missing, gpu_note)
        if not done:
            self._finish_with_error(details)
            return
        # una parte e' andata a buon fine: i file completati sono salvati
        self.set_ui_running(False)
        self.progress.config(value=100)
        self.lbl_status.config(text="⚠ Completato con problemi.", foreground=self.COL_ERROR)
        messagebox.showwarning("Whisper Studio", f"Completati {len(done)} file su "
                                                 f"{len(done) + len(failed) + len(missing)}.\n\n{details}")

    def _finish_ok(self, msg: str):
        self.set_ui_running(False)
        self.progress.config(value=100)
        self.lbl_status.config(text="✅ Operazione completata.", foreground=self.COL_SUCCESS)
        messagebox.showinfo("Whisper Studio", msg)

    def _finish_cancelled(self, done, failed, missing, gpu_note):
        self.set_ui_running(False)
        self.progress.config(value=0)
        self.lbl_status.config(text="⏹ Operazione annullata.", foreground=self.COL_TEXT_MAIN)
        details = self._summary_details(failed, missing, gpu_note)
        messagebox.showinfo("Whisper Studio", f"Operazione annullata dopo {len(done)} file completati. "
                                              "Il file in corso non è stato salvato; quelli già completati restano."
                                              + (f"\n\n{details}" if details else ""))

    def _finish_download_cancelled(self, model_name):
        self.set_ui_running(False)
        self.progress.config(value=0)
        self.lbl_status.config(text="⏹ Download annullato.", foreground=self.COL_TEXT_MAIN)
        messagebox.showinfo("Whisper Studio", f"Download del modello '{model_name}' annullato: nessun file è stato "
                                              "elaborato.\n\nAl prossimo avvio il download ripartirà da capo.")

    def _finish_with_error(self, msg: str):
        self.set_ui_running(False)
        self.progress.config(value=0)
        self.lbl_status.config(text="❌ Errore durante l'esecuzione.", foreground=self.COL_ERROR)
        messagebox.showerror("Errore", msg)

    def open_folder(self):
        if not self.output_dir:
            return
        try:
            if os.name == "nt":
                os.startfile(self.output_dir)
            elif os.name == "posix":
                subprocess.Popen(["open" if "darwin" in sys.platform else "xdg-open", self.output_dir])
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile aprire la cartella:\n{e}")

# =======================
#   RUN
# =======================
if __name__ == "__main__":
    app = WhisperGUI()
    app.mainloop()
