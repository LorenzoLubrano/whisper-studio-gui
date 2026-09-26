import os
import sys
import time
import queue
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# L'audio resta sul PC; con questa variabile anche la libreria dei modelli non invia statistiche d'uso
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

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

def write_outputs(media_path: str, segments: list, cfg: dict) -> list:
    outs = output_paths(media_path, cfg)
    for p in outs:
        if p.endswith(".segments.txt"):
            write_txt_segmented(segments, p)
        elif p.endswith(".txt"):
            full_text = "".join(seg["text"] for seg in segments).strip()
            with open(p, "w", encoding="utf-8") as f:
                f.write(full_text + "\n")
        elif p.endswith(".srt"):
            write_srt(segments, p)
        else:
            write_vtt(segments, p)
    return outs

def hhmmss(secs: float) -> str:
    secs = max(0, int(round(secs)))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def estimate_eta(elapsed: float, processed: float, total: float):
    """Secondi rimanenti stimati dal ritmo reale; None finche' non c'e' ancora nessun segmento."""
    if processed <= 0:
        return None
    return max(0.0, (total - processed) * elapsed / processed)

# =======================
#   GPU / DISPOSITIVO
# =======================

_dll_dir_handles = []  # vanno tenuti in vita, altrimenti Windows toglie le cartelle aggiunte

def cuda_library_dirs() -> list:
    """Cartelle in cui cercare le librerie CUDA 12 (cuBLAS) necessarie alla GPU."""
    dirs = [APP_DIR]
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        dirs.append(os.path.join(cuda_path, "bin"))
    # pacchetti pip nvidia-cublas-cu12 / nvidia-cudnn-cu12 (vedi requirements-gpu.txt)
    try:
        import importlib.util
        for pkg in ("nvidia.cublas", "nvidia.cudnn"):
            spec = importlib.util.find_spec(pkg)
            for loc in (spec.submodule_search_locations or []) if spec else []:
                dirs.append(os.path.join(loc, "bin"))
    except Exception:
        pass
    found = []
    for d in dirs:
        if os.path.isdir(d) and d not in found:
            found.append(d)
    return found

def enable_cuda_libraries():
    for d in cuda_library_dirs():
        if hasattr(os, "add_dll_directory"):
            try:
                _dll_dir_handles.append(os.add_dll_directory(d))
            except OSError:
                pass
        path = os.environ.get("PATH", "")
        if d not in path.split(os.pathsep):
            os.environ["PATH"] = d + os.pathsep + path

def _cublas_loadable() -> bool:
    import ctypes
    try:
        if os.name == "nt":
            # winmode=0: stessa ricerca (PATH compreso) che usa CTranslate2 per caricare cuBLAS
            ctypes.WinDLL("cublas64_12.dll", winmode=0)
        else:
            ctypes.CDLL("libcublas.so.12")
        return True
    except OSError:
        return False

def cuda_status():
    """('cuda', descrizione) se la GPU NVIDIA e' davvero utilizzabile, altrimenti ('cpu', motivo)."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() == 0:
            return "cpu", "CPU"
    except Exception:
        return "cpu", "CPU"
    enable_cuda_libraries()
    if not _cublas_loadable():
        return "cpu", "CPU (GPU NVIDIA trovata, ma mancano le librerie CUDA 12 cuBLAS)"
    return "cuda", "GPU NVIDIA (CUDA)"

def resolve_device(requested: str, probe=cuda_status) -> str:
    if requested == "cpu":
        return "cpu"
    device, description = probe()
    if requested == "cuda" and device != "cuda":
        raise RuntimeError(f"GPU non utilizzabile: {description}.\n"
                           "Scegli «Automatico» o «CPU» in Dispositivo.")
    return device

def is_cuda_error(err: Exception) -> bool:
    text = str(err).lower()
    return any(k in text for k in ("cuda", "cublas", "cudnn"))

def compute_type_for(device: str, requested: str) -> str:
    # float16 non e' disponibile su CPU: CTranslate2 rifiuterebbe il modello
    if device == "cpu" and requested == "float16":
        return "auto"
    return requested

def load_model(name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel
    from huggingface_hub.utils import LocalEntryNotFoundError
    try:
        # prima dal disco: niente rete se il modello e' gia' stato scaricato
        return WhisperModel(name, device=device, compute_type=compute_type, local_files_only=True)
    except LocalEntryNotFoundError:
        return WhisperModel(name, device=device, compute_type=compute_type)

# =======================
#   APP (FASTER-WHISPER)
# =======================

class WhisperGUI(tk.Tk):
    def __init__(self, model_loader=None, device_probe=None):
        super().__init__()
        self.title("Whisper Studio")
        self.geometry("1000x700")
        self.minsize(950, 650)

        self._load_model_fn = model_loader or load_model
        self._device_probe = device_probe or cuda_status

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

        # Il lavoro in background non tocca mai Tk: manda eventi su questa coda
        self._events = queue.Queue()

        self.accel_label_var = tk.StringVar(value="Acceleratore: rilevamento...")

        self._build_ui()
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
        cb_model = ttk.Combobox(opt_card, textvariable=self.model_name, state="readonly", values=["tiny", "base", "small", "medium", "large-v3"])
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
                _, description = self._device_probe()
            except Exception:
                description = "CPU"
            self._post("_set_accel_label", f"Acceleratore: {description}")
        threading.Thread(target=probe, daemon=True).start()

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
        # senza questo il controllo periodico resterebbe programmato dopo la chiusura
        try:
            self.after_cancel(self._poll_id)
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
            self.listbox.config(state="normal")
            self.lbl_eta.config(text="--:--:--")

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

    def _file_done(self, idx, total):
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
            "language": (self.language.get().strip() or None),
            "compute_type": self.compute_type.get() or "auto",
            "device": DEVICE_CHOICES.get(self.device_choice.get(), "auto"),
            "preset": self.speed_preset.get(),
            "save_txt": self.save_txt.get(),
            "save_srt": self.save_srt.get(),
            "save_vtt": self.save_vtt.get(),
            "save_txt_seg": self.save_txt_seg.get(),
        }

        if not output_paths("x", cfg):
            messagebox.showwarning("Nessun formato", "Scegli almeno un formato di output da salvare.")
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

    # ---------- CORE LOGIC (thread in background: comunica solo con _post) ----------
    def _run(self, cfg):
        try:
            self._process(cfg)
        except Exception as err:
            # il messaggio viene calcolato qui: `err` non esiste piu' fuori da questo blocco
            self._post("_finish_with_error", str(err) or type(err).__name__)

    def _process(self, cfg):
        device = resolve_device(cfg["device"], self._device_probe)
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
        skipped = []

        for idx, path in enumerate(files, start=1):
            if self.stop_requested.is_set():
                break
            if not os.path.isfile(path):
                skipped.append(os.path.basename(path))
                continue

            name = os.path.basename(path)
            self._post("_set_status", f"Elaborazione ({idx}/{total_files}): {name}")
            try:
                segments = self._transcribe(model, path, cfg, decode)
            except Exception as err:
                if not (cfg["device"] == "auto" and device == "cuda" and is_cuda_error(err)):
                    raise RuntimeError(f"Errore durante la trascrizione di «{name}»:\n{err}") from err
                # la GPU c'e' ma non funziona: si riparte con un modello nuovo sulla CPU
                self._post("_set_status", f"La GPU non risponde, passo alla CPU: {name}")
                device = "cpu"
                model = self._load(cfg, device)
                try:
                    segments = self._transcribe(model, path, cfg, decode)
                except Exception as err2:
                    raise RuntimeError(f"Errore durante la trascrizione di «{name}»:\n{err2}") from err2

            if segments is None:  # interrotto dall'utente
                break

            write_outputs(path, segments, cfg)
            self.output_dir = os.path.dirname(path)
            self._post("_file_done", idx, total_files)

        if self.stop_requested.is_set():
            self._post("_finish_cancelled")
        elif skipped:
            self._post("_finish_with_error", "Questi file non sono stati trovati e sono stati saltati:\n" + "\n".join(skipped))
        else:
            self._post("_finish_ok", "Tutti i file sono stati elaborati con successo.")

    def _load(self, cfg, device):
        model_name = cfg["model_name"]
        compute_type = compute_type_for(device, cfg["compute_type"])
        self._post("_set_status", f"Caricamento modello '{model_name}' in memoria... (al primo utilizzo viene scaricato)")
        try:
            model = self._load_model_fn(model_name, device, compute_type)
        except Exception as err:
            raise RuntimeError(f"Errore caricamento modello '{model_name}':\n{err}") from err
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

    def _finish_ok(self, msg: str):
        self.set_ui_running(False)
        self.progress.config(value=100)
        self.lbl_status.config(text="✅ Operazione completata.", foreground=self.COL_SUCCESS)
        messagebox.showinfo("Whisper Studio", msg)

    def _finish_cancelled(self):
        self.set_ui_running(False)
        self.progress.config(value=0)
        self.lbl_status.config(text="⏹ Operazione annullata.", foreground=self.COL_TEXT_MAIN)
        messagebox.showinfo("Whisper Studio", "Operazione annullata. Il file in corso non è stato salvato; "
                                              "quelli già completati restano.")

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
