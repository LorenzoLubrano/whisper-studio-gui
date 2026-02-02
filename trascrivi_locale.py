import os
import time
import json
import threading
import tempfile
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# =======================
#   UTILS (Logica invariata)
# =======================

AUDIO_EXT = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
VIDEO_EXT = (".mp4", ".mkv", ".mov", ".avi")

def format_timestamp(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    seconds = int(seconds)
    s = seconds % 60
    minutes = (seconds // 60) % 60
    hours = seconds // 3600
    return f"{hours:02d}:{minutes:02d}:{s:02d},{ms:03d}"

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
            f.write(f"{format_timestamp(seg['start']).replace(',', '.')} --> {format_timestamp(seg['end']).replace(',', '.')}\n")
            f.write(seg['text'].strip() + "\n\n")

def write_txt_segmented(segments, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{format_timestamp(seg['start'])}–{format_timestamp(seg['end'])}] {seg['text'].strip()}\n")

def hhmmss(secs: float) -> str:
    secs = max(0, int(round(secs)))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def ffprobe_duration(path: str) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
            stderr=subprocess.STDOUT
        )
        info = json.loads(out.decode("utf-8", "ignore"))
        for s in info.get("streams", []):
            if s.get("codec_type") == "audio" and "duration" in s:
                return float(s["duration"])
        if "format" in info and "duration" in info["format"]:
            return float(info["format"]["duration"])
    except Exception:
        pass
    return 0.0

def make_clip(src: str, seconds: int) -> str:
    fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(src)[1] or ".m4a")
    os.close(fd)
    try:
        subprocess.check_call(
            ["ffmpeg", "-y", "-ss", "0", "-t", str(seconds), "-i", src, "-c", "copy", tmp],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return tmp
    except subprocess.CalledProcessError:
        try: os.remove(tmp)
        except OSError: pass
        tmp_wav = tmp + ".wav"
        subprocess.check_call(
            ["ffmpeg", "-y", "-ss", "0", "-t", str(seconds), "-i", src, tmp_wav],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return tmp_wav

# =======================
#   APP (FASTER-WHISPER)
# =======================

class WhisperGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Whisper Studio")
        self.geometry("1000x700")
        self.minsize(950, 650)

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

        # ETA/Progress logic vars
        self.eta_thread      = None
        self.eta_stop        = threading.Event()
        self.stop_requested  = threading.Event()
        self.job_start_time  = None
        self.audio_total_sec = 0.0
        self.processed_audio_sec = 0.0
        self.output_dir      = None

        self.accel_label_var = tk.StringVar(value="Accelerator: CPU")

        self._build_ui()
        self._detect_accelerator()

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
        
        ttk.Button(btn_row_files, text="+ Aggiungi Media", command=self.add_files, style="Ghost.TButton").pack(side="left", padx=(0, 5))
        ttk.Button(btn_row_files, text="Rimuovi Selezionati", command=self.remove_selected, style="Ghost.TButton").pack(side="left", padx=5)
        ttk.Button(btn_row_files, text="Svuota Tutto", command=self.clear_list, style="Ghost.TButton").pack(side="right")

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
        ttk.Combobox(opt_card, textvariable=self.model_name, state="readonly", values=["tiny", "base", "small", "medium", "large-v3"]).grid(row=1, column=0, sticky="ew", pady=(0, 12), padx=(0, 5))
        
        # Compute Type
        ttk.Label(opt_card, text="Precisione", style="Muted.TLabel").grid(row=0, column=1, sticky="w", pady=(0, 2))
        ttk.Combobox(opt_card, textvariable=self.compute_type, state="readonly", values=["auto", "int8", "float16", "float32"]).grid(row=1, column=1, sticky="ew", pady=(0, 12))

        # Language
        ttk.Label(opt_card, text="Lingua (Codice ISO)", style="Muted.TLabel").grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 2))
        lang_entry = ttk.Entry(opt_card, textvariable=self.language)
        lang_entry.grid(row=3, column=0, sticky="ew", pady=(0, 12), padx=(0, 5))
        ttk.Label(opt_card, text="(es. it, en, fr)", style="Muted.TLabel").grid(row=3, column=1, sticky="w")

        # Preset
        ttk.Label(opt_card, text="Velocità vs Qualità", style="Muted.TLabel").grid(row=4, column=0, columnspan=2, sticky="w", pady=(0, 2))
        ttk.Combobox(opt_card, textvariable=self.speed_preset, state="readonly", values=["Fast", "Balanced", "Accurate"]).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 5))

        # -- Task & Output Card --
        out_card = ttk.Labelframe(right_col, text=" Task & Output ", style="Card.TLabelframe", padding=15)
        out_card.pack(fill="x")

        # Task Radio
        tk.Frame(out_card, height=1, bg=self.COL_BG_CARD).pack(pady=2) # Spacer
        task_frame = ttk.Frame(out_card, style="Card.TFrame")
        task_frame.pack(fill="x", pady=(0, 10))
        ttk.Radiobutton(task_frame, text="Trascrivi", value="transcribe", variable=self.task).pack(side="left", padx=(0, 15))
        ttk.Radiobutton(task_frame, text="Traduci (→ Inglese)", value="translate", variable=self.task).pack(side="left")

        tk.Frame(out_card, height=1, bg=self.COL_BORDER).pack(fill="x", pady=10) # Separator

        # Checkboxes grid
        chk_frame = ttk.Frame(out_card, style="Card.TFrame")
        chk_frame.pack(fill="x")
        ttk.Checkbutton(chk_frame, text="Salva .txt", variable=self.save_txt).grid(row=0, column=0, sticky="w", pady=4, padx=(0,10))
        ttk.Checkbutton(chk_frame, text="Salva .srt (Sottotitoli)", variable=self.save_srt).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Checkbutton(chk_frame, text="Salva .vtt (Web)", variable=self.save_vtt).grid(row=1, column=0, sticky="w", pady=4, padx=(0,10))
        ttk.Checkbutton(chk_frame, text="Salva .txt (Segmentato)", variable=self.save_txt_seg).grid(row=1, column=1, sticky="w", pady=4)

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

        self.progress = ttk.Progressbar(status_bar_container, mode="indeterminate", style="Horizontal.TProgressbar")
        self.progress.pack(fill="x")
        self.progress_mode = "indeterminate"

    # ---------- ACCEL DETECTION ----------
    def _detect_accelerator(self):
        accel = "CPU"
        try:
            import torch
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                accel = f"GPU: {torch.cuda.get_device_name(0)}"
        except Exception:
            pass
        self.accel_label_var.set(f"Acceleratore hardware rilevato: {accel}")

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
        if running:
            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="normal")
            self.btn_open.config(state="disabled")
            self.listbox.config(state="disabled")
            
            if self.progress_mode == "indeterminate":
                self.progress.config(mode="indeterminate")
                self.progress.start(10)
            
            self.lbl_status.config(foreground=self.COL_ACCENT)
        else:
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.listbox.config(state="normal")
            
            if self.progress_mode == "indeterminate":
                self.progress.stop()
            
            self.lbl_status.config(foreground=self.COL_TEXT_MAIN)

    # ---------- ACTIONS ----------
    def start(self):
        if not self.files_selected:
            messagebox.showwarning("Nessun File", "Seleziona almeno un file audio o video per iniziare.")
            return

        # Capture settings in main thread
        cfg = {
            "model_name": self.model_name.get(),
            "task": self.task.get(),
            "language": (self.language.get().strip() or None),
            "compute_type": self.compute_type.get() or "auto",
            "preset": self.speed_preset.get(),
            "save_txt": self.save_txt.get(),
            "save_srt": self.save_srt.get(),
            "save_vtt": self.save_vtt.get(),
            "save_txt_seg": self.save_txt_seg.get(),
        }

        self.stop_requested.clear()
        self.eta_stop.clear()
        self.progress_mode = "indeterminate"
        self.set_ui_running(True)
        self.lbl_status.config(text="Inizializzazione ambiente e modelli...")
        
        t = threading.Thread(target=self._run, args=(cfg,), daemon=True)
        t.start()

    def request_stop(self):
        self.stop_requested.set()
        self.lbl_status.config(text="Interruzione in corso...", foreground=self.COL_ERROR)

    # ---------- CORE LOGIC (UNCHANGED) ----------
    def _run(self, cfg):
        try:
            from faster_whisper import WhisperModel
        except Exception as e:
            self.after(0, lambda: self._finish_with_error(
                "faster-whisper non è installato.\nInstalla con: pip install faster-whisper\n\nDettagli: " + str(e)))
            return

        if not self._ffmpeg_available():
            self.after(0, lambda: self._finish_with_error(
                "FFmpeg/FFprobe non trovati. Installa FFmpeg e aggiungi al PATH."))
            return

        # parametri
        model_name   = cfg["model_name"]
        task         = cfg["task"]
        language     = cfg["language"]
        compute_type = cfg["compute_type"]

        # caricamento modello
        self.after(0, lambda: self.lbl_status.config(text=f"Caricamento modello '{model_name}' in memoria..."))
        try:
            model = WhisperModel(model_name, device="auto", compute_type=compute_type)
        except Exception as e:
            self.after(0, lambda: self._finish_with_error(f"Errore caricamento modello: {e}"))
            return

        # preset decoding
        preset = cfg["preset"]
        if preset == "Fast":
            decode = {"beam_size": 1, "temperature": 0.5}
        elif preset == "Accurate":
            decode = {"beam_size": 5, "temperature": 0.0}
        else:  # Balanced
            decode = {"beam_size": 3, "temperature": 0.2}

        total_files = len(self.files_selected)

        for idx, path in enumerate(self.files_selected, start=1):
            if self.stop_requested.is_set():
                break
            if not os.path.isfile(path):
                continue

            base, _ = os.path.splitext(path)
            self.output_dir = os.path.dirname(path)
            self.audio_total_sec = ffprobe_duration(path)
            self.processed_audio_sec = 0.0

            # mini-benchmark
            bench_len = min(60, int(self.audio_total_sec // 2) if self.audio_total_sec else 60)
            self.after(0, lambda p=path, i=idx, t=total_files:
                       self.lbl_status.config(text=f"Analisi preliminare ({i}/{t}): {os.path.basename(p)}..."))
            rtf_est = self._mini_benchmark(model, path, task, language, bench_len, decode, model_name=model_name)

            # progress determinato
            if self.audio_total_sec and rtf_est:
                self.progress_mode = "determinate"
                self.after(0, lambda: self.progress.config(mode="determinate", maximum=100, value=0))
            else:
                self.progress_mode = "indeterminate"
                self.after(0, lambda: self.progress.config(mode="indeterminate"))

            # ETA thread
            self.job_start_time = time.time()
            self.eta_stop.clear()
            self.eta_thread = threading.Thread(target=self._eta_updater_stream, args=(rtf_est,), daemon=True)
            self.eta_thread.start()

            # trascrizione
            if self.stop_requested.is_set(): break
            self.after(0, lambda p=path, i=idx, t=total_files:
                       self.lbl_status.config(text=f"Elaborazione ({i}/{t}): {os.path.basename(p)}"))

            segments_out = []
            text_concat = []
            try:
                gen, info = model.transcribe(
                    path,
                    task="translate" if task == "translate" else "transcribe",
                    language=None if task == "translate" else language,
                    vad_filter=True,
                    **decode
                )
                for seg in gen:
                    if self.stop_requested.is_set():
                        break
                    self.processed_audio_sec = float(seg.end or self.processed_audio_sec)
                    segments_out.append({"start": float(seg.start or 0.0),
                                         "end": float(seg.end or 0.0),
                                         "text": seg.text or ""})
                    text_concat.append(seg.text or "")
            except Exception as e:
                self.after(0, lambda: self._finish_with_error(f"Errore trascrizione:\n{e}"))
                self.eta_stop.set()
                return

            self.eta_stop.set()

            if self.stop_requested.is_set():
                break

            # salvataggio
            full_text = ("".join(text_concat)).strip()
            outs = []
            if cfg["save_txt"]:
                p = f"{base}.txt"
                with open(p, "w", encoding="utf-8") as f: f.write(full_text + "\n")
                outs.append(p)
            if cfg["save_txt_seg"]:
                p = f"{base}.segments.txt"
                write_txt_segmented(segments_out, p); outs.append(p)
            if cfg["save_srt"]:
                p = f"{base}.srt"
                write_srt(segments_out, p); outs.append(p)
            if cfg["save_vtt"]:
                p = f"{base}.vtt"
                write_vtt(segments_out, p); outs.append(p)

            self.after(0, lambda: self.btn_open.config(state="normal"))
            self.after(0, lambda i=idx, t=total_files:
                       self.lbl_status.config(text=f"Completato file {i} di {t}."))

        if self.stop_requested.is_set():
            self.after(0, lambda: self._finish_with_error("Operazione annullata dall'utente."))
        else:
            self.after(0, lambda: self._finish_ok("Tutti i file sono stati elaborati con successo."))

    def _mini_benchmark(self, model, path, task, language, bench_len, decode, model_name="small"):
        clip = None
        try:
            clip = make_clip(path, bench_len)
        except Exception:
            pass

        try:
            t0 = time.time()
            gen, info = model.transcribe(
                clip or path,
                task="translate" if task == "translate" else "transcribe",
                language=None if task == "translate" else language,
                vad_filter=True,
                **decode
            )
            last_end = 0.0
            for seg in gen:
                last_end = seg.end or last_end
                if (clip and last_end >= bench_len - 0.25):
                    break
            elapsed = max(0.001, time.time() - t0)
            audio_used = float(bench_len if clip else min(ffprobe_duration(path) or 60, 60))
            return elapsed / audio_used
        except Exception:
            return 1.0 if model_name in ("tiny","base","small") else 2.0
        finally:
            if clip and os.path.exists(clip):
                try: os.remove(clip)
                except Exception: pass

    def _eta_updater_stream(self, rtf_initial):
        rtf_est = max(1e-6, rtf_initial)
        while not self.eta_stop.is_set():
            try:
                elapsed = time.time() - self.job_start_time if self.job_start_time else 0.0
                proc = self.processed_audio_sec
                if proc > 1e-3:
                    rtf_now = elapsed / proc
                    rtf_est = 0.7 * rtf_est + 0.3 * rtf_now

                if self.audio_total_sec > 0:
                    remaining_audio = max(0.0, self.audio_total_sec - proc)
                    eta_secs = remaining_audio * rtf_est
                    perc = min(100.0, (proc / self.audio_total_sec) * 100.0)
                    self.after(0, lambda s=eta_secs: self.lbl_eta.config(text=f"ETA: {hhmmss(s)}"))
                    if self.progress_mode == "determinate":
                        self.after(0, lambda p=perc: self.progress.config(value=p))
                else:
                    self.after(0, lambda: self.lbl_eta.config(text="ETA: Calcolo..."))
            except Exception:
                pass
            time.sleep(1)

    def _finish_ok(self, msg: str):
        self.set_ui_running(False)
        if self.progress_mode == "determinate":
            self.progress.config(value=100)
        self.lbl_status.config(text="✅ Operazione completata.", foreground=self.COL_SUCCESS)
        self.lbl_eta.config(text="--:--:--")
        messagebox.showinfo("Whisper Studio", msg)

    def _finish_with_error(self, msg: str):
        self.set_ui_running(False)
        if self.progress_mode == "determinate":
            self.progress.config(value=0)
        self.lbl_status.config(text="❌ Errore durante l'esecuzione.", foreground=self.COL_ERROR)
        self.lbl_eta.config(text="--:--:--")
        messagebox.showerror("Errore", msg)

    def _ffmpeg_available(self) -> bool:
        import shutil
        return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

    def open_folder(self):
        if not self.output_dir:
            return
        try:
            if os.name == "nt":
                os.startfile(self.output_dir)
            elif os.name == "posix":
                import subprocess
                subprocess.Popen(["open" if "darwin" in os.sys.platform else "xdg-open", self.output_dir])
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile aprire la cartella:\n{e}")

# =======================
#   RUN
# =======================
if __name__ == "__main__":
    app = WhisperGUI()
    app.mainloop()