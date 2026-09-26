# 🎙️ Whisper Studio

![Windows](https://img.shields.io/badge/Windows-EXE_Available-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Faster-Whisper](https://img.shields.io/badge/AI-Faster_Whisper-purple.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Whisper Studio** è un'interfaccia grafica moderna ed elegante per `faster-whisper`. Permette di trascrivere e tradurre file audio e video localmente con elevata velocità e precisione, sfruttando l'accelerazione GPU (se disponibile) e garantendo la totale privacy dei tuoi dati.

---

## 📥 Download Rapido (Per utenti Windows)
Vuoi usare l'applicazione subito senza installare Python?

1. Vai nella sezione **Releases** di questo repository (nel menu a destra).
2. Scarica `WhisperStudio.exe` dall'ultima versione (circa 100 MB).
3. Fai doppio clic per avviare il programma.

Note:
* **FFmpeg non serve**: l'audio dei file viene letto direttamente dal programma.
* **Primo utilizzo di un modello**: viene scaricato da Internet una sola volta (tiny ≈ 75 MB, small ≈ 480 MB, medium ≈ 1,5 GB, turbo ≈ 1,6 GB, large-v3 ≈ 3 GB). Durante il download la finestra mostra la percentuale e i MB scaricati, e «Interrompi» lo ferma (la volta dopo riparte da capo); da quel momento funziona anche offline.
* Windows può mostrare l'avviso di SmartScreen perché l'eseguibile non è firmato: *Ulteriori informazioni → Esegui comunque*.

### ⚡ GPU NVIDIA (facoltativa)
Con **Dispositivo: Automatico** il programma usa la GPU NVIDIA quando trova le librerie CUDA 12 (cuBLAS); altrimenti usa la CPU e lo scrive in basso a sinistra (es. *«CPU (GPU NVIDIA trovata, ma mancano le librerie CUDA 12 cuBLAS)»*). Per attivare la GPU:
* **il modo più semplice:** premi **«Attiva GPU NVIDIA»** in basso a sinistra (compare solo se hai una scheda NVIDIA senza le librerie). Il programma scarica 527 MB da pypi.org (il pacchetto ufficiale NVIDIA `nvidia-cublas-cu12` 12.9.2.10), controlla che l'impronta SHA-256 sia esattamente quella attesa e installa le due librerie in `%LOCALAPPDATA%\WhisperStudio\cuda12` (circa 740 MB). La GPU si usa subito, senza riavviare; «Interrompi» ferma il download. Le librerie sono di NVIDIA e soggette alla sua licenza;
* oppure installare il [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) **12.x**, dalla 12.8 in poi (la versione 13 non contiene le librerie CUDA 12);
* oppure copiare `cublas64_12.dll` e `cublasLt64_12.dll` nella stessa cartella di `WhisperStudio.exe` (si trovano nel pacchetto pip `nvidia-cublas-cu12`, cartella `nvidia/cublas/bin`).

Le schede RTX serie 50 richiedono cuBLAS 12.8 o successivo. Se la GPU dà errore (all'avvio del modello o durante il lavoro), in modalità Automatico il lavoro prosegue sulla CPU e il riepilogo finale lo segnala; per riprovare la GPU basta riavviare il programma.

---

## ✨ Funzionalità

* **Interfaccia Moderna:** UI pulita e professionale basata su `tkinter` e `ttk` con tema chiaro.
* **Supporto Multimediale:** Compatibile con file video (`.mp4`, `.mkv`, `.mov`, `.avi`) e audio (`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`).
* **Batch Processing:** Carica più file contemporaneamente e lasciali elaborare in coda in modo completamente automatico. Se un file non si può leggere, gli altri vengono elaborati comunque e alla fine un riepilogo dice cosa non è andato.
* **Formati di Output Multipli:** Scegli tra `.txt` (Testo semplice), `.srt` (Sottotitoli standard), `.vtt` (Sottotitoli Web) e `.segments.txt` (Testo con timestamp). I file vengono salvati accanto all'originale, con lo stesso nome; se esistono già, il programma chiede prima di sovrascriverli. File con lo stesso nome (es. `lezione.mp4` e `lezione.m4a`) vengono segnalati prima di iniziare, perché uno cancellerebbe la trascrizione dell'altro.
* **Modelli Flessibili:** Scegli la "taglia" del modello AI in base alle tue esigenze (es. `tiny` per la massima velocità, `large-v3` per la massima precisione). `turbo` ha quasi la precisione di `large-v3` ed è molto più veloce, ideale con la GPU; non sa tradurre, quindi con «Traduci» il programma chiede di scegliere un altro modello.
* **Velocità vs Qualità:** `Fast`, `Balanced` e `Accurate` cambiano l'ampiezza della ricerca (beam size 1, 3, 5). Le parole non vengono scelte a caso: il campionamento interviene solo come ripiego sui pezzi che vengono male.
* **Dispositivo:** Automatico, GPU (CUDA) o CPU.
* **Avanzamento e ETA in tempo reale**, calcolati sul ritmo effettivo della trascrizione.
* **100% Offline:** La trascrizione avviene sul tuo PC; i tuoi file non vengono inviati a nessun server esterno. Serve Internet solo per scaricare un modello la prima volta e, se lo chiedi con il pulsante, le librerie NVIDIA.

---

## 💻 Per Sviluppatori: Installazione dal Sorgente

### 🛠️ Requisiti
* **Python 3.9+** (provato con Python 3.13 su Windows 11)

### 🚀 Installazione e avvio
```bash
git clone https://github.com/LorenzoLubrano/whisper-studio-gui.git
cd whisper-studio-gui
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-gpu.txt   # facoltativo, solo per GPU NVIDIA (circa 700 MB)
python trascrivi_locale.py
```

### 🧪 Test
```bash
pip install -r requirements-dev.txt
pytest            # test veloci (modello finto)
pytest -m slow    # prove complete con il motore vero: modello tiny su CPU, voce italiana di Windows
```

### 📦 Creare l'eseguibile
```bash
pip install -r requirements-dev.txt
pyinstaller WhisperStudio.spec
```
Il file viene creato in `dist\WhisperStudio.exe`.

---

## 📄 Componenti di terzi
L'eseguibile include librerie di terzi (faster-whisper, CTranslate2, ONNX Runtime, PyAV/FFmpeg e altre) con le rispettive licenze: l'elenco completo è in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
