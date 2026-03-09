# 🎙️ Whisper Studio

![Windows](https://img.shields.io/badge/Windows-EXE_Available-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Faster-Whisper](https://img.shields.io/badge/AI-Faster_Whisper-purple.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Whisper Studio** è un'interfaccia grafica moderna ed elegante per `faster-whisper`. Permette di trascrivere e tradurre file audio e video localmente con elevata velocità e precisione, sfruttando l'accelerazione GPU (se disponibile) e garantendo la totale privacy dei tuoi dati.

---

## 📥 Download Rapido (Per utenti Windows)
Vuoi usare l'applicazione subito senza installare Python o configurare l'ambiente di sviluppo? 
Puoi scaricare la versione eseguibile pronta all'uso!

1. Vai nella sezione **Releases** di questo repository (nel menu a destra).
2. Scarica l'archivio contenente l'eseguibile (es. `WhisperStudio.exe`).
3. Fai doppio clic per avviare il programma.
*(Nota: Assicurati di avere [FFmpeg](https://ffmpeg.org/) installato sul tuo sistema e aggiunto al PATH, in quanto è necessario al motore interno per l'elaborazione dei file multimediali).*

---

## ✨ Funzionalità

* **Interfaccia Moderna:** UI pulita e professionale basata su `tkinter` e `ttk` con tema chiaro.
* **Supporto Multimediale:** Compatibile con file video (`.mp4`, `.mkv`, `.mov`, `.avi`) e audio (`.mp3`, `.wav`, `.m4a`, `.flac`).
* **Batch Processing:** Carica più file contemporaneamente e lasciali elaborare in coda in modo completamente automatico.
* **Formati di Output Multipli:** Scegli tra `.txt` (Testo semplice), `.srt` (Sottotitoli standard), `.vtt` (Sottotitoli Web) e `.segments.txt` (Testo con timestamp).
* **Modelli Flessibili:** Scegli la "taglia" del modello AI in base alle tue esigenze (es. `tiny` per la massima velocità, `large-v3` per la massima precisione).
* **Performance Tracking:** Benchmark automatico integrato per stimare l'ETA (Tempo rimanente stimato) in tempo reale.
* **100% Offline:** Tutto il processo di trascrizione avviene localmente sul tuo PC, garantendo la massima sicurezza. I tuoi file non vengono inviati a nessun server esterno.

---

## 💻 Per Sviluppatori: Installazione dal Sorgente

Se preferisci eseguire il programma dal codice sorgente o vuoi contribuire al progetto, segui questi passaggi.

### 🛠️ Requisiti
* **Python 3.8+**
* **FFmpeg** installato e aggiunto al PATH di sistema.

### 🚀 Installazione
1. Clona il repository:
```bash
git clone [https://github.com/TUO_NOME/whisper-studio-gui.git](https://github.com/TUO_NOME/whisper-studio-gui.git)
cd whisper-studio-gui
