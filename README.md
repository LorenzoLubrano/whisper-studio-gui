**Whisper Studio**🎙️

Whisper Studio è un'interfaccia grafica moderna ed elegante per faster-whisper. Permette di trascrivere e tradurre file audio/video localmente con elevata velocità e precisione, sfruttando l'accelerazione GPU se disponibile.

✨ **Funzionalità**

**Interfaccia Moderna**: UI pulita basata su tkinter e ttk con tema chiaro professionale.

**Multimediale**: Supporta file video (mp4, mkv, mov, avi) e audio (mp3, wav, m4a, flac).

**Batch Processing**: Carica più file e lasciali elaborare in coda.

**Formati Output**:

.txt (Testo semplice)

.srt (Sottotitoli standard)

.vtt (Sottotitoli Web)

.segments.txt (Testo con timestamp)

**Modelli Flessibili**: Scegli tra tiny, base, small, medium, large-v3.

**Performance**: Benchmark automatico per stimare l'ETA (Tempo rimanente stimato).

**Offline**: Tutto gira in locale sul tuo PC, garantendo la privacy dei dati.

🛠️ **Requisiti**

Python 3.8+

FFmpeg installato e aggiunto al PATH di sistema.

🚀 **Installazione**

Clona il repository:

git clone [https://github.com/TUO_NOME/whisper-studio-gui.git](https://github.com/TUO_NOME/whisper-studio-gui.git)
cd whisper-studio-gui


**Installa le dipendenze:**
pip install -r requirements.txt


(Opzionale) Se hai una scheda video NVIDIA, installa le librerie CUDA per faster-whisper e torch.

▶️ **Utilizzo**

Avvia l'applicazione con:

python FasterWhisper_GUI_Modern.py


Clicca su + Aggiungi Media per selezionare i file.

Seleziona il Modello (es. small per velocità, large-v3 per precisione).

Scegli la Lingua e il Task (Trascrivi o Traduci in EN).

Premi Avvia Elaborazione.

📄 **Licenza**

Questo progetto è distribuito sotto licenza MIT.
