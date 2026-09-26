# Componenti di terzi

`WhisperStudio.exe` (release) contiene, oltre al codice di questo repository, i componenti elencati qui sotto con le rispettive licenze. Le versioni sono quelle usate per la release V1.1. Il codice sorgente completo di Whisper Studio e la ricetta di build (`WhisperStudio.spec`) sono in questo repository; i sorgenti dei componenti sono disponibili agli indirizzi indicati.

I modelli Whisper **non** sono inclusi: vengono scaricati al primo utilizzo da Hugging Face (conversioni Systran dei modelli OpenAI Whisper, licenza MIT).

## Motore di trascrizione

| Componente | Versione | Licenza | Sorgente |
|---|---|---|---|
| faster-whisper (incluso il modello VAD Silero `silero_vad_v6.onnx`, MIT) | 1.2.1 | MIT | https://github.com/SYSTRAN/faster-whisper |
| CTranslate2 | 4.8.2 | MIT | https://github.com/OpenNMT/CTranslate2 |
| ↳ `cudnn64_9.dll` (NVIDIA cuDNN, distribuito nel pacchetto ctranslate2) | 9 | NVIDIA cuDNN Software License Agreement | https://developer.nvidia.com/cudnn |
| ↳ `libiomp5md.dll` (Intel OpenMP, distribuito nel pacchetto ctranslate2) | — | Intel Simplified Software License | https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html |
| ONNX Runtime | 1.30.0 | MIT | https://github.com/microsoft/onnxruntime |
| tokenizers | 0.23.2 | Apache-2.0 | https://github.com/huggingface/tokenizers |
| huggingface_hub | 1.33.0 | Apache-2.0 | https://github.com/huggingface/huggingface_hub |
| hf-xet | 1.6.0 | Apache-2.0 | https://github.com/huggingface/xet-core |
| NumPy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://github.com/numpy/numpy |

## Lettura di audio e video

| Componente | Versione | Licenza | Sorgente |
|---|---|---|---|
| PyAV | 18.1.0 | BSD-3-Clause | https://github.com/PyAV-Org/PyAV |
| ↳ FFmpeg (libavcodec, libavformat, libavutil, libavfilter, libavdevice, libswresample, libswscale), compilato con `--enable-version3` | 8.1.2 | LGPL-3.0-or-later | https://ffmpeg.org/download.html |
| ↳ libx264 | — | GPL-2.0-or-later | https://code.videolan.org/videolan/x264 |
| ↳ libx265 | — | GPL-2.0-or-later | https://bitbucket.org/multicoreware/x265_git |
| ↳ LAME (libmp3lame) | — | LGPL-2.0-or-later | https://lame.sourceforge.io |
| ↳ libiconv | — | LGPL-2.1-or-later | https://www.gnu.org/software/libiconv/ |
| ↳ opencore-amr (amrnb, amrwb) | — | Apache-2.0 | https://sourceforge.net/projects/opencore-amr/ |
| ↳ Opus | — | BSD-3-Clause | https://opus-codec.org |
| ↳ libvpx | — | BSD-3-Clause | https://chromium.googlesource.com/webm/libvpx |
| ↳ dav1d | — | BSD-2-Clause | https://code.videolan.org/videolan/dav1d |
| ↳ SVT-AV1 | — | BSD-3-Clause-Clear | https://gitlab.com/AOMediaCodec/SVT-AV1 |
| ↳ libwebp, libsharpyuv | — | BSD-3-Clause | https://chromium.googlesource.com/webm/libwebp |
| ↳ oneVPL (libvpl) | — | MIT | https://github.com/intel/libvpl |
| ↳ zlib | — | Zlib | https://zlib.net |
| ↳ runtime MinGW-w64 (libgcc, libstdc++, libwinpthread) | — | GPL-3.0 con GCC Runtime Library Exception; winpthreads MIT/BSD | https://www.mingw-w64.org |

Le librerie di FFmpeg e dei codec sono caricate dinamicamente (DLL separate estratte all'avvio dell'exe). libx264 e libx265 servono solo per *codificare* video, cosa che Whisper Studio non fa, ma fanno parte della build di FFmpeg fornita da PyAV.

## Altre librerie Python

| Componente | Versione | Licenza | Sorgente |
|---|---|---|---|
| anyio | 4.15.1 | MIT | https://github.com/agronholm/anyio |
| certifi | 2026.7.22 | MPL-2.0 | https://github.com/certifi/python-certifi |
| click | 8.5.0 | BSD-3-Clause | https://github.com/pallets/click |
| filelock | 4.0.3 | MIT | https://github.com/tox-dev/py-filelock |
| flatbuffers | 25.12.19 | Apache-2.0 | https://github.com/google/flatbuffers |
| fsspec | 2026.9.0 | BSD-3-Clause | https://github.com/fsspec/filesystem_spec |
| h11 | 0.16.0 | MIT | https://github.com/python-hyper/h11 |
| httpcore | 1.0.9 | BSD-3-Clause | https://github.com/encode/httpcore |
| httpx | 0.28.1 | BSD-3-Clause | https://github.com/encode/httpx |
| idna | 3.20 | BSD-3-Clause | https://github.com/kjd/idna |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging |
| protobuf | 7.36.2 | BSD-3-Clause | https://github.com/protocolbuffers/protobuf |
| PyYAML | 6.0.3 | MIT | https://github.com/yaml/pyyaml |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | https://github.com/tqdm/tqdm |
| typing_extensions | 4.16.0 | PSF-2.0 | https://github.com/python/typing_extensions |

## Ambiente di esecuzione

| Componente | Versione | Licenza | Sorgente |
|---|---|---|---|
| Python | 3.13 | PSF-2.0 | https://www.python.org |
| Tcl/Tk | 8.6 | Licenza Tcl/Tk (tipo BSD) | https://www.tcl.tk |
| OpenSSL (libcrypto, libssl, incluso in Python) | 3.x | Apache-2.0 | https://www.openssl.org |
| Microsoft Visual C++ Runtime (msvcp140, vcruntime140) | 14 | Microsoft Visual C++ Redistributable | https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist |
| PyInstaller (bootloader) | 6.22.3 | GPL-2.0 con eccezione per il bootloader | https://github.com/pyinstaller/pyinstaller |
