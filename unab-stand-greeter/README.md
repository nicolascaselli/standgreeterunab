# UNAB Stand Greeter (👋 + 👍 + QR)

Aplicación para stand: detecta saludo (ola) y responde con UI + TTS.
Luego espera 👍 para mostrar un QR informativo.

## Estructura
```
unab-stand-greeter/
├─ src/
│  ├─ main.py
│  ├─ detector.py
│  ├─ wave_gesture.py
│  ├─ thumbs_gesture.py
│  ├─ ui_renderer.py
│  ├─ tts.py
│  ├─ metrics.py
│  └─ assets/
│     ├─ logo_unab.png
│     └─ fonts/
├─ config.yaml
├─ requirements.txt
├─ scripts/
│  └─ run_kiosk.sh
└─ .gitignore
```

## Setup rápido
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

Controles: `q` salir, `m` mute, `f` fullscreen, `r` reset.
