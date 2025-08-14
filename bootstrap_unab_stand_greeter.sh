#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# UNAB Stand Greeter - Bootstrap de estructura (macOS)
# Crea carpetas y archivos base sin sobreescribir existentes.
# Uso:
#   chmod +x bootstrap_unab_stand_greeter.sh
#   ./bootstrap_unab_stand_greeter.sh [ruta/proyecto]
# ============================================================

PROJECT_ROOT="${1:-unab-stand-greeter}"

info()  { printf "\033[1;32m[OK]\033[0m %s\n" "$*" ; }
warn()  { printf "\033[1;33m[WARN]\033[0m %s\n" "$*" ; }
note()  { printf "\033[1;34m[INFO]\033[0m %s\n" "$*" ; }

# 1) Validación simple de macOS
if [[ "$(uname -s)" != "Darwin" ]]; then
  warn "Este script fue pensado para macOS (Darwin). Continuando de todos 
modos..."
fi

# 2) Crear carpetas
mkdir -p "$PROJECT_ROOT"/{src,src/assets,src/assets/fonts,scripts}
info "Carpetas creadas en: $PROJECT_ROOT"

# 3) Archivos helper
create_if_missing() {
  local path="$1"
  local content="$2"
  if [[ -f "$path" ]]; then
    warn "Ya existe: $path (no se sobreescribe)"
  else
    printf "%s" "$content" > "$path"
    info "Creado: $path"
  fi
}

# 4) .gitignore
create_if_missing "$PROJECT_ROOT/.gitignore" \
"__pycache__/
.venv/
*.pyc
.DS_Store
*.log
metrics.csv
dist/
build/
*.spec
"

# 5) requirements.txt
create_if_missing "$PROJECT_ROOT/requirements.txt" \
"opencv-python
mediapipe
numpy
Pillow
pyyaml
pyttsx3
qrcode
"

# 6) config.yaml (con valores base)
create_if_missing "$PROJECT_ROOT/config.yaml" \
"camera_index: 0
resolution: [1280, 720]
fullscreen: true
idle_text: \"👋 Acércate y salúdanos\"
prompt_wave_text: \"Levanta tu mano y saluda 👋\"
greeting_texts:
  - \"¡Hola! Bienvenido/a a Ingeniería Civil Informática UNAB\"
  - \"Si quieres saber más, muéstranos 👍\"
qr:
  show_on_thumbs_up: true
  url: \"https://www.unab.cl/admision/ingenieria-civil-informatica\"
tts:
  enabled: true
  rate: 175
  volume: 1.0
cooldowns:
  wave_seconds: 6
  thumbs_seconds: 6
wave:
  window_seconds: 1.5
  amplitude_threshold: 0.04
  min_peaks: 3
thumbs:
  tip_above_wrist_margin: 0.02
  other_fingers_folded: true
brand:
  primary: \"#A00321\"
  text: \"#FFFFFF\"
assets:
  logo_path: \"src/assets/logo_unab.png\"
metrics:
  csv_path: \"metrics.csv\"
keys:
  quit: \"q\"
  mute: \"m\"
  fullscreen_toggle: \"f\"
  reset: \"r\"
"

# 7) README.md
create_if_missing "$PROJECT_ROOT/README.md" \
"# UNAB Stand Greeter (👋 + 👍 + QR)

Aplicación para stand: detecta saludo (ola) y responde con UI + TTS.
Luego espera 👍 para mostrar un QR informativo.

## Estructura
\`\`\`
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
\`\`\`

## Setup rápido
\`\`\`bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
\`\`\`

Controles: \`q\` salir, \`m\` mute, \`f\` fullscreen, \`r\` reset.
"

# 8) Scripts de ejecución
create_if_missing "$PROJECT_ROOT/scripts/run_kiosk.sh" \
"#!/usr/bin/env bash
set -euo pipefail
cd \"\$(dirname \"\${BASH_SOURCE[0]}\")/..\"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt >/dev/null
python src/main.py
"

chmod +x "$PROJECT_ROOT/scripts/run_kiosk.sh"

# 9) Stubs de módulos Python
create_if_missing "$PROJECT_ROOT/src/__init__.py" ""
create_if_missing "$PROJECT_ROOT/src/main.py" \
"#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"Entry point del UNAB Stand Greeter.
Pega aquí el contenido de main.py que generamos en el chat.
\"\"\"
if __name__ == \"__main__\":
    print(\"placeholder main - pega el código real aquí\")"

create_if_missing "$PROJECT_ROOT/src/detector.py" \
"# -*- coding: utf-8 -*-\n\"\"\"Wrapper MediaPipe (stub). Pega aquí el 
detector real.\"\"\"\n"

create_if_missing "$PROJECT_ROOT/src/wave_gesture.py" \
"# -*- coding: utf-8 -*-\n\"\"\"Detector de saludo (stub). Pega aquí el 
código real.\"\"\"\n"

create_if_missing "$PROJECT_ROOT/src/thumbs_gesture.py" \
"# -*- coding: utf-8 -*-\n\"\"\"Detector de pulgar arriba (stub). Pega 
aquí el código real.\"\"\"\n"

create_if_missing "$PROJECT_ROOT/src/ui_renderer.py" \
"# -*- coding: utf-8 -*-\n\"\"\"Renderer UI (stub). Pega aquí el código 
real.\"\"\"\n"

create_if_missing "$PROJECT_ROOT/src/tts.py" \
"# -*- coding: utf-8 -*-\n\"\"\"TTS offline (stub). Pega aquí el código 
real.\"\"\"\n"

create_if_missing "$PROJECT_ROOT/src/metrics.py" \
"# -*- coding: utf-8 -*-\n\"\"\"Métricas a CSV (stub). Pega aquí el código 
real.\"\"\"\n"

# 10) Placeholder del logo para recordar al usuario
LOGO_PATH="$PROJECT_ROOT/src/assets/logo_unab.png"
if [[ ! -f "$LOGO_PATH" ]]; then
  note "Recuerda colocar tu logo en: $LOGO_PATH"
fi

# 11) Vista de la estructura creada
note "Estructura creada:"
( cd "$PROJECT_ROOT" && find . -maxdepth 3 -print | sed 's#^\./##' | sort 
)

info "Listo. Activa el entorno y ejecuta:"
echo "  cd \"$PROJECT_ROOT\""
echo "  python3 -m venv .venv && source .venv/bin/activate"
echo "  pip install -r requirements.txt"
echo "  python src/main.py   # (cuando pegues el código real)"

