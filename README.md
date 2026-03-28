## 🔤 Indian Script OCR System v3.0

### AI-Powered Recognition — 20+ Indian Language Scripts

> **Gulzar Group of Institutions, Ludhiana, Punjab — NAAC A+**  
> **SDG 4 – Quality Education | SDG 9 – Industry, Innovation & Infrastructure**

---

## ✅ What's in v3.0

| Feature | Status |
|---------|--------|
| Python 3.8–3.14 compatible | ✅ No TensorFlow needed |
| FastAPI 0.115+ with Pydantic v2 | ✅ Latest |
| OpenCV 4.10+ preprocessing | ✅ CLAHE + Deskew + Adaptive binarize |
| Tesseract 5.x LSTM engine | ✅ OEM 3 |
| 20+ Indian language scripts | ✅ Including Punjabi, Urdu, Kashmiri |
| RTL support (Urdu/Kashmiri/Sindhi) | ✅ |
| GGI-styled web frontend | ✅ Live API status dot |
| Region-level OCR with bounding boxes | ✅ |
| CSV / JSON / TXT export | ✅ |
| Swagger / ReDoc auto-docs | ✅ `/docs` and `/redoc` |
| Single `app.py` — no extra modules | ✅ |

---

## 📁 Project Structure

```
indian-ocr-project/
│
├── backend/
│   ├── app.py              ← FastAPI server (all-in-one, no extra modules)
│   ├── requirements.txt    ← Python dependencies (no TensorFlow)
│   ├── .env.example        ← Copy to .env and configure
│   └── .env                ← Your local config (Tesseract path etc.)
│
├── frontend/
│   └── index.html          ← Complete GGI-styled web UI
│
└── README.md
```

---

## 🚀 Setup — Step by Step

### Step 1 — Install Python Packages

Open PowerShell (Windows) or Terminal (Mac/Linux):

```powershell
# Navigate to your project folder
cd indian-ocr-project

# Install all dependencies (works on Python 3.8 to 3.14)
pip install -r backend/requirements.txt
```

**What gets installed:**

```
fastapi          → Web framework
uvicorn          → ASGI server
opencv-python    → Image processing
pytesseract      → Tesseract wrapper
Pillow           → Image loading
numpy            → Array operations
loguru           → Logging
python-dotenv    → .env file support
pydantic         → Data validation
python-multipart → File upload support
```

---

### Step 2 — Install Tesseract OCR Engine

#### Windows

1. Download installer from:  
   **<https://github.com/UB-Mannheim/tesseract/wiki>**

2. Run installer — select these language packs during install:
   - Hindi (hin)
   - Punjabi (pan)  
   - Urdu (urd)
   - Bengali (ben)
   - Gujarati (guj)
   - Tamil (tam)
   - Telugu (tel)
   - Kannada (kan)
   - Malayalam (mal)
   - (Add more as needed)

3. Create `backend/.env` file:

```
# Copy .env.example to .env then edit:
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

#### Ubuntu / Debian (Linux)

```bash
sudo apt update
sudo apt install -y tesseract-ocr \
  tesseract-ocr-hin tesseract-ocr-pan \
  tesseract-ocr-urd tesseract-ocr-ben \
  tesseract-ocr-guj tesseract-ocr-tam \
  tesseract-ocr-tel tesseract-ocr-kan \
  tesseract-ocr-mal tesseract-ocr-ori \
  tesseract-ocr-mar tesseract-ocr-san \
  tesseract-ocr-asm tesseract-ocr-mai
```

#### macOS

```bash
brew install tesseract tesseract-lang
```

---

### Step 3 — Configure .env

```bash
# Windows
copy backend\.env.example backend\.env

# Linux / macOS
cp backend/.env.example backend/.env
```

Edit `backend/.env`:

```
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
HOST=0.0.0.0
PORT=8000
MAX_FILE_MB=25
LOG_LEVEL=INFO
```

---

### Step 4 — Start the Server

```powershell
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**

```
HH:MM:SS | INFO     | ══════════════════════════════════════════════════
HH:MM:SS | INFO     |   🔤 Indian Script OCR System  v3.0.0
HH:MM:SS | INFO     |   Gulzar Group of Institutions, Ludhiana, Punjab
HH:MM:SS | INFO     |   Python     : 3.14.x
HH:MM:SS | INFO     |   Tesseract  : 5.x.x ✅
HH:MM:SS | INFO     |   Languages  : 22
HH:MM:SS | INFO     |   Docs       : http://localhost:8000/docs
HH:MM:SS | INFO     |   UI         : http://localhost:8000/ui
```

---

### Step 5 — Open the Web UI

Open `frontend/index.html` in your browser.

The **green dot** in the top-right of the navbar = API is connected and Tesseract is ready.

---

## 🌐 API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/` | API info |
| GET | `/health` | Tesseract status + server info |
| GET | `/docs` | Swagger interactive docs |
| GET | `/redoc` | ReDoc docs |
| GET | `/ocr/languages` | All 20+ supported languages |
| POST | `/ocr/recognize` | Upload image → text |
| POST | `/ocr/base64` | Base64 image → text |
| GET | `/ui` | Serve web frontend |

### Test with curl

```bash
# Upload a Hindi document
curl -X POST http://localhost:8000/ocr/recognize \
  -F "file=@document.jpg" \
  -F "language=hindi"

# Auto-detect language
curl -X POST http://localhost:8000/ocr/recognize \
  -F "file=@document.jpg" \
  -F "language=auto"

# Health check
curl http://localhost:8000/health
```

### Test with Python

```python
import requests

# Upload image file
with open("hindi_newspaper.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/ocr/recognize",
        files={"file": ("image.jpg", f, "image/jpeg")},
        data={"language": "hindi", "psm": "6"},
    )

result = response.json()
print("Text:", result["text"])
print("Words:", result["word_count"])
print("Confidence:", result["confidence"], "%")
print("Time:", result["processing_ms"], "ms")
print("Regions:", result["region_count"])
```

---

## 🌍 Supported Languages (20+)

| Code | Language | Script | Region | RTL |
|------|----------|--------|--------|-----|
| `hin` | Hindi | Devanagari | North India | No |
| `pan` | Punjabi | Gurmukhi | Punjab | No |
| `urd` | Urdu | Nastaliq | North India | **Yes** |
| `kas` | Kashmiri | Nastaliq | J&K | **Yes** |
| `snd` | Sindhi | Nastaliq | West India | **Yes** |
| `san` | Sanskrit | Devanagari | Classical | No |
| `mai` | Maithili | Devanagari | Bihar | No |
| `doi` | Dogri | Devanagari | Jammu | No |
| `brx` | Bodo | Devanagari | Assam | No |
| `kok` | Konkani | Devanagari | Goa | No |
| `guj` | Gujarati | Gujarati | Gujarat | No |
| `mar` | Marathi | Devanagari | Maharashtra | No |
| `ben` | Bengali | Bengali | West Bengal | No |
| `asm` | Assamese | Bengali | Assam | No |
| `ori` | Odia | Odia | Odisha | No |
| `mni` | Manipuri | Meitei Mayek | Manipur | No |
| `tam` | Tamil | Tamil | Tamil Nadu | No |
| `tel` | Telugu | Telugu | Andhra | No |
| `kan` | Kannada | Kannada | Karnataka | No |
| `mal` | Malayalam | Malayalam | Kerala | No |
| `eng` | English | Latin | All India | No |

---

## ❓ Troubleshooting

| Problem | Fix |
|---------|-----|
| `TesseractNotFoundError` | Create `backend/.env` and set `TESSERACT_PATH` |
| `ModuleNotFoundError` | Run `pip install -r backend/requirements.txt` |
| Red dot / API offline | Start server: `cd backend && uvicorn app:app --port 8000` |
| Poor OCR accuracy | Use higher resolution scan (300 DPI+). Try different PSM mode. |
| Language not working | Install language pack: `apt install tesseract-ocr-XXX` |
| Port 8000 in use | Use `--port 8080` instead |
| `Permission denied` | Run PowerShell as Administrator |

---

## 📄 License

MIT License — Free to use for educational and research purposes.

**Developed at:** Gulzar Group of Institutions, Ludhiana, Punjab  
**Department:** Computer Science & Engineering  
**Contact:** 9914666777 | [ggi.ac.in](https://ggi.ac.in)

# AI-OCR-Indian-Scripts

AI-based OCR system for recognizing handwritten and printed Indian scripts from images
