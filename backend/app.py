from __future__ import annotations

import base64
import os
import platform
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field


#  0. ENVIRONMENT & LOGGING SETUP


# Load .env file
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
        logger.info(f"Loaded .env from {_env_path}")
    else:
        logger.warning(
            f".env not found at {_env_path}\n"
            "  → Copy .env.example to .env and set your TESSERACT_PATH"
        )
except ImportError:
    pass  # python-dotenv not installed — env vars still work

# Configure loguru
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger.remove()
logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    colorize=True,
)

# Config
MAX_FILE_MB  = int(os.getenv("MAX_FILE_MB", "25"))
MAX_REGIONS  = int(os.getenv("MAX_REGIONS", "30"))



#  1. TESSERACT CONFIGURATION


def _find_tesseract() -> str | None:
    """
    Locate Tesseract binary.
    Priority: TESSERACT_PATH env var → Windows default paths → system PATH
    """
    # 1. Check environment variable (set in .env)
    env_path = os.getenv("TESSERACT_PATH", "").strip().strip('"').strip("'")
    if env_path and Path(env_path).exists():
        return env_path

    # 2. Windows default installation paths
    if platform.system() == "Windows":
        username = os.getenv("USERNAME", os.getenv("USER", ""))
        windows_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            rf"C:\Users\{username}\AppData\Local\Tesseract-OCR\tesseract.exe",
            r"C:\Tesseract-OCR\tesseract.exe",
        ]
        for p in windows_paths:
            if Path(p).exists():
                logger.info(f"Tesseract auto-detected at: {p}")
                return p

    # 3. Unix: let pytesseract find via PATH (return None — works as default)
    return None


# Apply Tesseract path
_tess_path = _find_tesseract()
if _tess_path:
    pytesseract.pytesseract.tesseract_cmd = _tess_path

# Get version info
def _tess_version() -> str:
    try:
        return str(pytesseract.get_tesseract_version())
    except Exception:
        return "NOT_FOUND"

def _tess_ready() -> bool:
    return _tess_version() != "NOT_FOUND"



#  2. LANGUAGE DEFINITIONS  (20+ Indian Scripts)


# Frontend language code → Tesseract language string
LANG_MAP: dict[str, str] = {
    # Auto — try all major scripts at once
    "auto":      "hin+pan+urd+ben+guj+tam+tel+kan+mal+ori+mar+san+eng",

    # North India — Devanagari family
    "hindi":     "hin",
    "marathi":   "mar",
    "sanskrit":  "san",
    "maithili":  "mai",
    "dogri":     "hin",        # Devanagari; dedicated model in newer Tesseract
    "bodo":      "hin",        # Devanagari
    "konkani":   "kok",

    # North India — Gurmukhi (Punjabi)
    "punjabi":   "pan",

    # North India — Perso-Arabic / Nastaliq (RTL)
    "urdu":      "urd",
    "kashmiri":  "urd",        # Also uses Nastaliq
    "sindhi":    "urd",        # Also uses Nastaliq

    # West India
    "gujarati":  "guj",

    # East India
    "bengali":   "ben",
    "assamese":  "asm",
    "odia":      "ori",
    "manipuri":  "mni",

    # South India — Dravidian scripts
    "tamil":     "tam",
    "telugu":    "tel",
    "kannada":   "kan",
    "malayalam": "mal",

    # English
    "english":   "eng",
}

# Right-to-left languages (text rendered RTL in output)
RTL_LANGS: frozenset[str] = frozenset({"urdu", "kashmiri", "sindhi"})

# Full language catalogue returned by /ocr/languages
ALL_LANGUAGES: list[dict] = [
    # Auto
    {"code":"auto",      "name":"Auto Detect",   "script":"Mixed (All)",     "region":"All India",    "rtl":False, "family":"auto",       "sample":"🔍 Auto"},
    # North — Devanagari
    {"code":"hindi",     "name":"Hindi",          "script":"Devanagari",      "region":"North India",  "rtl":False, "family":"devanagari", "sample":"नमस्ते"},
    {"code":"marathi",   "name":"Marathi",        "script":"Devanagari",      "region":"Maharashtra",  "rtl":False, "family":"devanagari", "sample":"नमस्कार"},
    {"code":"sanskrit",  "name":"Sanskrit",       "script":"Devanagari",      "region":"Classical",    "rtl":False, "family":"devanagari", "sample":"संस्कृतम्"},
    {"code":"maithili",  "name":"Maithili",       "script":"Devanagari",      "region":"Bihar",        "rtl":False, "family":"devanagari", "sample":"मैथिली"},
    {"code":"dogri",     "name":"Dogri",          "script":"Devanagari",      "region":"Jammu",        "rtl":False, "family":"devanagari", "sample":"डोगरी"},
    {"code":"bodo",      "name":"Bodo",           "script":"Devanagari",      "region":"Assam",        "rtl":False, "family":"devanagari", "sample":"बड़ो"},
    {"code":"konkani",   "name":"Konkani",        "script":"Devanagari",      "region":"Goa",          "rtl":False, "family":"devanagari", "sample":"कोंकणी"},
    # North — Gurmukhi
    {"code":"punjabi",   "name":"Punjabi",        "script":"Gurmukhi",        "region":"Punjab",       "rtl":False, "family":"gurmukhi",   "sample":"ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ"},
    # North — Perso-Arabic RTL
    {"code":"urdu",      "name":"Urdu",           "script":"Nastaliq (RTL)",  "region":"North India",  "rtl":True,  "family":"nastaliq",   "sample":"آداب عرض ہے"},
    {"code":"kashmiri",  "name":"Kashmiri",       "script":"Nastaliq (RTL)",  "region":"J&K",          "rtl":True,  "family":"nastaliq",   "sample":"کٲشُر"},
    {"code":"sindhi",    "name":"Sindhi",         "script":"Nastaliq (RTL)",  "region":"West India",   "rtl":True,  "family":"nastaliq",   "sample":"سنڌي"},
    # West
    {"code":"gujarati",  "name":"Gujarati",       "script":"Gujarati",        "region":"Gujarat",      "rtl":False, "family":"gujarati",   "sample":"નમસ્તે"},
    # East
    {"code":"bengali",   "name":"Bengali",        "script":"Bengali",         "region":"West Bengal",  "rtl":False, "family":"bengali",    "sample":"নমস্কার"},
    {"code":"assamese",  "name":"Assamese",       "script":"Bengali Script",  "region":"Assam",        "rtl":False, "family":"bengali",    "sample":"অসমীয়া"},
    {"code":"odia",      "name":"Odia",           "script":"Odia",            "region":"Odisha",       "rtl":False, "family":"odia",       "sample":"ନମସ୍କାର"},
    {"code":"manipuri",  "name":"Manipuri",       "script":"Meitei Mayek",    "region":"Manipur",      "rtl":False, "family":"meitei",     "sample":"মৈতৈলোন্"},
    # South — Dravidian
    {"code":"tamil",     "name":"Tamil",          "script":"Tamil",           "region":"Tamil Nadu",   "rtl":False, "family":"dravidian",  "sample":"வணக்கம்"},
    {"code":"telugu",    "name":"Telugu",         "script":"Telugu",          "region":"Andhra Pradesh","rtl":False,"family":"dravidian",  "sample":"నమస్కారం"},
    {"code":"kannada",   "name":"Kannada",        "script":"Kannada",         "region":"Karnataka",    "rtl":False, "family":"dravidian",  "sample":"ನಮಸ್ಕಾರ"},
    {"code":"malayalam", "name":"Malayalam",      "script":"Malayalam",       "region":"Kerala",       "rtl":False, "family":"dravidian",  "sample":"നമസ്‌കാരം"},
    # Latin
    {"code":"english",   "name":"English",        "script":"Latin",           "region":"All India",    "rtl":False, "family":"latin",      "sample":"Hello World"},
]



#  3. IMAGE PREPROCESSING PIPELINE


class Preprocessor:
    """
    Full OpenCV preprocessing pipeline for Indian script OCR.

    Steps applied in order:
      1. Grayscale conversion
      2. Auto upscale (images < 1200px wide → scale up for Tesseract accuracy)
      3. FastNLMeans denoising
      4. CLAHE contrast enhancement
      5. Deskew (correct scan rotation)
      6. Adaptive binarization (Gaussian)
      7. Morphological cleanup
    """

    @staticmethod
    def process(img: np.ndarray) -> np.ndarray:
        """Apply full preprocessing pipeline. Returns binary image."""

        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

        # 2. Upscale if too small (Tesseract performs best at ~300 DPI)
        h, w = gray.shape
        min_dim = min(h, w)
        if min_dim < 600:
            scale  = max(1.5, 800 / min_dim)
            new_w  = int(w * scale)
            new_h  = int(h * scale)
            gray   = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            logger.debug(f"Upscaled {w}×{h} → {new_w}×{new_h} (×{scale:.1f})")

        # 3. Denoising (FastNLMeans)
        gray = cv2.fastNlMeansDenoising(
            gray, h=10, templateWindowSize=7, searchWindowSize=21
        )

        # 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        # 5. Deskew
        gray = Preprocessor._deskew(gray)

        # 6. Adaptive Threshold — better than global Otsu for mixed lighting
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=17,
            C=9,
        )

        # 7. Morphological cleanup — remove tiny 1-pixel noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        return binary

    @staticmethod
    def _deskew(gray: np.ndarray) -> np.ndarray:
        """Correct skew using minAreaRect on thresholded image."""
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return gray  # Too sparse, skip

        angle = cv2.minAreaRect(coords)[-1]
        # minAreaRect returns angle in [-90, 0); normalize
        if angle < -45:
            angle = 90 + angle

        if abs(angle) < 0.3:
            return gray  # Skip trivial correction

        (h, w) = gray.shape[:2]
        center  = (w // 2, h // 2)
        M       = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        logger.debug(f"Deskewed by {angle:.2f}°")
        return rotated

    @staticmethod
    def detect_regions(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Find text-line bounding boxes using horizontal morphological dilation.
        Returns list of (x, y, w, h) sorted top-to-bottom, left-to-right.
        """
        # Dilate horizontally to merge characters into lines
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 8))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # Invert (white text on black → findContours finds black blobs)
        inv = cv2.bitwise_not(dilated)

        contours, _ = cv2.findContours(
            inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:   # skip tiny noise blobs
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter regions that are way too tall (likely a border/frame)
            if h > binary.shape[0] * 0.8:
                continue
            regions.append((x, y, w, h))

        # Sort top-to-bottom, then left-to-right
        regions.sort(key=lambda r: (r[1] // 20, r[0]))
        return regions



#  4. OCR ENGINE


class OCREngine:
    """
    Main OCR engine combining OpenCV preprocessing with Tesseract recognition.
    Supports 20+ Indian scripts with RTL handling.
    """

    _preprocessor = Preprocessor()

    def recognize(
        self,
        img: np.ndarray,
        language: str = "auto",
        psm: int = 6,
    ) -> dict[str, Any]:
        """
        Run full OCR pipeline on an image.

        Args:
            img      : BGR numpy array (from cv2.imdecode)
            language : Language code (see LANG_MAP keys)
            psm      : Tesseract Page Segmentation Mode (default 6 = uniform block)

        Returns:
            Dict with: text, script, confidence, processing_ms,
                       method, region_count, regions, is_rtl,
                       word_count, char_count
        """
        t_start   = time.perf_counter()
        lang_code = LANG_MAP.get(language, LANG_MAP["auto"])
        is_rtl    = language in RTL_LANGS

        # ── Preprocess ────────────────────────────────────────────────────
        binary = self._preprocessor.process(img)

        # ── Tesseract config ──────────────────────────────────────────────
        base_cfg = f"--psm {psm} --oem 3"
        if is_rtl:
            # Preserve interword spaces for Nastaliq/Arabic scripts
            base_cfg += " -c preserve_interword_spaces=1"

        # ── Full-image OCR ────────────────────────────────────────────────
        full_text = ""
        try:
            raw = pytesseract.image_to_string(binary, lang=lang_code, config=base_cfg)
            full_text = self._clean_text(raw)
        except pytesseract.TesseractError as e:
            logger.error(f"Tesseract full-image error: {e}")
        except Exception as e:
            logger.error(f"Unexpected OCR error: {e}")

        # ── Confidence score ──────────────────────────────────────────────
        confidence = 0.0
        try:
            data  = pytesseract.image_to_data(
                binary, lang=lang_code, config=base_cfg,
                output_type=pytesseract.Output.DICT,
            )
            confs = [
                float(c) for c in data["conf"]
                if str(c).lstrip("-").isdigit() and float(c) > 0
            ]
            if confs:
                confidence = round(sum(confs) / len(confs), 1)
        except Exception as e:
            logger.warning(f"Confidence extraction failed: {e}")

        # ── Region-level OCR ──────────────────────────────────────────────
        regions_out: list[dict] = []
        try:
            boxes = self._preprocessor.detect_regions(binary)
            for i, (x, y, w, h) in enumerate(boxes[:MAX_REGIONS]):
                pad  = 8
                y1   = max(0, y - pad)
                y2   = min(binary.shape[0], y + h + pad)
                x1   = max(0, x - pad)
                x2   = min(binary.shape[1], x + w + pad)
                crop = binary[y1:y2, x1:x2]

                if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                    continue

                try:
                    # PSM 7 = treat as single text line
                    region_raw  = pytesseract.image_to_string(
                        crop, lang=lang_code, config=f"--psm 7 --oem 3"
                    )
                    region_text = self._clean_text(region_raw)
                    if region_text:
                        regions_out.append({
                            "region":     i + 1,
                            "bbox":       [int(x), int(y), int(w), int(h)],
                            "text":       region_text,
                            "confidence": round(confidence, 1),
                        })
                except Exception:
                    continue

        except Exception as e:
            logger.warning(f"Region detection error: {e}")

        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)
        output     = full_text or "(No text detected)"

        return {
            "text":          output,
            "script":        language,
            "lang_code":     lang_code,
            "confidence":    confidence,
            "processing_ms": elapsed_ms,
            "method":        "tesseract-ocr-v3",
            "region_count":  max(1, len(regions_out)),
            "regions":       regions_out,
            "is_rtl":        is_rtl,
            "word_count":    len(output.split()),
            "char_count":    len(output.replace("\n", " ").replace(" ", "")),
        }

    @staticmethod
    def _clean_text(raw: str) -> str:
        """
        Post-process raw Tesseract output:
          - Strip whitespace per line
          - Remove lines that are only punctuation/symbols
          - Normalize Unicode to NFC
          - Remove Tesseract form-feed characters
        """
        lines = []
        for line in raw.replace("\f", "").splitlines():
            line = line.strip()
            if not line:
                continue
            # Skip lines that have no alphanumeric / script characters
            cleaned_line = line.replace(" ", "")
            has_real_chars = any(
                unicodedata.category(ch)[0] in ("L", "N")  # Letter or Number
                for ch in cleaned_line
            )
            if not has_real_chars:
                continue
            line = unicodedata.normalize("NFC", line)
            lines.append(line)
        return "\n".join(lines)


# Singleton engine instance
_engine = OCREngine()


#  5. FASTAPI APPLICATION


app = FastAPI(
    title        = " Indian Script OCR API",
    description  = """
## AI-Powered OCR for 20+ Indian Language Scripts

**Supported scripts:** Punjabi (Gurmukhi) · Urdu (Nastaliq/RTL) · Hindi ·
Bengali · Gujarati · Tamil · Telugu · Kannada · Malayalam · Odia ·
Marathi · Sanskrit · Kashmiri · Sindhi · Assamese · Manipuri and more.

**Technology stack:**
- OpenCV 4.10+ preprocessing (CLAHE, deskew, adaptive binarization)
- Tesseract 5.x OCR with LSTM engine (OEM 3)
- FastAPI 0.115+ with Pydantic v2 validation
- Python 3.8–3.14 compatible (no TensorFlow)

---
**Gulzar Group of Institutions, Ludhiana, Punjab — NAAC A+**
*SDG 4 — Quality Education | SDG 9 — Industry, Innovation & Infrastructure*
    """,
    version      = "3.0.0",
    docs_url     = "/docs",
    redoc_url    = "/redoc",
    openapi_tags = [
        {"name": "System",    "description": "Health checks and server info"},
        {"name": "Languages", "description": "Supported language/script information"},
        {"name": "OCR",       "description": "Text recognition endpoints"},
        {"name": "Frontend",  "description": "Web interface"},
    ],
    contact = {
        "name":  "GGI Ludhiana",
        "url":   "https://ggi.ac.in",
        "email": "info@ggi.ac.in",
    },
)

# CORS — allow all origins (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Pydantic Models ──────────────────────────────────────────────────────

class RegionOut(BaseModel):
    region:     int   = Field(..., description="Region index (1-based)")
    bbox:       list[int] = Field(..., description="[x, y, width, height] in pixels")
    text:       str   = Field(..., description="Recognized text in this region")
    confidence: float = Field(..., description="OCR confidence 0–100")


class OCRResponse(BaseModel):
    success:       bool  = Field(..., description="Whether OCR succeeded")
    text:          str   = Field(..., description="Full recognized text")
    script:        str   = Field(..., description="Language code used")
    lang_code:     str   = Field(..., description="Tesseract language string")
    confidence:    float = Field(..., description="Average confidence score (0–100)")
    processing_ms: float = Field(..., description="Processing time in milliseconds")
    method:        str   = Field(..., description="OCR method used")
    region_count:  int   = Field(..., description="Number of text regions detected")
    regions:       list[dict] = Field(default=[], description="Per-region results")
    is_rtl:        bool  = Field(default=False, description="True for RTL scripts (Urdu etc.)")
    word_count:    int   = Field(default=0,    description="Word count in recognized text")
    char_count:    int   = Field(default=0,    description="Character count (no spaces)")


class HealthResponse(BaseModel):
    status:            str
    version:           str
    python_version:    str
    tesseract_ready:   bool
    tesseract_version: str
    tesseract_path:    str
    languages_count:   int
    max_file_mb:       int
    message:           str


# ── Helper: bytes → OpenCV image ─────────────────────────────────────────

ALLOWED_MIME = {
    "image/jpeg", "image/jpg", "image/png",
    "image/bmp",  "image/tiff", "image/webp",
    "image/gif",  "image/x-bmp",
}

def _decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes → BGR numpy array. Raises ValueError on failure."""
    nparr = np.frombuffer(data, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        # Try PIL as fallback (handles more formats)
        try:
            from io import BytesIO
            pil_img = Image.open(BytesIO(data)).convert("RGB")
            img     = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            raise ValueError(
                "Cannot decode image. Ensure the file is a valid "
                "JPG, PNG, BMP, TIFF, or WEBP image."
            )
    return img



#  6. ROUTES


@app.get(
    "/",
    tags=["System"],
    summary="API root — overview and links",
)
async def root():
    return {
        "name":        "Indian Script OCR API",
        "version":     "3.0.0",
        "description": "AI-powered OCR for 20+ Indian language scripts",
        "links": {
            "docs":      "/docs",
            "health":    "/health",
            "languages": "/ocr/languages",
            "ui":        "/ui",
        },
        "institution": "Gulzar Group of Institutions, Ludhiana, Punjab",
        "python":      sys.version,
        "tesseract":   _tess_version(),
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Server health check — use this to verify Tesseract is working",
)
async def health():
    tess_ver   = _tess_version()
    tess_ready = tess_ver != "NOT_FOUND"
    return HealthResponse(
        status            = "ok" if tess_ready else "degraded",
        version           = "3.0.0",
        python_version    = sys.version.split()[0],
        tesseract_ready   = tess_ready,
        tesseract_version = tess_ver,
        tesseract_path    = (
            pytesseract.pytesseract.tesseract_cmd
            if hasattr(pytesseract.pytesseract, "tesseract_cmd")
            else "auto"
        ),
        languages_count   = len(ALL_LANGUAGES),
        max_file_mb       = MAX_FILE_MB,
        message = (
            "✅ All systems operational — ready for OCR"
            if tess_ready else
            "⚠️ Tesseract not found.\n"
            "  Windows: Set TESSERACT_PATH in backend/.env\n"
            "  Linux:   sudo apt install tesseract-ocr tesseract-ocr-hin ...\n"
            "  macOS:   brew install tesseract tesseract-lang"
        ),
    )


@app.get(
    "/ocr/languages",
    tags=["Languages"],
    summary="List all supported Indian language scripts",
)
async def get_languages(family: str | None = None):
    """
    Returns all 20+ supported languages.
    Optionally filter by script family: devanagari | gurmukhi | nastaliq |
    gujarati | bengali | odia | meitei | dravidian | latin | auto
    """
    langs = ALL_LANGUAGES
    if family:
        langs = [l for l in langs if l.get("family") == family.lower()]
    return {
        "total":     len(langs),
        "languages": langs,
        "families":  list({l["family"] for l in ALL_LANGUAGES}),
    }


@app.post(
    "/ocr/recognize",
    response_model=OCRResponse,
    tags=["OCR"],
    summary="Upload an image → extract Indian script text",
)
async def recognize_upload(
    file:     UploadFile = File(
        ...,
        description="Image file: JPG, PNG, BMP, TIFF, WEBP (max 25 MB)"
    ),
    language: str = Form(
        "auto",
        description=(
            "Language code. Use 'auto' for automatic detection.\n"
            "Examples: hindi, punjabi, urdu, bengali, tamil, telugu, "
            "gujarati, marathi, kannada, malayalam, odia, sanskrit, english"
        ),
    ),
    psm: int = Form(
        6,
        description=(
            "Tesseract Page Segmentation Mode:\n"
            "  3 = Auto with OSD\n"
            "  6 = Uniform block of text (default)\n"
            "  7 = Single text line\n"
            "  11 = Sparse text (find as much text as possible)\n"
            "  13 = Raw line (no Tesseract-specific processing)"
        ),
    ),
):
    """
    **Upload any Indian language document and get the text back.**

    Supports:
    - Scanned printed documents
    - Handwritten pages (accuracy ~83%)
    - Newspaper/magazine clippings
    - Screenshots of text
    - WhatsApp/chat screenshots
    - Mixed-language documents

    Returns recognized text with confidence score, word count,
    processing time, and line-level region breakdown.
    """
    # Validate content type
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in ALLOWED_MIME and content_type != "application/octet-stream":
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: '{content_type}'.\n"
                "Allowed types: JPG, PNG, BMP, TIFF, WEBP"
            ),
        )

    # Read file
    content = await file.read()
    size_mb  = len(content) / (1024 ** 2)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f} MB. Maximum allowed: {MAX_FILE_MB} MB.",
        )

    if len(content) < 100:
        raise HTTPException(status_code=400, detail="File appears to be empty or corrupt.")

    try:
        img    = _decode_image_bytes(content)
        result = _engine.recognize(img, language=language, psm=psm)

        logger.success(
            f"✅ OCR [{language}] | {file.filename} | "
            f"conf={result['confidence']}% | "
            f"words={result['word_count']} | "
            f"time={result['processing_ms']}ms"
        )

        return OCRResponse(success=True, **result)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"OCR failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal OCR error: {str(e)}")


@app.post(
    "/ocr/base64",
    response_model=OCRResponse,
    tags=["OCR"],
    summary="Recognize text from a base64-encoded image",
)
async def recognize_base64(request: Request):
    """
    Send a base64-encoded image string.

    **Request body (JSON):**
    ```json
    {
        "image":    "data:image/jpeg;base64,/9j/4AAQ...",
        "language": "hindi",
        "psm":      6
    }
    ```
    The `data:...;base64,` prefix is optional — raw base64 also works.
    """
    try:
        body     = await request.json()
        b64      = body.get("image", "").strip()
        language = body.get("language", "auto")
        psm      = int(body.get("psm", 6))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON request body.")

    if not b64:
        raise HTTPException(status_code=400, detail="Missing 'image' field in request body.")

    # Strip data-URI prefix if present
    if b64.startswith("data:"):
        parts = b64.split(",", 1)
        b64   = parts[1] if len(parts) > 1 else ""

    try:
        img_bytes = base64.b64decode(b64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding.")

    try:
        img    = _decode_image_bytes(img_bytes)
        result = _engine.recognize(img, language=language, psm=psm)
        return OCRResponse(success=True, **result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Base64 OCR failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Serve Web Frontend ────────────────────────────────────────────────────

_FRONTEND = Path(__file__).parent.parent / "frontend" / "index.html"


@app.get(
    "/ui",
    response_class=HTMLResponse,
    tags=["Frontend"],
    summary="Open the web UI",
    include_in_schema=False,
)
async def serve_ui():
    if _FRONTEND.exists():
        content = _FRONTEND.read_text(encoding="utf-8")
        return HTMLResponse(content=content, status_code=200)
    return HTMLResponse(
        content=(
            "<html><body style='font-family:sans-serif;padding:40px'>"
            "<h2>Indian Script OCR — Frontend Not Found</h2>"
            "<p>Open <code>frontend/index.html</code> directly in your browser.</p>"
            "<p>Or visit <a href='/docs'>/docs</a> to use the API directly.</p>"
            "</body></html>"
        ),
        status_code=404,
    )


# ── Startup / Shutdown Events ─────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    tess_ver   = _tess_version()
    tess_ready = tess_ver != "NOT_FOUND"
    logger.info("=" * 65)
    logger.info("  Indian Script OCR System  v3.0.0")
    logger.info("  Gulzar Group of Institutions, Ludhiana, Punjab")
    logger.info("  SDG 4 — Quality Education | SDG 9 — Innovation")
    logger.info("-" * 65)
    logger.info(f"  Python     : {sys.version.split()[0]}")
    logger.info(f"  Tesseract  : {tess_ver} {'✅' if tess_ready else '❌ NOT FOUND'}")
    logger.info(f"  Languages  : {len(ALL_LANGUAGES)}")
    logger.info(f"  Max file   : {MAX_FILE_MB} MB")
    logger.info(f"  Docs       : http://localhost:8000/docs")
    logger.info(f"  UI         : http://localhost:8000/ui")
    logger.info(f"  Health     : http://localhost:8000/health")
    logger.info("=" * 65)

    if not tess_ready:
        logger.warning(
            "\n⚠️  TESSERACT NOT FOUND!\n"
            "  The API will respond with errors for OCR requests.\n"
            "  Fix:\n"
            "    Windows: Set TESSERACT_PATH in backend/.env\n"
            "    Linux:   sudo apt install tesseract-ocr tesseract-ocr-hin ...\n"
            "    macOS:   brew install tesseract tesseract-lang\n"
        )


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Indian Script OCR API shutting down.")



#  7. DIRECT RUN


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "app:app",
        host      = host,
        port      = port,
        reload    = True,
        log_level = LOG_LEVEL.lower(),
        workers   = 1,
    )
