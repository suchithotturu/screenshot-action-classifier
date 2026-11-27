"""The only image-to-feature entry point. No annotation, path, or model suggestion enters X."""
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import tempfile
import time
import threading
from importlib.metadata import version as package_version
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
import PIL
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()
ROOT = Path(__file__).resolve().parents[1]
CLIP_SHA = "583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299"
OCR_BACKEND = os.getenv("CLASSIFIER_OCR", "vision" if platform.system()=="Darwin" else "rapidocr").lower()
if OCR_BACKEND not in ("vision", "rapidocr"):
    raise ValueError("CLASSIFIER_OCR must be vision or rapidocr.")
OCR = (f"Apple Vision rev3 accurate en-US macOS {platform.mac_ver()[0]}" if OCR_BACKEND=="vision"
       else "RapidOCR-ONNXRuntime1.4.4/PP-OCRv4-default-cpu-v1")
VERSION = f"v1.2/letterbox224/tiles1800-overlap160/{OCR}/{CLIP_SHA}/ort{ort.__version__}/PIL{PIL.__version__}"

def compatible_version(version):
    # Cache keys retain the full OS provenance. A different macOS release alone
    # does not change the fitted vector layout or encoder/preprocessing contract.
    return re.sub(r"macOS [^/]+", "macOS", version)

NUMERIC = ["words","characters","digits","dates","times","currencies","urls","task_words","buy_words",
           "ocr_coverage","ocr_blocks","width","height","aspect","empty_text"]
PATTERNS = [r"\d",r"\b(?:\d{1,4}[-/]\d{1,2}(?:[-/]\d{1,4})?|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
            r"\b\d{1,2}:\d{2}\b",r"[$€£¥]|\b(?:usd|eur|gbp)\b",r"https?://|www\.",
            r"\b(?:deadline|due|remind|submit|register|reply|appointment|task)\b",
            r"\b(?:buy|price|sale|cart|order|checkout|offer|purchase)\b"]

def atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    os.replace(temp, path)

def decode(raw):
    with Image.open(io.BytesIO(raw)) as src:
        if src.width * src.height > 50_000_000:
            raise ValueError("Image exceeds 50 megapixels.")
        return ImageOps.exif_transpose(src).convert("RGB"), src.format

@lru_cache(maxsize=4)
def portable_ocr(thread_id=None):
    import rapidocr_onnxruntime
    from rapidocr_onnxruntime import RapidOCR
    if package_version("rapidocr-onnxruntime")!="1.4.4":
        raise RuntimeError("Install the pinned RapidOCR 1.4.4 package using setup.ps1.")
    expected = {
        "ch_PP-OCRv4_det_infer.onnx":"d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9",
        "ch_PP-OCRv4_rec_infer.onnx":"48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b",
        "ch_ppocr_mobile_v2.0_cls_infer.onnx":"e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
    }
    folder = Path(rapidocr_onnxruntime.__file__).parent/"models"
    for name,digest in expected.items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError("RapidOCR checkpoint missing or damaged. Rerun setup.ps1.")
    return RapidOCR(intra_op_num_threads=2,inter_op_num_threads=1)

def ocr_tile(tile):
    if OCR_BACKEND=="rapidocr":
        # RapidOCR's ndarray input uses OpenCV's BGR convention.
        results,_ = portable_ocr(threading.get_ident())(np.asarray(tile)[:,:,::-1].copy())
        lines = []
        for points,text,confidence in results or []:
            xy = np.asarray(points,dtype=float)
            lo,hi = xy.min(axis=0),xy.max(axis=0)
            x,y = np.maximum(lo,0)
            right,bottom = np.minimum(hi,[tile.width,tile.height])
            lines.append(dict(text=text,confidence=float(confidence),
                              box=[float(x/tile.width),float(y/tile.height),
                                   float(max(0,right-x)/tile.width),float(max(0,bottom-y)/tile.height)]))
        return lines
    buf = io.BytesIO()
    tile.save(buf,"PNG")
    cmd = subprocess.run([ROOT/".runtime/astra-ocr"],input=buf.getvalue(),capture_output=True,timeout=120)
    if cmd.returncode:
        raise RuntimeError("Apple Vision OCR failed. "+cmd.stderr.decode(errors="replace")[:160])
    return json.loads(cmd.stdout)

def ocr(im):
    width, height = im.size
    step, extent = 1640, 1800
    lines = []
    # Seam ownership is based on line centre, so identical words elsewhere survive.
    for top in range(0, height, step):
        bottom = min(height, top + extent)
        tile = im.crop((0, top, width, bottom))
        for line in ocr_tile(tile):
            x, y, w, h = line["box"]
            y, h = top + y*tile.height, h*tile.height
            mid = y+h/2
            if (top and mid < top+80) or (bottom < height and mid >= bottom-80):
                continue
            line["box"] = [x, y/height, w, h/height]
            lines.append(line)
        if bottom == height:
            break
    return sorted(lines, key=lambda l: (round(l["box"][1]*height/8), l["box"][0]))

@lru_cache(maxsize=1)
def encoder():
    weights = ROOT/".models/clip/vision.onnx"
    if not weights.is_file() or hashlib.sha256(weights.read_bytes()).hexdigest() != CLIP_SHA:
        raise RuntimeError("CLIP checkpoint missing or damaged. Rerun setup.ps1 on Windows or setup-features.sh on macOS.")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(weights), options, providers=["CPUExecutionProvider"])

def visual(im):
    mean = np.array([0.48145466,0.4578275,0.40821073], dtype=np.float32)
    std = np.array([0.26862954,0.26130258,0.27577711], dtype=np.float32)
    canvas = ImageOps.pad(im, (224,224), method=Image.Resampling.BICUBIC, color=(123,117,104))
    tensor = ((np.asarray(canvas, dtype=np.float32)/255-mean)/std).transpose(2,0,1)[None]
    v = encoder().run(["image_embeds"], {"pixel_values":tensor})[0][0]
    if v.shape != (512,) or not np.isfinite(v).all() or np.linalg.norm(v) == 0:
        raise ValueError("Invalid CLIP projected embedding")
    return (v/np.linalg.norm(v)).tolist()

def cache_key(digest):
    return hashlib.sha256((digest + VERSION).encode()).hexdigest()

def extract(raw, cache, refresh=False):
    digest = hashlib.sha256(raw).hexdigest()
    key = cache_key(digest)
    path = Path(cache)/(key+".json")
    if path.exists() and not refresh:
        record = json.loads(path.read_bytes())
        if record["status"] == "ok":
            return record
    im, _ = decode(raw)
    record = dict(cache_key=key, sha256=digest, version=VERSION, status="ok")
    start = time.perf_counter()
    try:
        blocks = ocr(im)
        record["ocr"] = dict(status="ok", engine=OCR, languages=["en-US"] if OCR_BACKEND=="vision" else ["en","zh"], blocks=blocks,
                             text="\n".join(b["text"] for b in blocks), cache_key=key)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as e:
        record["ocr"] = dict(status="failed", engine=OCR, error=str(e), cache_key=key)
    record["ocr_seconds"] = time.perf_counter()-start
    start = time.perf_counter()
    try:
        record["embedding"] = visual(im)
        record["clip"] = dict(status="ok", encoder="CLIP ViT-B/32 ONNX quantized", sha256=CLIP_SHA,
                              dimension=512, preprocessing="letterbox224", cache_ref=path.name)
    except (OSError, ValueError, RuntimeError) as e:
        record["clip"] = dict(status="failed", error=str(e), cache_ref=path.name)
    record["clip_seconds"] = time.perf_counter()-start
    if record["ocr"]["status"] == record["clip"]["status"] == "ok":
        s = record["ocr"]["text"]
        record["numeric"] = [len(s.split()),len(s),*(len(re.findall(p,s,re.I)) for p in PATTERNS),
                             min(1,sum(b["box"][2]*b["box"][3] for b in blocks)),len(blocks),
                             im.width,im.height,im.width/im.height,int(not s.strip())]
    else:
        record["status"] = "failed"
    atomic(path, json.dumps(record, ensure_ascii=False, allow_nan=False).encode())
    return record
