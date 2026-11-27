"""Download and verify the shared CLIP encoder during setup, never prediction."""
import hashlib
import os
import tempfile
from pathlib import Path
import requests
from .features import ROOT, CLIP_SHA, OCR_BACKEND, portable_ocr
from .learning import default_model_path, load_model
from .schema import NAMES

CLIP_URL = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/6ef1ebc/onnx/vision_model_quantized.onnx"


def prepare():
    target = ROOT/".models/clip/vision.onnx"
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=CLIP_SHA:
        print("Downloading CLIP image encoder (about 89 MB)...",flush=True)
        temp = None
        try:
            with requests.get(CLIP_URL,stream=True,timeout=(20,120)) as response:
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as out:
                    temp = Path(out.name)
                    digest = hashlib.sha256()
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        out.write(chunk)
                        digest.update(chunk)
            if digest.hexdigest()!=CLIP_SHA:
                raise ValueError("CLIP checksum mismatch; retry setup.")
            os.replace(temp,target)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
    model_path = default_model_path()
    checksum = (model_path.parent/"classifier.sha256").read_text().split()[0]
    if hashlib.sha256(model_path.read_bytes()).hexdigest()!=checksum:
        raise ValueError("Fitted classifier checksum mismatch. Restore it from the repository.")
    if OCR_BACKEND=="rapidocr":
        portable_ocr()
    bundle = load_model()
    assert set(bundle["tasks"]["content_type"]["model"].classes_)==set(NAMES)
    print(f"Eight-category model ready with {OCR_BACKEND} OCR. Qwen is not required.")


if __name__=="__main__":
    prepare()
