import io, os, hashlib, stat
from pathlib import Path
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from .model import ROOT, SIDE
register_heif_opener()
FORMATS={".jpg",".jpeg",".png",".webp",".heic",".heif"}
PRIVATE={ROOT/name for name in ("data","exports",".venv",".runtime",".models",".uv-cache",".python",".git",".pytest_cache","samples")}
def enumerate_files(root, allow_raw=False):
    def walk(folder):
        try: entries=sorted(folder.iterdir(),key=lambda p:p.name)
        except OSError as e:
            yield folder,"error",e.strerror
            return
        for path in entries:
            if path in PRIVATE: continue
            try:
                kind=path.lstat().st_mode
                if stat.S_ISDIR(kind): yield from walk(path)
                elif not stat.S_ISREG(kind): yield path,"skipped","Link or special file"
                elif path.suffix.lower() not in FORMATS: yield path,"skipped","Unsupported format"
                else: yield path,"pending",None
            except OSError as e: yield path,"error",e.strerror
    if any(root==p or p in root.parents for p in PRIVATE) and not (allow_raw and root.is_relative_to(ROOT/"data"/"raw")): raise ValueError("Select an input folder outside application storage.")
    yield from walk(root)

def read_image(root, path):
    if path.resolve()!=path or not path.is_relative_to(root): raise ValueError("Source path changed or is a link.")
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),"rb") as f:
        raw=f.read(64*1024*1024+1)
    if len(raw)>64*1024*1024: raise ValueError("File exceeds 64 MiB.")
    return hashlib.sha256(raw).hexdigest(),raw

def prepare(raw, digest, data):
    with Image.open(io.BytesIO(raw)) as im:
        if im.width*im.height>50_000_000: raise ValueError("Image exceeds 50 megapixels.")
        im=ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((SIDE,SIDE),Image.Resampling.LANCZOS)
    buf=io.BytesIO()
    im.save(buf,"JPEG",quality=90)
    im.thumbnail((260,300))
    im.save(data/(digest+".jpg"),quality=82)
    return buf.getvalue()
