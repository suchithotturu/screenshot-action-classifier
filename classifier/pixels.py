"""Deterministic descriptors from image pixels only; no OCR or learned encoder."""
import hashlib
import json
import time
from importlib.metadata import version as package_version
from pathlib import Path
import cv2
import numpy as np
import PIL
from PIL import Image,ImageOps
from skimage.feature import hog

VERSION=("pixels-v1/portrait128x256-gray127/hog9-cell16-block32-stride16-l2hys/"
         "hsv8x3-unpadded/edge4x4-canny100-200-l2/opencv"+
         package_version("opencv-python")+"/skimage"+package_version("scikit-image")+"/PIL"+PIL.__version__)
NAMES={"hog":[f"hog_{i:04d}" for i in range(3780)],
       "color":[f"hsv_{channel}_{i}" for channel in ("h","s","v") for i in range(8)],
       "edges":[f"edge_grid_r{y}_c{x}" for y in range(4) for x in range(4)]}


def blocks_for_mode(mode):
    tokens=set(mode.split("_"))
    return [k for k in NAMES if k in tokens or "visual" in tokens or ("pixels" in tokens and "all" in tokens)]


def describe(im):
    start=time.perf_counter()
    cv2.setNumThreads(1)
    canvas=ImageOps.pad(im.convert("RGB"),(128,256),method=Image.Resampling.BICUBIC,color=(127,127,127))
    gray=cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2GRAY)
    values=hog(gray,orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2),
               block_norm="L2-Hys",transform_sqrt=False,feature_vector=True)
    # Color statistics exclude the artificial letterbox padding.
    thumb=im.convert("RGB");thumb.thumbnail((256,256),Image.Resampling.BICUBIC)
    hsv=cv2.cvtColor(np.asarray(thumb),cv2.COLOR_RGB2HSV)
    color=[]
    for channel,upper in enumerate((180,256,256)):
        counts,_=np.histogram(hsv[:,:,channel],bins=8,range=(0,upper))
        color.extend((counts/counts.sum()).tolist())
    edges=cv2.Canny(gray,100,200,L2gradient=True)>0
    density=[float(edges[y*64:(y+1)*64,x*32:(x+1)*32].mean()) for y in range(4) for x in range(4)]
    result=dict(version=VERSION,hog=values.tolist(),color=color,edges=density,seconds=time.perf_counter()-start)
    for key,names in NAMES.items():
        a=np.asarray(result[key])
        if a.shape!=(len(names),) or not np.isfinite(a).all():
            raise ValueError("Invalid pixel descriptor: "+key)
    return result


def extract_pixels(raw,cache):
    from .features import atomic,decode
    digest=hashlib.sha256(raw).hexdigest()
    key=hashlib.sha256((digest+VERSION).encode()).hexdigest()
    path=Path(cache)/(key+".json")
    if path.exists():
        result=json.loads(path.read_bytes())
        if result.get("version")==VERSION and result.get("sha256")==digest:
            return result
    im,_=decode(raw)
    result=describe(im);result["sha256"]=digest
    atomic(path,json.dumps(result,allow_nan=False).encode())
    return result
