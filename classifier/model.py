import base64, hashlib, json, os, re
from datetime import datetime
from pathlib import Path
from typing import Literal, get_args
from zoneinfo import ZoneInfo
import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError

ROOT = Path(__file__).resolve().parents[1]
TZ = ZoneInfo(os.getenv("DISPLAY_TIMEZONE","America/New_York"))
MODEL = os.getenv("VISION_MODEL","qwen3-vl:4b-instruct")
SIDE = int(os.getenv("IMAGE_SIDE","1200"))
OPTIONS = dict(num_ctx=4096,num_predict=650,temperature=0,seed=11)
PROMPT = (ROOT/"prompt.txt").read_text()
Category = Literal["BUY_COMPARE","VISIT","ATTEND_REGISTER","READ_WATCH_TRY","REFERENCE","OTHER"]
CATEGORIES = list(get_args(Category))
class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    image_type: Literal["screenshot","ordinary_photo","uncertain"]
    category: Category
    title: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    evidence_text: str
    suggested_action: str | None
    entities: list[str]
    visible_url: str | None
    original_date_text: str | None
    date_role: Literal["event","deadline","other","unclear","none"]
    needs_review: bool
    uncertainty_reasons: list[str]

def api(path, body=None, timeout=8):
    with requests.Session() as s:
        s.trust_env = False
        r = s.request("GET" if body is None else "POST","http://127.0.0.1:11434/api/"+path,json=body,timeout=timeout,allow_redirects=False)
        r.raise_for_status()
        return r.json()

def provenance():
    info = next((m for m in api("tags")["models"] if m["name"]==MODEL),None)
    if not info or "vision" not in api("show",{"model":MODEL}).get("capabilities",[]):
        raise ValueError("Local vision model unavailable. Run setup.sh.")
    return dict(model=MODEL,digest=info["digest"],quantization=info["details"]["quantization_level"],runtime=api("version")["version"],prompt="v1-"+hashlib.sha256(PROMPT.encode()).hexdigest()[:12],preprocessing="rgb-exif-jpeg-v1",policy="v2",side=SIDE,options=OPTIONS,timezone=TZ.key)

def policy(p):
    d = p.model_dump()
    def review(reason):
        d["needs_review"]=True
        if reason not in d["uncertainty_reasons"]: d["uncertainty_reasons"].append(reason)
    if p.category in ("REFERENCE","OTHER") or p.image_type!="screenshot": d["suggested_action"]=None
    if p.image_type=="screenshot" and p.category in CATEGORIES[:4]:
        label=p.title if p.title.casefold() in p.evidence_text.casefold() else None
        fallback={"BUY_COMPARE":"Review this product","VISIT":"Consider visiting this place","ATTEND_REGISTER":"Review this signup","READ_WATCH_TRY":"Revisit this recommendation"}
        d["suggested_action"]=fallback[p.category]+(f" — {label}" if label else "")
    if p.image_type=="uncertain": review("Image origin is unclear.")
    if p.image_type!="ordinary_photo" and not p.evidence_text.strip(): review("No readable text was extracted.")
    if p.visible_url and (p.visible_url not in p.evidence_text or any(c.isspace() for c in p.visible_url)):
        d["visible_url"]=None
        review("Link lacks literal supporting text.")
    normalized=None
    if p.original_date_text:
        for fmt in ("%Y-%m-%d","%B %d, %Y","%b %d, %Y","%d %B %Y"):
            try: normalized=datetime.strptime(p.original_date_text.strip(),fmt).date().isoformat()
            except ValueError: continue
            break
        if p.original_date_text not in p.evidence_text: normalized=None
        if not normalized or p.date_role in ("none","unclear"): review("Verify the date and its purpose.")
    d["normalized_date"]=normalized
    d["normalized_due_date"]=normalized if p.date_role=="deadline" else None
    return d

def classify(image):
    schema=Prediction.model_json_schema()
    msg=dict(role="user",content="Return this JSON schema. "+json.dumps(schema),images=[base64.b64encode(image).decode()])
    for attempt in range(2):
        response=api("chat",dict(model=MODEL,messages=[dict(role="system",content=PROMPT),msg],format=schema,options=OPTIONS,stream=False),150)
        try: return policy(Prediction.model_validate_json(response["message"]["content"]))
        except (KeyError,ValidationError):
            if attempt: raise ValueError("Invalid structured output after two attempts.")
            msg["content"]+=" Supply every required field with the exact requested type."
