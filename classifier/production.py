"""Offline folder prediction. This module has no generative inference client."""
import json
import re
import time
from pathlib import Path
from . import store
from .features import extract
from .ingest import enumerate_files, read_image
from .learning import load_model, margins
from .content_rules import match_content
from .schema import NAMES

ROUTING_VERSION = "eight-category-content-v3"

ACTION = {"map_place":("visit_place","Consider visiting this place"),"shopping_product":("consider_purchase","Review this product"),
          "article_reading":("read_or_review","Read or review"),"receipt_confirmation":("keep_reference","Keep as reference"),
          "social_meme":("share","Consider sharing")}
EVIDENCE = {
    "visit_place":r"\b(?:map|street|avenue|road|traffic|directions|route|restaurant|cafe|station|park)\b",
    "consider_purchase":r"(?:[$€£]|\b(?:price|buy|cart|bag|sale|checkout|order)\b)",
    "keep_reference":r"\b(?:receipt|confirmation|invoice|paid|booking|order|ticket|reference)\b",
    "complete_task":r"\b(?:due|submit|deadline|reply|task|appointment)\b",
    "read_or_review":r"\b(?:read|article|news|report|story|guide|instructions|chapter)\b",
    "share":r"\b(?:share|send)\b",
}
def decision(head, f):
    x = head["transform"].transform([f])
    label = str(head["model"].predict(x)[0])
    score = float(margins(head["model"],x)[0])
    return dict(label=label,score=score,threshold=head["threshold"],accepted=score>=head["threshold"])

def predict_image(bundle, raw, assume_screen=False):
    f = extract(raw,store.DATA/"features")
    result = dict(sha256=f["sha256"],ocr=f["ocr"].get("text","")[:400],actions=[],reasons=[],tasks={},
                  route="other",status=f["status"],routing_version=ROUTING_VERSION,category_source="fallback")
    if f["status"]!="ok":
        result["route"] = "Processing error"
        result["reasons"] = [v["error"] for k in ("ocr","clip") if (v:=f[k]).get("error")]
        return result
    heads = bundle["tasks"]
    if "is_screenshot" in heads:
        det = result["tasks"]["is_screenshot"] = decision(heads["is_screenshot"],f)
    if "content_type" not in heads:
        result["reasons"].append("Content classifier unavailable.")
        return result
    content = result["tasks"]["content_type"] = decision(heads["content_type"],f)
    text = f["ocr"]["text"]
    matched = match_content(text)
    trained_categories = set(heads["content_type"]["model"].classes_) if "model" in heads["content_type"] else set()
    if set(NAMES)<=trained_categories and content["accepted"]:
        result.update(route=content["label"],category_source="supervised_model")
    elif matched:
        result.update(route=matched["label"],category_source="ocr_rule",content_rule=matched)
    elif content["accepted"] and text.strip():
        result.update(route=content["label"],category_source="supervised_model")
    else:
        result["reasons"].append("No clear content category.")
    purpose = None
    if "purpose" in heads:
        guess = result["tasks"]["purpose"] = decision(heads["purpose"],f)
        purpose = guess["label"] if guess["accepted"] else None
    result["purpose_status"] = "predicted" if purpose else ("uncertain" if "purpose" in heads else "unavailable")
    action = ACTION.get(result["route"])
    if result["route"]=="chat" and purpose=="act_on_it":
        action = ("complete_task","Review this possible task")
    if purpose=="keep_proof_or_reference":
        action = ("keep_reference","Keep as reference")
    if action:
        # Quote one observed OCR line, without synthesizing entities, deadlines or intent.
        quote = matched["evidence"] if matched and matched["label"]==result["route"] else ""
        quote = quote or next((b["text"] for b in f["ocr"].get("blocks",[]) if b["confidence"]>=0.7 and re.search(EVIDENCE[action[0]],b["text"],re.I)),"")
        result["actions"] = [dict(type=action[0],description=action[1],evidence=quote,date_time=None)]
    return result

def predict_folder(c, folder, assume_screen=False, progress=None, bundle=None):
    bundle = bundle or load_model()
    root = Path(folder).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Select an input folder.")
    files = list(enumerate_files(root))
    items = []
    for i,(p,state,note) in enumerate(files):
        begin = time.perf_counter()
        row = {"path":str(p.relative_to(root))}
        if state!="pending":
            row.update(status=state,route="Skipped" if state=="skipped" else "Processing error",reasons=[note],actions=[])
        else:
            try:
                _,raw = read_image(root,p)
                row.update(predict_image(bundle,raw,assume_screen))
            except (OSError, ValueError,RuntimeError) as e:
                note = "Unreadable or corrupt image." if "BytesIO object" in str(e) else str(e)
                row.update(status="error",route="Processing error",reasons=[note],actions=[])
        row["seconds"] = time.perf_counter()-begin
        items.append(row)
        if progress:
            progress(i+1,len(files))
    body = dict(directory=str(root),training_id=bundle["training_id"],items=items,
                supported=list(bundle["tasks"]["content_type"]["model"].classes_),purpose_available="purpose" in bundle["tasks"],
                routing_version=ROUTING_VERSION,routing_policy="Eight-category supervised content predictions take priority; local OCR rules assist uncertain decisions; remaining uncertainty is miscellaneous.")
    with c:
        run = c.execute("INSERT INTO prediction_runs(created,body) VALUES(?,?)",
                        (store.now(),json.dumps(body))).lastrowid
    return run,body

def save_actions(c,run,index,actions):
    with c:
        c.execute("BEGIN IMMEDIATE")
        doc = json.loads(c.execute("SELECT body FROM prediction_runs WHERE id=?",(run,)).fetchone()[0])
        row = doc["items"][index]
        row.setdefault("original_actions",row["actions"])
        row["actions"] = actions
        c.execute("UPDATE prediction_runs SET body=? WHERE id=?",(json.dumps(doc),run))
