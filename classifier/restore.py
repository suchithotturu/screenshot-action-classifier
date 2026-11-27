"""Restore an exported snapshot transactionally; existing annotation rows are never replaced."""
import hashlib
import json
from pathlib import Path
from . import dataset, store
from .features import atomic, cache_key
from .schema import Snapshot

def local_file(root,relative):
    p = (root/relative).resolve(strict=True)
    if not p.is_relative_to(root) or not p.is_file():
        raise ValueError("Manifest reference outside the selected dataset folder.")
    return p

def restore(c, path):
    path = Path(path).resolve(strict=True)
    added = existing = 0
    with c:
        c.execute("BEGIN IMMEDIATE")
        for line in path.read_text().splitlines():
            d = Snapshot.model_validate_json(line).model_dump()
            sid,im = d["id"],d["image"]
            raw = local_file(path.parent,im["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=im["sha256"]:
                raise ValueError("Image checksum mismatch for "+sid)
            before = c.execute("SELECT id,sha FROM samples WHERE id=? OR sha=?",(sid,im["sha256"])).fetchall()
            if before:
                if len(before)!=1 or tuple(before[0])!=(sid,im["sha256"]):
                    raise ValueError("ID conflict for "+sid)
                existing+=1
                continue
            # The destination is generated from the verified digest, never an arbitrary manifest path.
            ext = {"JPEG":"jpg","PNG":"png","HEIF":"heif","WEBP":"webp"}.get(im.get("format"),"img")
            im["path"] = "images/"+im["sha256"]+"."+ext
            atomic(dataset.image_path(d),raw)
            ref = d["features"].get("clip",{}).get("cache_ref")
            if ref:
                f = json.loads(local_file(path.parent,ref).read_bytes())
                if f["sha256"]!=im["sha256"]:
                    raise ValueError("Feature checksum reference mismatch")
                if f.get("cache_key")==cache_key(im["sha256"]):
                    atomic(store.DATA/"features"/(f["cache_key"]+".json"),json.dumps(f).encode())
            history = d["annotations"].pop("history",[])
            runs = d["annotations"].pop("qwen_runs",[])
            c.execute("INSERT INTO samples VALUES(?,?,?)",(sid,im["sha256"],json.dumps(d)))
            c.executemany("INSERT INTO revisions VALUES(?,?,?,?,?)",[(sid,r["revision"],r["event"],r["stamp"],json.dumps(r["body"])) for r in history])
            c.executemany("INSERT INTO proposals(id,body) VALUES(?,?)",[(sid,json.dumps(r)) for r in runs])
            s = d["status"]
            if s.get("split"):
                c.execute("INSERT INTO splits VALUES(?,?,?)",(sid,s["split"],s.get("split_group_id") or s["duplicate_group_id"]))
            if locked := d["annotations"].get("frozen_test_targets"):
                c.execute("CREATE TABLE IF NOT EXISTS locked_labels(id TEXT PRIMARY KEY,labels TEXT,status TEXT)")
                c.execute("INSERT INTO locked_labels VALUES(?,?,?)",(sid,json.dumps(locked["labels"]),json.dumps(locked["status"])))
            added+=1
    return {"added":added,"existing":existing}
