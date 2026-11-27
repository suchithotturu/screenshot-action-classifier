"""Transactional annotation library stored alongside the original inbox database."""
import csv
import hashlib
import io
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from . import store
from .features import atomic, cache_key, decode
from .ingest import enumerate_files, read_image
from .model import ROOT
from .schema import Draft, Snapshot, TARGETS

def connect():
    data = store.DATA
    data.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(data/"annotations.db", timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    c.execute("PRAGMA journal_mode=WAL")
    c.executescript("""
    CREATE TABLE IF NOT EXISTS samples(id TEXT PRIMARY KEY, sha TEXT UNIQUE NOT NULL, doc TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS revisions(id TEXT REFERENCES samples, revision INTEGER, event TEXT, stamp TEXT,
        body TEXT, PRIMARY KEY(id,revision));
    CREATE TABLE IF NOT EXISTS proposals(seq INTEGER PRIMARY KEY,id TEXT REFERENCES samples,body TEXT);
    CREATE TABLE IF NOT EXISTS drops(source TEXT,path TEXT,reason TEXT,PRIMARY KEY(source,path,reason));
    CREATE TABLE IF NOT EXISTS splits(id TEXT PRIMARY KEY REFERENCES samples,partition TEXT NOT NULL, group_id TEXT);
    CREATE TABLE IF NOT EXISTS prediction_runs(id INTEGER PRIMARY KEY,created TEXT,body TEXT);
    """)
    return c

def all_docs(c):
    return [json.loads(r[0]) for r in c.execute("SELECT doc FROM samples ORDER BY rowid")]
def get(c, sid):
    r = c.execute("SELECT doc FROM samples WHERE id=?", (sid,)).fetchone()
    if r is None:
        raise KeyError(sid)
    return json.loads(r[0])
def put(c, doc):
    c.execute("UPDATE samples SET doc=? WHERE id=?", (json.dumps(doc),doc["id"]))
def source_labels(origins):
    labels = dict.fromkeys(TARGETS)
    provenance = {k:{"origin":"unknown"} for k in TARGETS}
    for k in ("content_type","is_screenshot"):
        supplied = {s[k] for s in origins if s.get(k) is not None}
        if len(supplied) == 1:
            labels[k] = supplied.pop()
            provenance[k] = {"origin":"source"}
    return {**labels,"target_provenance":provenance}
def image_path(doc):
    return store.DATA/"aggregated"/doc["image"]["path"]

def ingest_one(c, raw, src):
    if c.in_transaction:
        return _ingest(c,raw,src)
    with c:
        c.execute("BEGIN IMMEDIATE")
        return _ingest(c,raw,src)

def _ingest(c, raw, src):
    sha = hashlib.sha256(raw).hexdigest()
    existing = c.execute("SELECT doc FROM samples WHERE sha=?", (sha,)).fetchone()
    if existing:
        doc = json.loads(existing[0])
        origins = doc["source"]["origins"]
        if src not in origins:
            origins.append(src)
            if doc["annotations"]["review"]["state"] != "reviewed":
                doc["labels"] = source_labels(origins)
            vals = [{s[k] for s in origins if s.get(k) is not None} for k in ("is_screenshot","content_type")]
            doc["status"]["source_conflict"] = any(len(v)>1 for v in vals)
            doc["status"]["fixture"] |= src["dataset"] == "fixture"
            put(c,doc)
        return doc["id"],False
    im, fmt = decode(raw)
    sid = src.get("id") or src["dataset"]+"_"+sha[:20]
    ext = {"JPEG":"jpg","PNG":"png","HEIF":"heif","WEBP":"webp"}.get(fmt, "img")
    path = Path("images")/(sha+"."+ext)
    small = im.convert("L").resize((9,8),Image.Resampling.LANCZOS)
    a = np.asarray(small)
    dhash = np.packbits(a[:,1:]>a[:,:-1]).tobytes().hex()
    thumb = np.asarray(im.resize((8,8),Image.Resampling.BILINEAR)).tobytes().hex()
    doc = dict(schema_version=2,id=sid,source={"origins":[src]},
               image=dict(sha256=sha,path=str(path),format=fmt,width=im.width,height=im.height,
                          dhash=dhash,color_signature=thumb,transform="original bytes; EXIF orientation on decode"),
               labels=source_labels([src]),status=dict(duplicate_group_id=sid,split=None,fixture=src["dataset"]=="fixture",source_conflict=False),
               annotations=dict(qwen=None,human=None,draft=None,review=dict(state="pending",revision=0,
                                mode=None,exposed=False,reviewer=None,role=None,updated=None)),features={})
    Snapshot.model_validate(doc)
    atomic(image_path(doc),raw)
    c.execute("INSERT INTO samples VALUES(?,?,?)",(sid,sha,json.dumps(doc)))
    return sid,True

def ingest_folder(c, folder, source, owner=None, session="", progress=None):
    if source not in ("personal","public","fixture","memes","masc","screen2words"):
        raise ValueError("Choose a known source.")
    if owner not in (None,"A","B","C","D") or (owner and source!="personal"):
        raise ValueError("Select owner A–D for personal images.")
    root = Path(folder).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Select a folder.")
    paths = list(enumerate_files(root,allow_raw=True))
    counts = Counter()
    for i,(p,state,note) in enumerate(paths):
        name = str(p.relative_to(root))
        try:
            if state != "pending":
                raise ValueError(note)
            _,raw = read_image(root,p)
            fixture = root == ROOT/"demo" or (ROOT/"demo") in p.parents
            src = dict(dataset="fixture" if fixture else source, relative_name=name,owner=owner,
                       session=session,license="private" if source=="personal" else "unverified")
            _,added = ingest_one(c,raw,src)
            counts["added" if added else "existing"] += 1
        except (ValueError,OSError,sqlite3.IntegrityError) as e:
            reason = "Unreadable or corrupt image" if isinstance(e,Image.UnidentifiedImageError) else str(e)
            with c:
                c.execute("INSERT OR IGNORE INTO drops VALUES(?,?,?)",(source,name,reason))
            counts["skipped" if state=="skipped" else "errors"] += 1
        if progress:
            progress(i+1,len(paths))
    return dict(counts)

def is_owner(doc, reviewer, role):
    return role=="owner" and reviewer in ("A","B","C","D") and any(
        s.get("owner")==reviewer and s["dataset"]=="personal" for s in doc["source"]["origins"])

def current_draft(doc,blind=False):
    a = doc["annotations"]
    if blind and a["review"]["exposed"]:
        raise ValueError("Assisted material was previously visible for this record.")
    if a["draft"] is not None:
        return Draft.model_validate(a["draft"])
    if blind:
        return Draft()
    q = a["qwen"] or {}
    fields = dict(q.get("fields") or {})
    # Plausible Qwen intent must be actively supplied by the owner.
    fields["purpose"] = None
    for k in TARGETS:
        if doc["labels"].get(k) is not None:
            fields[k] = doc["labels"][k]
    return Draft.model_validate(fields)

def reserve_blind(c,sid):
    with c:
        c.execute("BEGIN IMMEDIATE")
        doc = get(c,sid)
        r = doc["annotations"]["review"]
        if r["exposed"]:
            raise ValueError("Record already exposed in another window.")
        r["mode"] = "blind"
        put(c,doc)

def expose(c,sid):
    with c:
        c.execute("BEGIN IMMEDIATE")
        doc = get(c,sid)
        review = doc["annotations"]["review"]
        if review["mode"]=="blind" and review["state"]!="reviewed":
            raise ValueError("Finish the locked blind review first.")
        review["exposed"] = True
        put(c,doc)

def edit(c, sid, value, event, reviewer, role, blind, expected_revision):
    """Caller owns transaction so a batch commits entirely or rolls back."""
    doc = get(c,sid)
    a = doc["annotations"]
    r = a["review"]
    if r["revision"] != expected_revision:
        raise ValueError("This record changed in another window. Reload the batch.")
    if blind and r["exposed"]:
        raise ValueError("Record already exposed to assisted labels.")
    if not blind and r["mode"]=="blind" and r["state"]!="reviewed":
        raise ValueError("Continue in blind mode until approval.")
    if event not in ("save","approve","undo","skip","reset"):
        raise ValueError("Invalid review event")
    if event=="reset" and blind:
        raise ValueError("Suggestions are hidden for blind review.")
    if r["state"]=="reviewed" and event!="undo":
        raise ValueError("Undo approval first.")
    if event=="undo" and r["state"]!="reviewed":
        raise ValueError("This record has no approval to undo.")
    if role not in ("owner","reviewer") or not reviewer.strip():
        raise ValueError("Identify the reviewer.")
    d = Draft.model_validate(value).model_dump()
    owner = is_owner(doc,reviewer,role)
    previous = doc["labels"]
    if not owner:
        for k in ("purpose","still_useful"):
            if d[k] is not None and d[k]!=previous[k]:
                raise ValueError("Only the matching owner may confirm intent or usefulness.")
            d[k] = previous[k] if previous["target_provenance"][k]["origin"]=="owner" else None
    if d["is_screenshot"] is not True:
        for k in ("purpose","still_useful"):
            d[k] = None
    stamp = store.now()
    if event=="approve":
        labels = {k:d[k] for k in TARGETS}
        origins = {k:dict(origin="unknown") if d[k] is None else
                   dict(origin="owner" if owner else "human",reviewer=reviewer,role=role,at=stamp) for k in TARGETS}
        for k in ("purpose","still_useful"):
            if not owner and d[k] is not None:
                origins[k] = previous["target_provenance"][k]
        doc["labels"] = dict(**labels,target_provenance=origins)
        a["human"] = {k:v for k,v in d.items() if k not in TARGETS}
    if event=="undo":
        doc["labels"] = source_labels(doc["source"]["origins"])
        a["human"] = None
    if event=="reset":
        a["draft"] = None
        fields = dict((a["qwen"] or {}).get("fields") or {})
        fields["purpose"] = None
        d = Draft.model_validate(fields).model_dump()
    a["draft"] = d
    r.update(state={"approve":"reviewed","skip":"skipped"}.get(event,"draft"),reviewer=reviewer,role=role,
             revision=r["revision"]+1,mode="blind" if blind else "assisted",updated=stamp,exposed=r["exposed"] or not blind)
    Snapshot.model_validate(doc)
    put(c,doc)
    c.execute("INSERT INTO revisions VALUES(?,?,?,?,?)",(sid,r["revision"],event,stamp,json.dumps({"labels":doc["labels"],"annotations":a})))
    return r["revision"]

def batch_edit(c, entries):
    with c:
        c.execute("BEGIN IMMEDIATE")
        return [edit(c,*args) for args in entries]

def export(c):
    """Canonical pixels are hard-linked into a portable export with relative image paths."""
    dest = store.DATA/"exports"
    aggregate = store.DATA/"aggregated"
    lines = []
    with c:
        c.execute("BEGIN IMMEDIATE")
        from .splits import group_duplicates
        docs = all_docs(c)
        group_duplicates(c,docs)
        for doc in docs:
            p = image_path(doc)
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=doc["image"]["sha256"]:
                raise ValueError("Missing or changed image for "+doc["id"])
            cache = store.DATA/"features"/(cache_key(doc["image"]["sha256"])+".json")
            if cache.exists():
                f = json.loads(cache.read_bytes())
                doc["features"] = {"ocr":f["ocr"],"clip":dict(f["clip"])}
                ref = Path("features")/cache.name
                atomic(dest/ref,cache.read_bytes())
                aggregate_ref = aggregate/ref
                aggregate_ref.parent.mkdir(parents=True,exist_ok=True)
                if not aggregate_ref.exists():
                    try:
                        os.link(cache,aggregate_ref)
                    except FileExistsError:
                        pass
                elif not os.path.samefile(cache,aggregate_ref):
                    atomic(aggregate_ref,cache.read_bytes())
                doc["features"]["clip"]["cache_ref"] = str(ref)
            else:
                doc["features"] = {k:{"status":"pending"} for k in ("ocr","clip")}
            split = c.execute("SELECT * FROM splits WHERE id=?",(doc["id"],)).fetchone()
            if split:
                doc["status"].update(split=split["partition"],split_group_id=split["group_id"])
            if split and split["partition"].startswith("test") and c.execute("SELECT 1 FROM sqlite_master WHERE name='locked_labels'").fetchone():
                frozen = c.execute("SELECT labels,status FROM locked_labels WHERE id=?",(doc["id"],)).fetchone()
                if frozen:
                    doc["annotations"]["frozen_test_targets"] = {"labels":json.loads(frozen[0]),"status":json.loads(frozen[1])}
            doc["annotations"]["history"] = [dict(revision=r[0],event=r[1],stamp=r[2],body=json.loads(r[3])) for r in c.execute(
                "SELECT revision,event,stamp,body FROM revisions WHERE id=? ORDER BY revision",(doc["id"],))]
            doc["annotations"]["qwen_runs"] = [json.loads(r[0]) for r in c.execute("SELECT body FROM proposals WHERE id=? ORDER BY seq",(doc["id"],))]
            validated = Snapshot.model_validate(doc)
            out = dest/doc["image"]["path"]
            out.parent.mkdir(parents=True,exist_ok=True)
            if not out.exists():
                try:
                    os.link(p,out)
                except FileExistsError:
                    pass
                except OSError:
                    atomic(out,p.read_bytes())
            lines.append(validated.model_dump_json())
        drops = list(c.execute("SELECT * FROM drops ORDER BY source,path"))
        if c.execute("SELECT 1 FROM sqlite_master WHERE name='legacy_drops'").fetchone():
            dropped = [r[0] for r in c.execute("SELECT body FROM legacy_drops ORDER BY id")]
            atomic(dest/"legacy-drops.jsonl","".join(r+"\n" for r in dropped).encode())
    payload = "".join(s+"\n" for s in lines).encode()
    atomic(dest/"dataset.jsonl",payload)
    atomic(aggregate/"manifest.jsonl",payload)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["source","path","reason"])
    writer.writerows(drops)
    atomic(aggregate/"drops.csv",buf.getvalue().encode())
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id",*TARGETS,"provenance","split","review_state"])
    for line in lines:
        doc = json.loads(line)
        writer.writerow([doc["id"],*(doc["labels"][k] for k in TARGETS),json.dumps(doc["labels"]["target_provenance"]),
                         doc["status"]["split"],doc["annotations"]["review"]["state"]])
    atomic(dest/"dataset.csv",buf.getvalue().encode())
    return dest/"dataset.jsonl"
