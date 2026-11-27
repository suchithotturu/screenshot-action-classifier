"""Conservative bridge for older master JSONL records. Original metadata survives intact."""
import hashlib
import json
from pathlib import Path
from . import dataset
from .restore import local_file
from .schema import Labels, NAMES, TARGETS

def import_master(c, manifest, images=None):
    path = Path(manifest).resolve(strict=True)
    root = Path(images).resolve(strict=True) if images else path.parent
    result = {"added":0,"existing":0,"metadata_only_drops":0}
    for number,line in enumerate(path.read_text().splitlines(),1):
        record = json.loads(line)
        sid = record["id"]
        source,image,labels,status = [record.get(k) or {} for k in ("source","image","labels","status")]
        if not all(isinstance(v,dict) for v in (source,image,labels,status)):
            raise ValueError(f"Expected source/image/labels/status objects on line {number}")
        name = source.get("dataset") or source.get("name") or sid.split("_")[0].split("-")[0]
        src = {"id":sid,"dataset":name,"license":source.get("license") or ("private" if name=="personal" else "unverified"),
               "legacy_record":record}
        for k in ("owner","rico_id","package","session"):
            val = source.get(k,record.get(k))
            if val is not None:
                src[k] = val
        relative = next((image.get(k) for k in ("path","relative_path","rel_path","file_path") if image.get(k)),None)
        reason = status.get("drop_reason") or record.get("drop_reason")
        if reason and (not relative or not (root/relative).is_file()):
            with c:
                c.execute("CREATE TABLE IF NOT EXISTS legacy_drops(id TEXT PRIMARY KEY,body TEXT)")
                c.execute("INSERT OR IGNORE INTO legacy_drops VALUES(?,?)",(sid,json.dumps(record)))
                c.execute("INSERT OR IGNORE INTO drops VALUES(?,?,?)",(name,sid,str(reason)))
            result["metadata_only_drops"]+=1
            continue
        if not relative:
            raise ValueError(f"Missing image path for {sid} on line {number}")
        raw = local_file(root,relative).read_bytes()
        if image.get("sha256") and hashlib.sha256(raw).hexdigest()!=image["sha256"]:
            raise ValueError("Image checksum mismatch for "+sid)
        # Only an explicitly attributed source label can be installed automatically.
        # All other pre-v2 judgments/votes remain verbatim for review in the snapshot.
        provenance = labels.get("target_provenance",{})
        for k in ("is_screenshot","content_type"):
            origin = provenance.get(k)
            origin = origin.get("origin") if isinstance(origin,dict) else origin
            if origin=="source" and (k=="is_screenshot" and type(labels.get(k)) is bool or
                                     k=="content_type" and labels.get(k) in NAMES):
                src[k]=labels[k]
        with c:
            c.execute("BEGIN IMMEDIATE")
            canonical,added = dataset.ingest_one(c,raw,src)
            result["added" if added else "existing"]+=1
            d=dataset.get(c,canonical)
            for key,original in (("source",source),("image",image),("labels",labels),("status",status)):
                for field,value in original.items():
                    d[key].setdefault(field,value)
            if reason:
                d["status"].update(excluded=True,drop_reason=reason)
            if canonical==sid and d["annotations"]["review"]["revision"]==0:
                for k in TARGETS:
                    p=provenance.get(k)
                    if not isinstance(p,dict) or p.get("origin") not in ("human","owner") or labels.get(k) is None:
                        continue
                    d["labels"][k]=labels[k]
                    d["labels"]["target_provenance"][k]=p
                from .schema import Snapshot
                Snapshot.model_validate(d)
                if any(p["origin"] in ("human","owner") for p in d["labels"]["target_provenance"].values()):
                    d["annotations"]["review"].update(state="reviewed",revision=1,mode="imported")
                    c.execute("INSERT INTO revisions VALUES(?,1,'legacy import',?,?)",
                              (sid,dataset.store.now(),json.dumps({"labels":d["labels"],"annotations":d["annotations"]})))
            split = status.get("split")
            if split in ("train","validation","test_public","test_personal"):
                existing=c.execute("SELECT partition FROM splits WHERE id=?",(canonical,)).fetchone()
                if existing and existing[0]!=split:
                    raise ValueError("Conflicting legacy split for "+canonical)
                c.execute("INSERT OR IGNORE INTO splits VALUES(?,?,?)",(canonical,split,status.get("duplicate_group_id") or canonical))
            dataset.put(c,d)
    return result
