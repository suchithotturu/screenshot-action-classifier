"""Back up and rebuild the local working labels from the existing image collection.

This resets reviews and prediction runs. It makes no inference requests.
Run only when a working-library reset is intended.
"""
import json
import shutil
import sqlite3
from collections import Counter
from datetime import datetime, timezone

from classifier import dataset, store
from classifier.features import ROOT, atomic
from classifier.schema import Snapshot


def rebuild():
    c = dataset.connect()
    docs = dataset.all_docs(c)
    if not docs:
        raise ValueError("No local images to rebuild.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = store.DATA/"backups"/("before-rebuild-"+stamp)
    backup.mkdir(parents=True)
    with sqlite3.connect(backup/"annotations.db") as destination:
        c.backup(destination)
    for name in ("models","reports"):
        if (ROOT/name).exists():
            shutil.copytree(ROOT/name,backup/name)
    assignments = [tuple(r) for r in c.execute("SELECT * FROM splits")]
    partition = {r[0]:r[1] for r in assignments}
    proposals = [tuple(r) for r in c.execute("SELECT * FROM proposals")]
    drops = [tuple(r) for r in c.execute("SELECT * FROM drops")]
    images = [(d,dataset.image_path(d).read_bytes()) for d in docs]
    fixture_categories = {"01-shirt.png":"shopping_product","02-Café notes.png":"map_place",
                          "03-gym.png":"shopping_product","04-workshop.png":"article_reading",
                          "05-recipe.png":"article_reading","06-receipt.png":"receipt_confirmation",
                          "07-meme.png":"social_meme"}
    changes = []
    with c:
        c.execute("BEGIN IMMEDIATE")
        for table in ("revisions","proposals","locked_labels","prediction_runs","splits","drops","samples"):
            if c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(table,)).fetchone():
                c.execute("DELETE FROM "+table)
        for original,raw in images:
            for source in original["source"]["origins"]:
                sid,_ = dataset.ingest_one(c,raw,source)
            if sid!=original["id"]:
                raise ValueError("Canonical ID changed during rebuild.")
            doc = dataset.get(c,sid)
            source_labels = json.loads(json.dumps(doc["labels"]))
            doc["annotations"]["source_provided_targets"] = source_labels
            doc["annotations"]["qwen"] = original["annotations"].get("qwen")
            doc["annotations"]["review"].update(state="auto_labeled",updated=store.now())
            if doc["status"]["fixture"]:
                names = {s.get("relative_name") for s in doc["source"]["origins"]}
                for name,category in fixture_categories.items():
                    if name in names:
                        doc["labels"]["content_type"] = category
                        doc["labels"]["target_provenance"]["content_type"] = dict(
                            origin="pseudo",method="rendered_fixture_definition",at=store.now())
                if "portrait.png" in names:
                    doc["labels"]["is_screenshot"] = False
                    doc["labels"]["target_provenance"]["is_screenshot"] = dict(origin="source",method="NASA photo",at=store.now())
            qwen = doc["annotations"]["qwen"] or {}
            fields = qwen.get("fields") or {}
            suggestion = fields.get("content_type")
            # Frozen public evaluation targets remain source-provided. Only saved training
            # suggestions may replace a broad source category, with explicit pseudo provenance.
            if (partition.get(sid)=="train" and not doc["status"]["fixture"]
                    and qwen.get("status")=="ok" and fields.get("is_screenshot") is True
                    and suggestion is not None and suggestion!=doc["labels"]["content_type"]):
                changes.append(dict(id=sid,source=doc["labels"]["content_type"],provisional=suggestion))
                doc["labels"]["content_type"] = suggestion
                doc["labels"]["target_provenance"]["content_type"] = dict(
                    origin="pseudo",method="saved_qwen_suggestion",model=qwen["model"],at=qwen["timestamp"])
            for target in ("purpose","still_useful"):
                doc["labels"][target] = None
                doc["labels"]["target_provenance"][target] = {"origin":"unknown"}
            doc["status"]["automatic_labels"] = True
            Snapshot.model_validate(doc)
            dataset.put(c,doc)
        c.executemany("INSERT INTO splits VALUES(?,?,?)",assignments)
        c.executemany("INSERT INTO proposals VALUES(?,?,?)",proposals)
        c.executemany("INSERT INTO drops VALUES(?,?,?)",drops)
        c.execute("CREATE TABLE IF NOT EXISTS locked_labels(id TEXT PRIMARY KEY,labels TEXT,status TEXT)")
        for sid,part,group in assignments:
            if part.startswith("test"):
                doc = dataset.get(c,sid)
                c.execute("INSERT INTO locked_labels VALUES(?,?,?)",(sid,json.dumps(doc["labels"]),json.dumps(doc["status"])))
    current = dataset.all_docs(c)
    result = dict(images=len(current),backup=str(backup.relative_to(ROOT)),
                  sources=dict(Counter(d["source"]["origins"][0]["dataset"] for d in current)),
                  content=dict(Counter(d["labels"]["content_type"] or "unknown" for d in current)),
                  training_pseudo_labels=changes,new_qwen_calls=0,
                  labels="Public source labels, with saved Qwen corrections restricted to training. Owner targets unknown.")
    atomic(ROOT/"reports/rebuild.json",json.dumps(result,indent=2).encode())
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    rebuild()
