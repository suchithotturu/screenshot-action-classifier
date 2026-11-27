"""Fit and publish the eight-category study with Qwen and Python networking blocked."""
import json
import sqlite3
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from classifier import dataset, model, store
from classifier.features import ROOT, atomic
from classifier.learning import load_model, train
from classifier.schema import NAMES
from research.report import build
from research.review_audit import build as review_audit


def main():
    def reject(*args,**kwargs):
        raise AssertionError("Qwen is disabled during training.")
    def offline(event,args):
        if event in ("socket.connect","socket.getaddrinfo"):
            raise AssertionError("Python network access is disabled during training.")
    model.api = reject
    sys.addaudithook(offline)
    c = dataset.connect()
    docs = dataset.all_docs(c)
    with c:
        for doc in docs:
            if any(s.get("acquisition")=="eight_category_expansion" and s.get("dataset")=="masc" for s in doc["source"]["origins"]):
                if doc["labels"]["content_type"] is None:
                    doc["status"].update(excluded=True,drop_reason="Candidate has no accepted content annotation.")
                else:
                    doc["status"].pop("excluded",None)
                    doc["status"].pop("drop_reason",None)
                dataset.put(c,doc)
    observed = Counter(d["labels"]["content_type"] for d in docs if not d["status"].get("fixture"))
    if not set(NAMES)<=set(observed):
        raise ValueError("More category annotations are required before publishing an eight-class model.")
    with c:
        for doc in docs:
            if doc["labels"]["content_type"] is not None and doc["annotations"]["review"]["state"]=="pending":
                doc["annotations"]["review"].update(state="auto_labeled",updated=store.now())
                dataset.put(c,doc)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = store.DATA/"backups"/("before-eight-split-"+stamp)
    backup.mkdir(parents=True)
    with sqlite3.connect(backup/"annotations.db") as out:
        c.backup(out)
    # The enlarged label universe defines a new experiment. The prior assignments,
    # targets and reports survive in backups; grouping is recomputed across all sources.
    with c:
        c.execute("DELETE FROM locked_labels")
        c.execute("DELETE FROM splits")
    with tempfile.TemporaryDirectory(dir=store.DATA,prefix="eight-study-") as temp:
        stage = Path(temp)
        audit = train(c,stage/"models",stage/"reports")
        bundle = load_model(stage/"models/classifier.joblib")
        learned = set(bundle["tasks"]["content_type"]["model"].classes_)
        if learned!=set(NAMES):
            raise ValueError("The staged classifier lacks a category; the existing deployed model is preserved.")
        build(stage/"reports")
        review_audit(c,stage/"reports")
        for p in (stage/"reports").iterdir():
            atomic(ROOT/"reports"/p.name,p.read_bytes())
        atomic(ROOT/"models/classifier.joblib",(stage/"models/classifier.joblib").read_bytes())
    summary = dict(training_id=audit["training_id"],learned_categories=sorted(learned),
                   inventory={k:observed[k] for k in NAMES},split_counts=audit["split_counts"],
                   selected=audit["tasks"]["content_type"]["selected"],
                   training_origins=audit["tasks"]["content_type"]["training_label_origins"],
                   validation_origins=audit["tasks"]["content_type"]["validation_label_origins"],
                   qwen_calls_during_training=0,python_network_connections=0,
                   split_backup=str(backup.relative_to(ROOT)))
    atomic(ROOT/"reports/eight-category-check.json",json.dumps(summary,indent=2).encode())
    dataset.export(c)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
