"""Replay the saved model against frozen test labels with no fit or model selection."""
import json
from collections import defaultdict
import numpy as np
from classifier import dataset, splits, store
from classifier.features import ROOT, extract
from classifier.learning import load_model, margins, metrics, default_report_dir, requires_pixels

def replay(c):
    bundle = load_model()
    report = json.loads((default_report_dir()/"results.json").read_text())
    if report["training_id"]!=bundle["training_id"]:
        raise ValueError("Model/report training IDs differ.")
    outcomes = {}
    for task,head in bundle["tasks"].items():
        results = {}
        for part,old in report["tasks"][task]["tests"].items():
            rows = list(c.execute("SELECT samples.doc, locked_labels.labels,locked_labels.status FROM splits JOIN samples USING(id) JOIN locked_labels USING(id) WHERE partition=? ORDER BY samples.rowid",(part,)))
            X,y = [],[]
            for body,labels,status in rows:
                d = json.loads(body)
                d.update(labels=json.loads(labels),status=json.loads(status))
                target = splits.eligible(d,task)
                if target is None:
                    continue
                f = extract(dataset.image_path(d).read_bytes(),store.DATA/"features",False,requires_pixels(bundle))
                if f["status"]=="ok":
                    X.append(f)
                    y.append(target)
            x = head["transform"].transform(X)
            p = head["model"].predict(x)
            new = metrics(y,p,old["classes"])
            accepted = int((margins(head["model"],x)>=head["threshold"]).sum())
            if new["matrix"]!=old["matrix"] or accepted!=old["accepted"]:
                raise ValueError(f"Saved evaluation differs for {task}/{part}.")
            results[part] = dict(n=new["n"],macro_f1=new["macro_f1"],accepted=accepted,identical_matrix=True)
        outcomes[task]=results
    return outcomes
