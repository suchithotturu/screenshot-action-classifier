"""Aggregate review agreement; export a text-free public sample for the presentation."""
from collections import Counter, defaultdict
import json
from classifier import dataset
from classifier.features import ROOT, atomic

def build(c,folder=None):
    folder = folder or ROOT/"reports"
    docs = dataset.all_docs(c)
    counts = Counter()
    agreement = defaultdict(Counter)
    samples = {}
    assigned = dict(c.execute("SELECT id,partition FROM splits"))
    for d in docs:
        a = d["annotations"]
        r = a["review"]
        counts[r["state"]] += 1
        if r["state"]=="reviewed" and not d["status"]["fixture"]:
            q = a.get("qwen") or {}
            if q.get("status")=="ok":
                for k in ("is_screenshot","content_type"):
                    if q["fields"].get(k) is None or d["labels"][k] is None:
                        continue
                    cell=agreement[r["mode"]+"_"+k]
                    cell["compared"]+=1
                    cell["agreed"]+=int(q["fields"][k]==d["labels"][k])
        src = d["source"]["origins"][0]
        label = d["labels"]["content_type"] or "photo"
        if src["dataset"] in ("masc","enrico","tf_flowers") and label not in samples:
            samples[label]=dict(id=d["id"],dataset=src["dataset"],source_class=src.get("source_class"),
                                is_screenshot=d["labels"]["is_screenshot"],content_type=d["labels"]["content_type"],
                                purpose=None,content_origin=d["labels"]["target_provenance"]["content_type"]["origin"],
                                split=assigned.get(d["id"]))
    payload = dict(review_states=dict(counts),real_qwen_comparisons={mode:dict(v) for mode,v in agreement.items()},
                   definition="Agreement counts paired non-null Qwen and approved targets by review mode, excluding fixtures.")
    atomic(folder/"review-summary.json",json.dumps(payload,indent=2).encode())
    atomic(folder/"public-samples.jsonl","".join(json.dumps(r)+"\n" for r in samples.values()).encode())
