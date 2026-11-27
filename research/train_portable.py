"""Fit the RapidOCR model on the existing public study's frozen partitions."""
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from classifier import dataset, model, store
from classifier.features import ROOT, OCR_BACKEND, atomic, encoder, extract
from classifier.learning import train, load_model, default_model_path, default_report_dir
from classifier.schema import NAMES
from research.report import build
from research.review_audit import build as review_audit


def main():
    if OCR_BACKEND!="rapidocr":
        raise ValueError("Set CLASSIFIER_OCR=rapidocr before running this command.")
    def offline(event,args):
        if event in ("socket.connect","socket.getaddrinfo"):
            raise AssertionError("Network blocked during portable training")
    def reject(*args,**kwargs):
        raise AssertionError("Qwen blocked during portable training")
    model.api = reject
    sys.addaudithook(offline)
    c = dataset.connect()
    ids = {r[0] for r in c.execute("SELECT id FROM splits")}
    docs = [d for d in dataset.all_docs(c) if d["id"] in ids]
    if not docs:
        raise ValueError("Rebuild the public study before fitting the portable model.")
    old_splits = list(c.execute("SELECT id,partition,group_id FROM splits ORDER BY id"))
    encoder()
    def one(doc):
        f = extract(dataset.image_path(doc).read_bytes(),store.DATA/"features")
        if f["status"]!="ok":
            raise RuntimeError("Feature extraction failed for "+doc["id"])
    with ThreadPoolExecutor(3) as pool:
        for n,_ in enumerate(pool.map(one,docs),1):
            if n%100==0:
                print(f"Portable features: {n}/{len(docs)}",flush=True)
    with tempfile.TemporaryDirectory(dir=store.DATA,prefix="portable-study-") as temp:
        stage = Path(temp)
        audit = train(c,stage/"models",stage/"reports")
        assert [tuple(r) for r in old_splits]==[tuple(r) for r in c.execute("SELECT id,partition,group_id FROM splits ORDER BY id")]
        bundle = load_model(stage/"models/classifier.joblib")
        assert set(bundle["tasks"]["content_type"]["model"].classes_)==set(NAMES)
        build(stage/"reports")
        review_audit(c,stage/"reports")
        for p in (stage/"reports").iterdir():
            atomic(default_report_dir()/p.name,p.read_bytes())
        atomic(default_model_path(),(stage/"models/classifier.joblib").read_bytes())
    summary = dict(training_id=audit["training_id"],learned_categories=sorted(NAMES),
                   selected=audit["tasks"]["content_type"]["selected"],
                   split_counts=audit["split_counts"],same_partitions=True,
                   qwen_calls=0,python_network_connections=0,
                   tested_host="macOS with cross-platform RapidOCR backend; native Windows verification pending")
    atomic(default_report_dir()/"portable-check.json",json.dumps(summary,indent=2).encode())
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
