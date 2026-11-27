# /// script
# requires-python = ">=3.12"
# dependencies = ["scikit-learn>=1.5,<2"]
# ///
"""Compare three classifiers using independent human annotations and a JSON run export."""
import argparse
import csv
import json
import time
from pathlib import Path
from statistics import median

LABELS = ["BUY_COMPARE","VISIT","ATTEND_REGISTER","READ_WATCH_TRY","REFERENCE","OTHER"]

def evaluate(annotations, report, partition):
    with annotations.open(newline="") as f:
        rows=[r for r in csv.DictReader(f) if r["human_image_type"]=="screenshot"]
    groups={}
    ids=set()
    for r in rows:
        if not r["image_id"] or r["image_id"] in ids or not r["group_id"]:
            raise ValueError("Each unique image needs a content hash and a related-screen group.")
        ids.add(r["image_id"])
        if r["split"] not in ("train","validation","test") or r["human_category"] not in LABELS:
            raise ValueError("Unknown split or human label.")
        if groups.setdefault(r["group_id"],r["split"])!=r["split"]:
            raise ValueError("Related screens cross split boundaries.")
    train=[r for r in rows if r["split"]=="train"]
    held=[r for r in rows if r["split"]==partition]
    if len({r["human_category"] for r in train})<2 or not held:
        raise ValueError("Supply human-labeled training and held-out screenshots before evaluation.")
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import make_pipeline
    truth=[r["human_category"] for r in held]
    def metrics(pred):
        labels=LABELS+["ABSTAIN"]
        scores=classification_report(truth,pred,labels=LABELS,output_dict=True,zero_division=0)
        return dict(sample_count=len(truth),macro_f1=scores["macro avg"]["f1-score"],scores=scores,confusion_labels=labels,confusion=confusion_matrix(truth,pred,labels=labels).tolist())
    saved={r["image_id"]:r for r in json.loads(report.read_text())["items"] if r["state"]!="duplicate"}
    pred=[]; timings=[]
    for r in held:
        item=saved.get(r["image_id"],{})
        p=item.get("prediction") or {}
        pred.append(p["category"] if p.get("image_type")=="screenshot" else "ABSTAIN")
        if item.get("state")=="analyzed": timings.append(item["seconds"])
    results={"vlm":metrics(pred)|dict(median_image_seconds=median(timings) if timings else None,timed_count=len(timings))}
    for name,estimator in [("naive_bayes",MultinomialNB()),("linear_svm",LinearSVC(random_state=11))]:
        model=make_pipeline(TfidfVectorizer(analyzer="char",ngram_range=(2,4)),estimator)
        model.fit([r["ocr_text"] for r in train],[r["human_category"] for r in train])
        start=time.monotonic()
        predicted=model.predict([r["ocr_text"] for r in held])
        results[name]=metrics(predicted)|dict(seconds_per_image_excluding_ocr=(time.monotonic()-start)/len(held))
    return dict(train_count=len(train),partition=partition,results=results)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations",type=Path)
    parser.add_argument("report",type=Path)
    parser.add_argument("--partition",choices=["validation","test"],default="validation")
    args=parser.parse_args()
    try: print(json.dumps(evaluate(args.annotations,args.report,args.partition),indent=2))
    except ValueError as e: parser.exit(2,str(e)+"\n")
