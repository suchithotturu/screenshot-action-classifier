"""Reclassify a saved folder with the current local content policy and audit the change.

The report contains aggregate counts. Original runs remain in SQLite for comparison.
"""
import argparse
import json
import sys
from collections import Counter
from classifier import dataset, model, production
from classifier.features import ROOT, atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=int,required=True)
    args = parser.parse_args()
    c = dataset.connect()
    row = c.execute("SELECT body FROM prediction_runs WHERE id=?",(args.run,)).fetchone()
    if row is None:
        raise ValueError("Saved run does not exist.")
    before = json.loads(row[0])
    def offline(event,args):
        if event in ("socket.connect","socket.getaddrinfo"):
            raise AssertionError("Network call during local classification.")
    sys.addaudithook(offline)
    def reject(*args,**kwargs):
        raise AssertionError("Qwen called during classification.")
    model.api = reject
    run,after = production.predict_folder(c,before["directory"])
    old = {r["path"]:r for r in before["items"]}
    same_images = all(r.get("sha256")==old[r["path"]].get("sha256") for r in after["items"])
    assert same_images and len(old)==len(after["items"])
    assert all(r["route"]!="Needs review" for r in after["items"])
    assert all("content_type" in r["tasks"] for r in after["items"] if r["status"]=="ok")
    trained = {r[0] for r in c.execute("SELECT sha FROM samples JOIN splits USING(id) WHERE partition='train'")}
    assert not trained & {r.get("sha256") for r in after["items"]}
    audit = dict(previous_run=args.run,current_run=run,images=len(old),same_images=same_images,
                 routing_version=production.ROUTING_VERSION,training_id=after["training_id"],
                 before=dict(Counter(r["route"] for r in before["items"])),
                 after=dict(Counter(r["route"] for r in after["items"])),
                 category_sources=dict(Counter(r.get("category_source","error") for r in after["items"])),
                 local_rules=dict(Counter(r.get("content_rule",{}).get("rule") for r in after["items"] if r.get("content_rule"))),
                 qwen_calls=0,python_network_connections=0,training_image_overlap=0,
                 limitation="Deployment regression and coverage check, without independently labeled accuracy measurements. The supervised head learns eight categories; OCR rules assist uncertain predictions.")
    atomic(ROOT/"reports/routing-check.json",json.dumps(audit,indent=2).encode())
    print(json.dumps(audit,indent=2))


if __name__=="__main__":
    main()
