"""Create the demonstration folder from frozen public test groups, then run disconnected inference."""
import argparse
import json
import shutil
import sys
import time
import tempfile
from pathlib import Path
from classifier import dataset, production, store
from classifier.features import ROOT, atomic
from classifier.learning import load_model

def make_folder(c):
    target = ROOT/"demo-unseen-eight"
    target.mkdir(exist_ok=True)
    seen = set()
    for r in c.execute("SELECT id FROM splits WHERE partition='test_public' ORDER BY id"):
        d = dataset.get(c,r[0])
        label = d["labels"]["content_type"] or "photo"
        if label in seen:
            continue
        seen.add(label)
        source = dataset.image_path(d)
        shutil.copyfile(source,target/(str(len(seen)).zfill(2)+source.suffix))
    for name,display in [("08-crop.webp","90-blank.webp"),("portrait.png","91-portrait.png")]:
        shutil.copyfile(ROOT/"demo"/name,target/display)
    return target

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold",action="store_true",help="Use a new empty feature cache (requires local Vision service access).")
    cold = parser.parse_args().cold
    c = dataset.connect()
    folder = make_folder(c)
    def offline(event,args):
        if event in ("socket.connect","socket.getaddrinfo"):
            raise RuntimeError("Network operation attempted during classification")
    sys.addaudithook(offline)
    # Any accidental use of the existing Qwen client also fails, even before a connection attempt.
    from classifier import model
    def reject(*args,**kwargs):
        raise AssertionError("Qwen client called during production")
    model.api = reject
    original_data = store.DATA
    with tempfile.TemporaryDirectory() as temp:
        try:
            if cold:
                store.DATA = Path(temp)
            start = time.perf_counter()
            run,body = production.predict_folder(c,folder,bundle=load_model())
            elapsed = time.perf_counter()-start
            assert all(r["status"]=="ok" for r in body["items"]),body["items"]
            start = time.perf_counter()
            _,again = production.predict_folder(c,folder,bundle=load_model())
            warm = time.perf_counter()-start
        finally:
            store.DATA = original_data
    for a,b in zip(body["items"],again["items"],strict=True):
        assert {k:v for k,v in a.items() if k!="seconds"}=={k:v for k,v in b.items() if k!="seconds"}
    sha = {r["sha256"] for r in body["items"]}
    trained = {r[0] for r in c.execute("SELECT sha FROM samples JOIN splits USING(id) WHERE partition='train'")}
    assert not sha&trained
    proof = dict(images=len(sha),training_overlap=0,identical_after_reload=True,python_network_connections=0,
                 qwen_calls=0,initial_cache="empty" if cold else "existing",first_run_seconds=elapsed,warm_cache_seconds=warm,training_id=body["training_id"],
                 examples=[{"file":r["path"],"route":r["route"],"status":r["status"]} for r in body["items"]])
    atomic(ROOT/"reports/offline-check.json",json.dumps(proof,indent=2).encode())
    print(json.dumps(proof,indent=2))
if __name__=="__main__":
    main()
