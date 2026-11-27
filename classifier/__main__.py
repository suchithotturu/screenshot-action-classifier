"""Local command line for the annotation and supervised classification workflow."""
import argparse
import json
import os
import sys
from pathlib import Path
from . import dataset, store
from .features import ROOT, extract

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd",required=True)
    p = sub.add_parser("app")
    p.add_argument("--mode",choices=["classify","annotation","legacy"],default="classify")
    p.add_argument("--port",type=int,default=8514)
    p = sub.add_parser("ingest")
    p.add_argument("folder")
    p.add_argument("--source",choices=["public","personal","fixture","masc","screen2words","memes"],default="personal")
    p.add_argument("--owner",choices=list("ABCD"))
    p.add_argument("--session",default="")
    p = sub.add_parser("import-master")
    p.add_argument("manifest")
    p.add_argument("--images")
    p = sub.add_parser("restore")
    p.add_argument("manifest")
    p = sub.add_parser("features")
    p.add_argument("--refresh",action="store_true")
    p = sub.add_parser("predict")
    p.add_argument("folder")
    p.add_argument("--screenshots-only",action="store_true",help="Compatibility option. Content classification now runs for every image type.")
    for name in ("train","evaluate","export","inventory"):
        sub.add_parser(name)
    args = parser.parse_args()
    if args.cmd=="app":
        os.chdir(ROOT)
        os.environ["ASTRA_MODE"] = args.mode
        os.execv(sys.executable,[sys.executable,"-m","streamlit","run","app.py","--server.address","127.0.0.1","--server.port",str(args.port)])
    c = dataset.connect()
    if args.cmd=="ingest":
        out = dataset.ingest_folder(c,args.folder,args.source,args.owner,args.session)
    elif args.cmd=="export":
        out = str(dataset.export(c))
    elif args.cmd=="import-master":
        from .legacy_manifest import import_master
        out = import_master(c,args.manifest,args.images)
    elif args.cmd=="restore":
        from .restore import restore
        out = restore(c,args.manifest)
    elif args.cmd=="features":
        from collections import Counter
        out = Counter(extract(dataset.image_path(d).read_bytes(),store.DATA/"features",args.refresh)["status"] for d in dataset.all_docs(c))
    elif args.cmd=="predict":
        from .production import predict_folder
        run,body = predict_folder(c,args.folder,args.screenshots_only)
        out = {"run":run,"items":[{k:v for k,v in r.items() if k in ("path","route","actions","tasks","reasons")} for r in body["items"]]}
    elif args.cmd=="train":
        from .learning import train
        from research.report import build
        audit = train(c)
        build()
        from research.review_audit import build as review_audit
        review_audit(c)
        out = {"training_id":audit["training_id"],"model":str(ROOT/"models/classifier.joblib"),"report":str(ROOT/"reports/REPORT.md")}
    elif args.cmd=="evaluate":
        from research.replay import replay
        out = replay(c)
    elif args.cmd=="inventory":
        from collections import Counter
        docs = dataset.all_docs(c)
        out = dict(images=len(docs),states=Counter(d["annotations"]["review"]["state"] for d in docs),
                   source=Counter(d["source"]["origins"][0]["dataset"] for d in docs),
                   content=Counter(d["labels"]["content_type"] or "unknown" for d in docs))
    print(json.dumps(out,indent=2) if not isinstance(out,str) else out)

if __name__=="__main__":
    try:
        main()
    except (ValueError,OSError,RuntimeError) as exc:
        sys.exit(str(exc))
