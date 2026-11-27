"""Prepare additional real public images and obtain local Qwen content pseudo-labels.

Public UI hierarchy text selects candidates only. Features are extracted from pixels.
Personal prediction folders and synthetic fixtures are excluded from acquisition.
"""
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

from classifier import annotation, dataset, store
from classifier.features import ROOT, atomic, extract
from classifier.schema import Snapshot

PATTERNS = {
    "shopping_product":r"add to (?:cart|bag|basket)|buy now|checkout|free shipping|discount|sale price|\$\s*\d",
    "social_meme":r"likes|comment|retweet|tweet|funny|meme|entertain|news feed|playlist|lyrics|now playing|music|watch video|game|posts",
}


def node_text(obj):
    if isinstance(obj,dict):
        for key,value in obj.items():
            if key=="text" and isinstance(value,str):
                yield value
            elif isinstance(value,(dict,list)):
                yield from node_text(value)
    elif isinstance(obj,list):
        for value in obj:
            yield from node_text(value)


def candidates(c):
    docs = dataset.all_docs(c)
    original = {s.get("id") for d in docs for s in d["source"]["origins"]
                if s.get("acquisition")!="eight_category_expansion"}
    by_target = defaultdict(list)
    with zipfile.ZipFile(store.DATA/"raw/masc.zip") as z:
        for name in z.namelist():
            if "/Full/" not in name or not name.endswith(".json"):
                continue
            topic,sid = name.rsplit("/",2)[-2:]
            sid = sid[:-5]
            if "masc_"+sid in original:
                continue
            tree = json.loads(z.read(name))
            app = tree.get("activity_name","").split("/")[0]
            if not app:
                continue
            text = "\n".join(node_text(tree))
            for target,pattern in PATTERNS.items():
                score = len(re.findall(pattern,text,re.I))
                if score:
                    by_target[target].append(dict(id="masc_"+sid,rico_id=sid,package=app,source_class=topic,
                        member=f"MASC_UI/MASC_UI/{topic}/{sid}.jpg",target=target,score=score))
    queues = {}
    for target,items in by_target.items():
        apps = defaultdict(deque)
        for item in sorted(items,key=lambda x:(-x["score"],hashlib.sha256(x["id"].encode()).hexdigest())):
            if len(apps[item["package"]])<5:
                apps[item["package"]].append(item)
        balanced = []
        while apps:
            for app in list(apps):
                balanced.append(apps[app].popleft())
                if not apps[app]:
                    del apps[app]
        queues[target] = deque(balanced)
    atomic(store.DATA/"raw/expansion-candidates.json",json.dumps({k:list(v) for k,v in queues.items()},indent=2).encode())
    return queues


def import_receipts(c):
    import pyarrow.parquet as pq
    path = store.DATA/"raw/cord-train.parquet"
    table = pq.read_table(path)
    count = 0
    for row_number,row in enumerate(table.slice(0,120).to_pylist()):
        raw = row["image"]["bytes"]
        src = dict(id=f"cord_train_{row_number:04d}",dataset="cord",license="CC BY 4.0",
                   attribution="Park et al., CORD, NAVER CLOVA",source_split="train",source_row=row_number,
                   content_type="receipt_confirmation",is_screenshot=False,
                   url="https://huggingface.co/datasets/naver-clova-ix/cord-v2",
                   acquisition="eight_category_expansion")
        dataset.ingest_one(c,raw,src)
        count+=1
    return count


def prepare():
    annotation.model.provenance()
    c = dataset.connect()
    print("Imported receipt records",import_receipts(c),flush=True)
    with c:
        c.execute("BEGIN IMMEDIATE")
        for d in dataset.all_docs(c):
            if d["source"]["origins"][0]["dataset"]=="tf_flowers":
                for s in d["source"]["origins"]:
                    s["content_type"]="other"
                d["labels"]["content_type"]="other"
                d["labels"]["target_provenance"]["content_type"]={"origin":"source","mapping":"ordinary_flower_photo_to_miscellaneous"}
                Snapshot.model_validate(d)
                dataset.put(c,d)
    queues = candidates(c)
    print("Candidate queues",{k:len(v) for k,v in queues.items()},flush=True)
    quotas = {"shopping_product":60,"social_meme":45}
    counts = Counter(d["labels"]["content_type"] for d in dataset.all_docs(c) if d["labels"]["content_type"] is not None
                     and any(s.get("acquisition")=="eight_category_expansion" and s["dataset"]=="masc" for s in d["source"]["origins"]))
    seen = set()
    jobs = []
    attempted = failures = 0
    with zipfile.ZipFile(store.DATA/"raw/masc.zip") as z, ThreadPoolExecutor(3) as pool:
        while queues and attempted<330:
            target = min(queues,key=lambda k:counts[k]/quotas[k])
            if counts[target]>=quotas[target] or not queues[target]:
                del queues[target]
                continue
            candidate = queues[target].popleft()
            if candidate["id"] in seen:
                continue
            seen.add(candidate["id"])
            src = {k:candidate[k] for k in ("id","rico_id","package","source_class")}
            src.update(dataset="masc",license="MIT (MASC release)",is_screenshot=True,
                       url="https://github.com/Ali-Aahmed/MASC-Dataset",acquisition="eight_category_expansion")
            sid,_ = dataset.ingest_one(c,z.read(candidate["member"]),src)
            doc = dataset.get(c,sid)
            if sid!=candidate["id"] or doc["annotations"]["review"]["state"]=="reviewed":
                continue
            if doc["labels"]["target_provenance"]["content_type"].get("method")=="local_qwen_annotation":
                continue
            q = annotation.propose(c,sid,compact=True)
            attempted+=1
            fields = q.get("fields") or {}
            label = fields.get("content_type")
            if q["status"]=="ok" and label is not None:
                with c:
                    c.execute("BEGIN IMMEDIATE")
                    doc = dataset.get(c,sid)
                    doc["labels"]["content_type"]=label
                    doc["labels"]["target_provenance"]["content_type"] = dict(origin="pseudo",method="local_qwen_annotation",
                        model=q["model"],prompt_version=q["prompt_version"],at=q["timestamp"])
                    doc["annotations"]["review"].update(state="auto_labeled",updated=store.now())
                    Snapshot.model_validate(doc)
                    dataset.put(c,doc)
                counts[label]+=1
                jobs.append(pool.submit(extract,dataset.image_path(doc).read_bytes(),store.DATA/"features"))
            else:
                failures+=1
                if failures>=3 and not counts:
                    raise RuntimeError("Annotation is unavailable. Successful records are saved; rerun after checking the local model service.")
            if attempted%5==0:
                progress=dict(attempted=attempted,labels=dict(counts),failed_or_abstained=failures)
                atomic(ROOT/"reports/expansion-progress.json",json.dumps(progress,indent=2).encode())
                print(json.dumps(progress),flush=True)
        for job in jobs:
            if job.result()["status"]!="ok":
                failures+=1
    # Extract source-labeled receipts and any other new rows with missing cached features.
    docs = dataset.all_docs(c)
    with ThreadPoolExecutor(3) as pool:
        work = [d for d in docs if any(s.get("acquisition")=="eight_category_expansion" for s in d["source"]["origins"])]
        for f in pool.map(lambda d:extract(dataset.image_path(d).read_bytes(),store.DATA/"features"),work):
            if f["status"]!="ok":
                failures+=1
    summary=dict(qwen_images=attempted,pseudo_content_counts=dict(counts),failed_or_abstained=failures,
                 total_records=len(docs),content_counts=dict(Counter(d["labels"]["content_type"] or "unknown" for d in docs)),
                 label_policy="CORD and public mappings provide source labels; additional UI content is local-Qwen pseudo-labeling. No owner intent targets.")
    atomic(ROOT/"reports/expansion.json",json.dumps(summary,indent=2).encode())
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    prepare()
