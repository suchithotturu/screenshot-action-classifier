"""Conservative pixel grouping + app/photographer/owner isolation, persisted before fitting."""
import hashlib
import json
from collections import Counter, defaultdict
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from . import dataset, store
from .features import atomic

class Groups:
    def __init__(self,ids):
        self.parent = {i:i for i in ids}
    def find(self,i):
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i
    def union(self,a,b):
        a,b = sorted((self.find(a),self.find(b)))
        self.parent[b] = a

def eligible(doc,task,allow_pseudo=False):
    label = doc["labels"][task]
    if label is None or doc["status"].get("fixture") or doc["status"].get("excluded"):
        return None
    origin = doc["labels"]["target_provenance"][task]["origin"]
    if task in ("purpose","still_useful"):
        return str(label) if origin=="owner" and doc["labels"]["is_screenshot"] is True else None
    if origin=="source" and task in doc["status"].get("conflicting_targets",[]):
        return None
    accepted = ("source","human","owner","pseudo") if allow_pseudo else ("source","human","owner")
    return str(label) if origin in accepted else None

def group_duplicates(c, docs):
    dup = Groups([d["id"] for d in docs])
    seen_rico, prior = {}, []
    for d in docs:
        sid, im = d["id"], d["image"]
        for src in d["source"]["origins"]:
            if src.get("rico_id"):
                dup.union(sid,seen_rico.setdefault(str(src["rico_id"]),sid))
        sig = np.frombuffer(bytes.fromhex(im["color_signature"]),dtype=np.uint8).astype("int16")
        bits = int(im["dhash"],16)
        ratio = im["width"]/im["height"]
        for other,dh,ar,rgb in prior:
            if abs(ratio/ar-1)<0.03 and (bits^dh).bit_count()<=4 and np.mean(np.abs(sig-rgb))<=7:
                dup.union(sid,other)
        prior.append((sid,bits,ratio,sig))
    fixture_groups = {dup.find(d["id"]) for d in docs if d["status"].get("fixture")}
    votes = defaultdict(lambda: defaultdict(set))
    for d in docs:
        for origin in d["source"]["origins"]:
            for target in ("is_screenshot","content_type"):
                if origin.get(target) is not None:
                    votes[dup.find(d["id"])][target].add(origin[target])
    for d in docs:
        group = dup.find(d["id"])
        d["status"]["duplicate_group_id"] = group
        conflicts = [target for target,values in votes[group].items() if len(values)>1]
        d["status"].update(source_conflict=bool(conflicts),conflicting_targets=conflicts)
        if group in fixture_groups:
            d["status"]["fixture"] = True
        dataset.put(c,d)
    return dup

def prepare(c):
    with c:
        c.execute("BEGIN IMMEDIATE")
        docs = dataset.all_docs(c)
        dup = group_duplicates(c, docs)
        c.execute("CREATE TABLE IF NOT EXISTS locked_labels(id TEXT PRIMARY KEY,labels TEXT,status TEXT)")
        frozen_ids = {r[0] for r in c.execute("SELECT id FROM locked_labels")}
        docs = [d for d in docs if not d["status"].get("fixture") and (d["id"] in frozen_ids or
                any(eligible(d,k) is not None for k in ("is_screenshot","content_type","purpose")))]
        if len(docs)<3:
            raise ValueError("Need eligible labels from at least three independent groups.")
        ids = [d["id"] for d in docs]
        groups = Groups(ids)
        first = {}
        def link(sid,key):
            groups.union(sid,first.setdefault(key,sid))
        owners = sorted({s["owner"] for d in docs for s in d["source"]["origins"] if s.get("owner")})
        personal = set()
        for d in docs:
            sid = d["id"]
            link(sid,("duplicate",dup.find(sid)))
            for s in d["source"]["origins"]:
                for k in ("package","photographer"):
                    if s.get(k):
                        link(sid,(k,s[k]))
                if s["dataset"]=="personal":
                    personal.add(sid)
                    if len(owners)>=3 and s.get("owner"):
                        link(sid,("owner",s["owner"]))
                    elif s.get("session"):
                        link(sid,("session",s.get("owner"),s["session"]))
        g = [groups.find(i) for i in ids]
        personal_g = {groups.find(i) for i in personal}
        old = {r["id"]:r["partition"] for r in c.execute("SELECT * FROM splits")}
        splits = {}
        for sid,group in zip(ids,g):
            if sid in old:
                if splits.setdefault(group,old[sid]) != old[sid]:
                    raise ValueError("New duplicate or source grouping crosses frozen splits. Review the data in a new study.")
        y = [eligible(d,"content_type",allow_pseudo=True) or eligible(d,"is_screenshot") or "unknown" for d in docs]
        n = min(7,len(set(g)),max(Counter(y).values()))
        if n<3:
            raise ValueError("Need at least three independent groups.")
        folds = StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=513)
        for fold,(_,indices) in enumerate(folds.split(ids,y,g)):
            for j in indices:
                p = "validation" if fold==1 else "train"
                if fold==0:
                    p = "test_personal" if g[j] in personal_g else "test_public"
                splits.setdefault(g[j],p)
        # With several owners, assign an entire owner to personal test even on a public-heavy initial corpus.
        if len(owners)>=3:
            holdout = owners[-1]
            for j,d in enumerate(docs):
                if any(s.get("owner")==holdout for s in d["source"]["origins"]) and not any(
                    old.get(i) for i in ids if groups.find(i)==g[j]):
                    splits[g[j]] = "test_personal"
        c.execute("CREATE TABLE IF NOT EXISTS locked_labels(id TEXT PRIMARY KEY,labels TEXT,status TEXT)")
        for d,group in zip(docs,g):
            sid,partition = d["id"],splits[group]
            c.execute("INSERT INTO splits VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET group_id=excluded.group_id",
                      (sid,partition,group))
            if partition.startswith("test"):
                c.execute("INSERT OR IGNORE INTO locked_labels VALUES(?,?,?)",
                          (sid,json.dumps(d["labels"]),json.dumps(d["status"])))
                locked = c.execute("SELECT labels,status FROM locked_labels WHERE id=?",(sid,)).fetchone()
                d["labels"] = json.loads(locked[0])
                d["status"].update(source_conflict=json.loads(locked[1]).get("source_conflict",False))
            d["status"].update(split=partition,split_group_id=group)
    for name in ("train","validation","test_public","test_personal"):
        subset = [dict(id=d["id"],group=d["status"]["split_group_id"],duplicate_group=d["status"]["duplicate_group_id"])
                  for d in docs if d["status"]["split"]==name]
        atomic(store.DATA/"splits"/(name+".jsonl"),"".join(json.dumps(r)+"\n" for r in subset).encode())
    return docs
