"""Fit supervised heads to exactly the pixel features used in classification."""
import hashlib
import io
import json
import time
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
import sklearn
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from . import dataset, splits, store
from .features import NUMERIC, ROOT, VERSION, OCR_BACKEND, atomic, extract, compatible_version

SEED = 513
TASKS = ("content_type","is_screenshot","purpose")

def default_model_path():
    return ROOT/"models"/("rapidocr/classifier.joblib" if OCR_BACKEND=="rapidocr" else "classifier.joblib")

def default_report_dir():
    return ROOT/"reports"/("rapidocr" if OCR_BACKEND=="rapidocr" else "")

class Fusion:
    def __init__(self, mode):
        self.mode = mode
        self.vectorizer = None
        self.scaler = None
        self.names = []
    def fit(self, rows):
        if self.mode != "clip":
            texts = [r["ocr"]["text"] for r in rows]
            for cutoff in (2,1):
                vectorizer = TfidfVectorizer(ngram_range=(1,2),max_features=3000,min_df=cutoff,sublinear_tf=True)
                try:
                    vectorizer.fit(texts)
                except ValueError:
                    continue
                self.vectorizer = vectorizer
                self.names.extend("word:"+s for s in vectorizer.get_feature_names_out())
                break
            if self.mode != "text":
                self.scaler = StandardScaler()
                self.scaler.fit([r["numeric"] for r in rows])
                self.names.extend(NUMERIC)
            elif self.vectorizer is None:
                self.names.append("empty_vocabulary")
        if self.mode not in ("ocr","text"):
            self.names.extend(f"clip_{i}" for i in range(512))
        return self
    def transform(self, rows):
        parts = []
        if self.vectorizer is not None:
            parts.append(self.vectorizer.transform([r["ocr"]["text"] for r in rows]))
        if self.scaler is not None:
            parts.append(sparse.csr_matrix(self.scaler.transform([r["numeric"] for r in rows])))
        if self.mode not in ("ocr","text"):
            parts.append(sparse.csr_matrix([r["embedding"] for r in rows]))
        if not parts:
            return sparse.csr_matrix((len(rows),1))
        return sparse.hstack(parts,format="csr")

def margins(model, x):
    scores = model.decision_function(x) if hasattr(model,"decision_function") else model.predict_proba(x)
    if scores.ndim==1:
        return np.abs(scores)*2
    top = np.sort(scores,axis=1)[:,-2:]
    return top[:,1]-top[:,0]

def metrics(y, p, classes):
    return dict(n=len(y),accuracy=accuracy_score(y,p),macro_f1=f1_score(y,p,labels=classes,average="macro",zero_division=0),
                supported_macro_f1=f1_score(y,p,labels=sorted(set(y)),average="macro",zero_division=0),
                weighted_f1=f1_score(y,p,labels=classes,average="weighted",zero_division=0),
                per_class=classification_report(y,p,labels=classes,output_dict=True,zero_division=0),
                matrix=confusion_matrix(y,p,labels=classes).tolist(),classes=classes)

def train(c, model_dir=None, report_dir=None):
    model_dir = Path(model_dir or default_model_path().parent)
    report_dir = Path(report_dir or default_report_dir())
    docs = splits.prepare(c)
    feature_rows, failures = {}, []
    for d in docs:
        f = extract(dataset.image_path(d).read_bytes(),store.DATA/"features")
        if f["status"]=="ok":
            feature_rows[d["id"]] = f
        else:
            failures.append(d["id"])
    tasks = {}
    partitions = Counter(d["status"]["split"] for d in docs)
    groups = {p:{d["status"]["split_group_id"] for d in docs if d["status"]["split"]==p} for p in partitions}
    overlaps = sum(len(groups[a]&groups[b]) for a in groups for b in groups if a<b)
    if overlaps:
        raise ValueError("Frozen split overlap")
    audit = dict(feature_version=VERSION,sklearn=sklearn.__version__,seed=SEED,split_counts=dict(partitions),
                 group_counts={p:len(v) for p,v in groups.items()},group_overlaps=overlaps,failed_features=len(failures),
                 sources=dict(Counter(d["source"]["origins"][0]["dataset"] for d in docs)),
                 fixtures_excluded=sum(d["status"]["fixture"] for d in dataset.all_docs(c)),
                 latency_ms={k:float(np.median([1000*f[k] for f in feature_rows.values()])) for k in ("ocr_seconds","clip_seconds")} if feature_rows else {},
                 tasks={},total_eligible=len(docs),
                 label_policy="Source and human labels plus explicit Qwen pseudo-labels support training and provisional validation. Primary frozen test metrics exclude pseudo-label targets; teacher agreement is reported separately.")
    for task in TASKS:
        rows = {p:[] for p in ("train","validation","test_public","test_personal")}
        for d in docs:
            y = splits.eligible(d,task,allow_pseudo=d["status"]["split"] in ("train","validation"))
            if y is not None and d["id"] in feature_rows:
                rows[d["status"]["split"]].append((d["id"],y))
        counts = {p:dict(Counter(y for _,y in v)) for p,v in rows.items()}
        info = audit["tasks"][task] = dict(counts=counts,comparisons=[],tests={})
        info["training_label_origins"] = dict(Counter(d["labels"]["target_provenance"][task]["origin"]
            for d in docs if d["status"]["split"]=="train" and splits.eligible(d,task,allow_pseudo=True) is not None))
        info["validation_label_origins"] = dict(Counter(d["labels"]["target_provenance"][task]["origin"]
            for d in docs if d["status"]["split"]=="validation" and splits.eligible(d,task,allow_pseudo=True) is not None))
        info["provisional_validation"] = bool(info["validation_label_origins"].get("pseudo"))
        classes = sorted({y for part in rows.values() for _,y in part})
        if len(counts["train"])<2 or not rows["validation"] or (task=="purpose" and
               (len(counts["train"])<4 or min(counts["train"].values())<5)):
            info["unavailable"] = "Insufficient eligible class or validation support."
            continue
        selection_classes = sorted(set(counts["train"]) | set(counts["validation"]))
        train_rows = [feature_rows[i] for i,_ in rows["train"]]
        valid_rows = [feature_rows[i] for i,_ in rows["validation"]]
        yt = [y for _,y in rows["train"]]
        yv = [y for _,y in rows["validation"]]
        best = None
        for mode in ("ocr","clip","fusion","text"):
            transform = Fusion(mode).fit(train_rows)
            xt,xv = transform.transform(train_rows),transform.transform(valid_rows)
            candidates = [("Logistic regression",LogisticRegression(max_iter=1500,class_weight="balanced",random_state=SEED)),
                          ("Linear SVM",LinearSVC(class_weight="balanced",dual="auto",random_state=SEED,max_iter=5000)),
                          ("k-NN",KNeighborsClassifier(n_neighbors=min(5,len(yt)),weights="distance",metric="cosine",algorithm="brute"))]
            if mode=="text":
                candidates.append(("Complement Naive Bayes",ComplementNB(alpha=1.0)))
            if mode=="ocr":
                candidates.insert(0,("Most frequent",DummyClassifier(strategy="most_frequent")))
            for title,model in candidates:
                model.fit(xt,yt)
                start = time.perf_counter()
                pred = model.predict(xv)
                ms = 1000*(time.perf_counter()-start)/len(yv)
                val = metrics(yv,pred,selection_classes)
                info["comparisons"].append(dict(algorithm=title,features=mode,columns=xt.shape[1],ms=ms,
                                                settings=model.get_params(),validation=val))
                if title=="Most frequent":
                    continue
                # Stable tie-break favours fewer features then fixed candidate order.
                score = (val["macro_f1"],-xt.shape[1])
                if best is None or score>best[0]:
                    correct = margins(model,xv)[np.asarray(yv)==pred]
                    threshold = float(np.quantile(correct,0.10)) if len(correct) else float(np.finfo(float).max)
                    best = score,dict(algorithm=title,features=mode,transform=transform,model=model,threshold=threshold)
        chosen = tasks[task] = best[1]
        info["selected"] = {k:chosen[k] for k in ("algorithm","features","threshold")}
        info["classes"] = classes
        for part in ("test_public","test_personal"):
            if not rows[part]:
                continue
            ids,y = zip(*rows[part])
            features = [feature_rows[i] for i in ids]
            x = chosen["transform"].transform(features)
            model = chosen["model"]
            start = time.perf_counter()
            p = model.predict(x)
            confidence = margins(model,x)
            latency = 1000*(time.perf_counter()-start)/len(ids)
            accept = confidence>=chosen["threshold"]
            summary = metrics(y,p,classes)
            summary.update(coverage=float(np.mean(accept)),accepted=int(accept.sum()),inference_ms=latency,
                           accepted_metrics=metrics(np.asarray(y)[accept],p[accept],classes) if accept.any() else None)
            info["tests"][part] = summary
        pseudo_test = [d for d in docs if d["status"]["split"].startswith("test") and d["id"] in feature_rows
                       and d["labels"]["target_provenance"][task]["origin"]=="pseudo"
                       and splits.eligible(d,task,allow_pseudo=True) is not None]
        if pseudo_test:
            x = chosen["transform"].transform([feature_rows[d["id"]] for d in pseudo_test])
            info["machine_labeled_test_agreement"] = metrics(
                [splits.eligible(d,task,allow_pseudo=True) for d in pseudo_test],chosen["model"].predict(x),classes)
            info["machine_labeled_test_agreement"]["interpretation"] = "Agreement with held-out machine labels, without independent human verification."
    if "content_type" not in tasks:
        raise ValueError("Content training needs eligible labels for two training classes and a validation group. Existing models preserved.")
    ident = hashlib.sha256(json.dumps(dict(rows=[(d["id"],d["labels"],d["status"]["split"]) for d in docs],
             feature_version=VERSION,experiment="eight-content-classes-v3"),sort_keys=True).encode()).hexdigest()[:16]
    artifact = dict(format=1,training_id=ident,sklearn=sklearn.__version__,features=VERSION,tasks=tasks)
    buf = io.BytesIO()
    joblib.dump(artifact,buf,compress=3)
    audit["training_id"] = ident
    atomic(model_dir/"classifier.joblib",buf.getvalue())
    atomic(report_dir/"results.json",json.dumps(audit,indent=2,allow_nan=False).encode())
    return audit

def load_model(path=None):
    p = Path(path or default_model_path())
    if not p.is_file():
        raise ValueError("Trained model missing. Run .venv/bin/python -m classifier train.")
    model = joblib.load(p)
    if model.get("format")!=1 or compatible_version(model.get("features",""))!=compatible_version(VERSION) or model.get("sklearn")!=sklearn.__version__:
        raise ValueError("Incompatible model or feature versions. Retrain in this environment.")
    return model
