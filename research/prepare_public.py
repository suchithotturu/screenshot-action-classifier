"""Prepare a deterministic, private research subset from the authors' archives."""
import csv
import hashlib
import io
import json
import re
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
from classifier.features import atomic
from classifier.model import ROOT

DATA = ROOT/"data"
RAW = DATA/"raw"
MAPPING = {"Chat":"chat", "Map":"map_place", "Setting":"app_screen", "news":"article_reading"}
def rank(key):
    return hashlib.sha256(("513/"+key).encode()).hexdigest()

def prepare():
    index = []
    def add(raw, **src):
        path = Path("public-input")/(src["id"]+".jpg")
        atomic(DATA/path, raw)
        index.append(dict(path=str(path), **src))

    with zipfile.ZipFile(RAW/"masc.zip") as z:
        labels = list(csv.DictReader(io.StringIO(z.read("Labels.csv").decode())))
        unique_ids = Counter(row["Screen Id"] for row in labels)
        for topic in ("Chat","Map","Setting"):
            count = Counter()
            for row in sorted(labels, key=lambda r: rank(r["Screen Id"])):
                sid = row["Screen Id"]
                if row["class"] != topic or unique_ids[sid] != 1:
                    continue
                tree = json.loads(z.read(f"MASC_Json/MASC_Json/Full/{topic}/{sid}.json"))
                app = tree.get("activity_name","").split("/")[0]
                if not app or count[app] >= 8:
                    continue
                add(z.read(f"MASC_UI/MASC_UI/{topic}/{sid}.jpg"), id="masc_"+sid, dataset="masc",
                    rico_id=sid, package=app, source_class=topic, content_type=MAPPING[topic],
                    is_screenshot=True, license="MIT (MASC release)", url="https://github.com/Ali-Aahmed/MASC-Dataset")
                count[app] += 1
                if sum(count.values()) >= 180:
                    break
    # Exclude flagged screenshot issues; recover the dominant app resource namespace for grouping only.
    with zipfile.ZipFile(RAW/"enrico-screens.zip") as images, zipfile.ZipFile(RAW/"enrico-hierarchies.zip") as hier:
        with (RAW/"issues.csv").open() as f:
            issues = {r["screen_id"] for r in csv.DictReader(f) if r["source"]=="screenshot"}
        with (RAW/"design_topics.csv").open() as f:
            for r in csv.DictReader(f):
                sid = r["screen_id"]
                if sid in issues or r["topic"] != "news":
                    continue
                tree = hier.read(f"hierarchies/{sid}.json").decode()
                packages = Counter(p for p in re.findall(r'"resource-id":\s*"([^":]+):id/',tree)
                                   if "." in p and p != "com.android.systemui")
                if not packages:
                    continue
                add(images.read(f"screenshots/{sid}.jpg"), id="enrico_"+sid, dataset="enrico",
                    rico_id=sid, package=packages.most_common(1)[0][0], source_class="news",
                    is_screenshot=True, content_type=MAPPING["news"], license="unverified",
                    url="https://userinterfaces.aalto.fi/enrico/")
    with tarfile.open(RAW/"flowers.tgz") as t:
        license_text = t.extractfile("flower_photos/LICENSE.txt").read().decode()
        atomic(RAW/"flower-attribution.txt",license_text.encode())
        rows = [s for s in license_text.splitlines() if ".jpg CC-BY by " in s and " - http" in s]
        authors = Counter()
        chosen = {}
        for line in sorted(rows,key=rank):
            name, credit = line.split(" CC-BY by ",1)
            author, url = credit.rsplit(" - ",1)
            if authors[author] >= 5:
                continue
            chosen["flower_photos/"+name] = dict(id="flower_"+Path(name).stem, dataset="tf_flowers", license="CC BY 2.0",
                photographer=author, url=url, source_class=name.split("/")[0], is_screenshot=False,
                content_type=None)
            authors[author] += 1
            if sum(authors.values()) == 360:
                break
        for member in t.getmembers():
            if member.name in chosen:
                add(t.extractfile(member).read(), **chosen[member.name])
    atomic(RAW/"public-index.json",json.dumps(index,indent=2).encode())
    print(len(index), Counter(r["dataset"] for r in index),flush=True)

if __name__ == "__main__":
    prepare()
