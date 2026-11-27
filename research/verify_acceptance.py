"""Live Mac acceptance checks with isolated fixture edits and no changes to research labels."""
import io
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest
from classifier import dataset, features, store
from classifier.restore import restore

def verify():
    c = dataset.connect()
    items = dataset.all_docs(c)
    live = next(d for d in items if any(s.get("relative_name")=="01-shirt.png" for s in d["source"]["origins"]))
    original = live["annotations"]["qwen"]
    assert original["status"]=="ok" and original["fields"]["content_type"]=="shopping_product"
    aggregate = {"qwen_successes":sum((d["annotations"]["qwen"] or {}).get("status")=="ok" for d in items)}
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        raw = (features.ROOT/"demo/01-shirt.png").read_bytes()
        fresh = features.extract(raw,base/"cache")
        cached = features.extract(raw,base/"cache")
        repeated = features.extract(raw,base/"cache",refresh=True)
        assert fresh["status"]==cached["status"]==repeated["status"]=="ok"
        assert fresh["ocr"]["text"]==cached["ocr"]["text"]==repeated["ocr"]["text"] and "shirt" in fresh["ocr"]["text"].lower()
        assert np.allclose(fresh["embedding"],repeated["embedding"],atol=1e-6)
        assert np.isclose(np.linalg.norm(fresh["embedding"]),1)
        blank = io.BytesIO()
        Image.new("RGB",(330,400),"white").save(blank,"PNG")
        empty = features.extract(blank.getvalue(),base/"cache")
        assert empty["status"]=="ok" and empty["ocr"]["text"]=="" and empty["numeric"][-1]==1
        aggregate.update(clip_dimension=len(fresh["embedding"]),fresh_repeat_max_difference=float(np.max(np.abs(
                         np.array(fresh["embedding"])-repeated["embedding"]))),blank_ocr_status=empty["ocr"]["status"])
        # Exercise a second tile with actual text at both ends of a long image.
        tile = Image.open(io.BytesIO(raw)).convert("RGB")
        tall = Image.new("RGB",(tile.width,3100),"white")
        tall.paste(tile,(0,0))
        tall.paste(tile,(0,tall.height-tile.height))
        repeated_lines = features.ocr(tall)
        assert sum("Cotton camp shirt" in b["text"] for b in repeated_lines)==2
        aggregate["long_image_shirt_mentions"]=2
        src_data = store.DATA
        try:
            store.DATA = base/"review"
            test = dataset.connect()
            sid,_ = dataset.ingest_one(test,raw,dict(dataset="fixture",license="private"))
            with test:
                d=dataset.get(test,sid)
                d["annotations"]["qwen"]=original
                dataset.put(test,d)
            def open_review():
                view=AppTest.from_file(str(features.ROOT/"app.py"),default_timeout=30).run()
                view.radio[0].set_value("Dataset Annotation").run()
                view.toggle[0].set_value(True).run()
                assert not view.exception
                return view
            view=open_review()
            # A deliberate QA correction leaves capture provenance unknown for a rendered fixture.
            next(x for x in view.selectbox if x.label=="Screenshot capture").set_value("__unknown__").run()
            assert not view.exception
            view=open_review()
            next(b for b in view.button if b.label=="Approve").click().run()
            assert not view.exception
            test.close()
            test=dataset.connect()
            snap=dataset.export(test)
            record=json.loads(snap.read_text())
            assert record["labels"]["is_screenshot"] is None and record["annotations"]["review"]["state"]=="reviewed"
            assert record["annotations"]["qwen"]==original
            store.DATA=base/"restore"
            restored=dataset.connect()
            assert restore(restored,snap)["added"]==1
            assert dataset.get(restored,sid)["labels"]==record["labels"]
            aggregate.update(real_proposal_review_persisted=True,snapshot_restore=True)
        finally:
            store.DATA=src_data
    legacy=sqlite3.connect(store.DATA/"library.db")
    legacy.execute("ATTACH ? AS orig",(str(store.DATA/"library.pre-astra.db"),))
    counts={}
    for name in ("runs","items","cache","edits"):
        for a,b in (("main","orig"),("orig","main")):
            assert not legacy.execute(f"SELECT * FROM {a}.{name} EXCEPT SELECT * FROM {b}.{name}").fetchone()
        counts[name]=legacy.execute("SELECT count(*) FROM "+name).fetchone()[0]
    aggregate["legacy_rows_unchanged"]=counts
    overlap=c.execute("""SELECT count(*) FROM (SELECT group_id FROM splits GROUP BY group_id
                        HAVING count(DISTINCT partition)>1)""").fetchone()[0]
    assert overlap==0
    aggregate["split_group_overlaps"]=overlap
    features.atomic(features.ROOT/"reports/acceptance.json",json.dumps(aggregate,indent=2).encode())
    print(json.dumps(aggregate,indent=2))

if __name__=="__main__":
    verify()
