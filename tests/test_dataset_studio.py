import io
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest
from classifier import annotation, dataset, features, learning, production, splits, store
from classifier.restore import restore
from classifier.schema import Draft, Snapshot

@pytest.fixture
def db(tmp_path,monkeypatch):
    monkeypatch.setattr(store,"DATA",tmp_path/"library")
    c = dataset.connect()
    yield c
    c.close()

def pixels(seed,fmt="PNG"):
    rgb = np.random.default_rng(seed).integers(0,255,(48,80,3),dtype=np.uint8)
    out = io.BytesIO()
    Image.fromarray(rgb).save(out,format=fmt)
    return out.getvalue()

def add(c,n,**source):
    return dataset.ingest_one(c,pixels(n),{"dataset":"personal","owner":"A","license":"private",**source})[0]

def test_ingest_drops_alias_and_restore(db,tmp_path,monkeypatch):
    folder=tmp_path/"input"
    folder.mkdir()
    (folder/"one.png").write_bytes(pixels(4))
    (folder/"two.png").write_bytes(pixels(4))
    (folder/"bad.png").write_bytes(b"unreadable")
    for _ in range(2):
        count=dataset.ingest_folder(db,folder,"personal","A")
        assert count["errors"]==1
    assert len(dataset.all_docs(db))==1 and db.execute("SELECT count(*) FROM drops").fetchone()[0]==1
    doc=dataset.all_docs(db)[0]
    sid=doc["id"]
    assert len(doc["source"]["origins"])==2
    d=Draft(is_screenshot=True,content_type="other",purpose="act_on_it",still_useful=False).model_dump()
    dataset.batch_edit(db,[(sid,d,"approve","A","owner",True,0)])
    export=dataset.export(db)
    one=Snapshot.model_validate_json(export.read_text()).model_dump()
    assert one["labels"]["still_useful"] is False and one["labels"]["target_provenance"]["purpose"]["origin"]=="owner"
    assert (export.parent/one["image"]["path"]).exists()
    monkeypatch.setattr(store,"DATA",tmp_path/"restored")
    c=dataset.connect()
    assert restore(c,export)=={"added":1,"existing":0}
    assert restore(c,export)=={"added":0,"existing":1}
    assert dataset.get(c,sid)["labels"]==one["labels"]
    assert c.execute("SELECT count(*) FROM revisions").fetchone()[0]==1

def test_blind_lock_owner_and_atomic_batch(db):
    a,b=add(db,1),add(db,2)
    dataset.reserve_blind(db,a)
    with pytest.raises(ValueError,match="blind"):
        dataset.expose(db,a)
    assert dataset.current_draft(dataset.get(db,a),True)==Draft()
    d=Draft(is_screenshot=True,content_type="chat",purpose="act_on_it").model_dump()
    with pytest.raises(ValueError,match="owner"):
        dataset.batch_edit(db,[(a,d,"approve","B","owner",True,0)])
    with pytest.raises(ValueError,match="another window"):
        dataset.batch_edit(db,[(a,d,"approve","A","owner",True,0),(b,d,"approve","A","owner",True,8)])
    assert dataset.get(db,a)["annotations"]["review"]["revision"]==0
    dataset.batch_edit(db,[(a,d,"approve","A","owner",True,0)])
    dataset.expose(db,b)
    with pytest.raises(ValueError,match="visible"):
        dataset.current_draft(dataset.get(db,b),True)
    dataset.batch_edit(db,[(a,d,"undo","A","owner",True,1)])
    photo={**d,"is_screenshot":False}
    dataset.batch_edit(db,[(a,photo,"approve","A","owner",True,2)])
    assert dataset.get(db,a)["labels"]["content_type"]=="chat"
    assert all(dataset.get(db,a)["labels"][k] is None for k in ("purpose","still_useful"))

def test_qwen_retries_manual_and_immutable(db,monkeypatch):
    sid=add(db,8)
    monkeypatch.setattr(annotation.model,"provenance",lambda:{"runtime":"test","digest":"fixture"})
    calls=[]
    def invalid(*args):
        calls.append(1)
        return {"message":{"content":"{}"}}
    monkeypatch.setattr(annotation.model,"api",invalid)
    assert annotation.propose(db,sid)["status"]=="failed" and len(calls)==2
    q=Draft(is_screenshot=True,content_type="app_screen",description="Fixture suggestion").model_dump()
    proposal={k:q[k] for k in annotation.Proposal.model_fields}
    monkeypatch.setattr(annotation.model,"api",lambda *a:{"message":{"content":json.dumps(proposal)}})
    original=annotation.propose(db,sid)
    assert original["status"]=="ok" and dataset.get(db,sid)["labels"]["content_type"] is None
    for event,rev in [("save",0),("skip",1),("reset",2),("approve",3)]:
        dataset.batch_edit(db,[(sid,{**q,"content_type":"chat"},event,"Reviewer 1","reviewer",False,rev)])
    assert annotation.propose(db,sid)==original
    assert db.execute("SELECT count(*) FROM proposals").fetchone()[0]==2

def test_streamlit_autosave_reopen_and_approve(db):
    sid=add(db,6,dataset="fixture")
    original=Draft(is_screenshot=True,content_type="app_screen",description="Original proposal").model_dump()
    with db:
        doc=dataset.get(db,sid)
        doc["annotations"]["qwen"]={"status":"ok","fields":{k:original[k] for k in annotation.Proposal.model_fields}}
        dataset.put(db,doc)
    def editor():
        app=AppTest.from_file(str(features.ROOT/"app.py"),default_timeout=20).run()
        app.radio[0].set_value("Dataset Annotation").run()
        app.toggle[0].set_value(True).run()
        assert not app.exception
        return app
    app=editor()
    next(x for x in app.selectbox if x.label=="Dominant content").set_value("shopping_product").run()
    assert not app.exception and dataset.get(db,sid)["annotations"]["draft"]["content_type"]=="shopping_product"
    app=editor()
    assert next(x for x in app.selectbox if x.label=="Dominant content").value=="shopping_product"
    next(b for b in app.button if b.label=="Approve 1 displayed records").click().run()
    assert not app.exception
    verified=json.loads(dataset.export(db).read_text())
    assert verified["labels"]["content_type"]=="shopping_product"
    assert verified["annotations"]["qwen"]["fields"]["content_type"]=="app_screen"

def test_split_duplicate_and_training_only_transform(db):
    ids=[add(db,i,dataset="masc",owner=None,is_screenshot=True,content_type="chat" if i%2 else "map_place",package="app"+str(i)) for i in range(20)]
    # JPEG re-encoding an existing image is a separate SHA but the same visual group.
    out=io.BytesIO()
    Image.open(io.BytesIO(pixels(2))).save(out,"JPEG",quality=98,subsampling=0)
    copy,_=dataset.ingest_one(db,out.getvalue(),{"dataset":"public","is_screenshot":True,"content_type":"map_place"})
    prepared={d["id"]:d for d in splits.prepare(db)}
    assert prepared[ids[2]]["status"]["duplicate_group_id"]==prepared[copy]["status"]["duplicate_group_id"]
    assert prepared[ids[2]]["status"]["split"]==prepared[copy]["status"]["split"]
    def row(s):
        return {"ocr":{"text":s},"numeric":[0]*len(features.NUMERIC),"embedding":[0]*512}
    f=learning.Fusion("fusion").fit([row("train words"),row("train labels")])
    before=f.names.copy()
    x=f.transform([{**row("private_test_token"),"owner":"forbidden","description":"forbidden"}])
    assert x.shape[1]==len(before) and f.names==before and "word:private_test_token" not in f.names

def test_missing_model_and_uncertainty(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match="missing"):
        learning.load_model(tmp_path/"no-model")
    monkeypatch.setattr(production,"extract",lambda *a:{"status":"ok","sha256":"x","ocr":{"text":""}})
    r=production.predict_image({"tasks":{}},b"")
    assert r["route"]=="other" and not r["actions"] and "Content classifier unavailable" in r["reasons"][0]
    monkeypatch.setattr(production,"decision",lambda *a:{"label":"True","accepted":False})
    r=production.predict_image({"tasks":{"is_screenshot":"test"}},b"")
    assert r["route"]=="other" and not r["actions"]

def test_legacy_ui_loads(db,monkeypatch):
    monkeypatch.setattr("classifier.model.provenance",lambda:{"quantization":"test"})
    app=AppTest.from_file(str(features.ROOT/"app.py")).run()
    app.radio[0].set_value("Legacy inbox").run()
    assert not app.exception and "Screenshot inbox" in app.title.values

def test_legacy_master_keeps_votes_captions_and_ids(db,tmp_path):
    from classifier.legacy_manifest import import_master
    raw=pixels(31)
    (tmp_path/"screen.png").write_bytes(raw)
    source={"dataset":"screen2words","rico_id":"31","captions":["Fixture caption"],"license":"unverified"}
    original={"id":"s2w_31","source":source,"image":{"path":"screen.png"},
              "labels":{"content_type":"other","votes":[{"source":"original","value":"other"}],
                        "target_provenance":{"content_type":"source","is_screenshot":"source"},"is_screenshot":True},
              "status":{"split":"validation"}}
    manifest=tmp_path/"legacy.jsonl"
    manifest.write_text(json.dumps(original))
    assert import_master(db,manifest)["added"]==1
    assert import_master(db,manifest)["existing"]==1
    saved=dataset.get(db,"s2w_31")
    assert saved["source"]["origins"][0]["legacy_record"]==original
    assert saved["labels"]["content_type"]=="other" and saved["labels"]["votes"]==original["labels"]["votes"]
    assert saved["source"]["captions"]==source["captions"]
    assert db.execute("SELECT partition FROM splits").fetchone()[0]=="validation"


def test_review_queue_mixes_category_blocks_and_blind_order_hides_targets():
    from copy import deepcopy
    from classifier.studio import mix_review_queue
    docs = [{"id":f"{label}-{i}","labels":{"content_type":label,"is_screenshot":True}}
            for label in ("chat","map_place","app_screen") for i in range(12)]
    ordered = mix_review_queue(docs,False)
    assert len(ordered)==len(docs)
    assert {d["id"] for d in ordered}=={d["id"] for d in docs}
    for start in range(0,len(ordered),3):
        assert {d["labels"]["content_type"] for d in ordered[start:start+3]}=={"chat","map_place","app_screen"}
    changed = deepcopy(docs)
    for doc in changed:
        doc["labels"]["content_type"]="other"
    assert [d["id"] for d in mix_review_queue(docs,True)]==[d["id"] for d in mix_review_queue(changed,True)]


def test_pseudo_targets_require_opt_in_and_owner_rules_still_apply():
    doc={"labels":{"content_type":"chat","is_screenshot":True,"purpose":"act_on_it",
                   "target_provenance":{"content_type":{"origin":"pseudo"},"purpose":{"origin":"pseudo"}}},
         "status":{"fixture":False}}
    assert splits.eligible(doc,"content_type") is None
    assert splits.eligible(doc,"content_type",allow_pseudo=True)=="chat"
    assert splits.eligible(doc,"purpose",allow_pseudo=True) is None


def test_compact_qwen_annotation_and_receipt_photo_targets(db,monkeypatch):
    sid=add(db,70,dataset="cord",is_screenshot=False,content_type="receipt_confirmation")
    monkeypatch.setattr(annotation.model,"provenance",lambda:{"runtime":"test","digest":"fixture"})
    monkeypatch.setattr(annotation.model,"api",lambda *args:{"message":{"content":json.dumps({
        "content_type":"receipt_confirmation","description":"A paid restaurant receipt."})}})
    q=annotation.propose(db,sid,compact=True)
    assert q["status"]=="ok" and q["fields"]["content_type"]=="receipt_confirmation"
    assert q["fields"]["is_screenshot"] is False and q["fields"]["purpose"] is None
    assert splits.eligible(dataset.get(db,sid),"content_type")=="receipt_confirmation"


def test_knn_and_naive_bayes_scores_and_text_inputs():
    from sklearn.naive_bayes import ComplementNB
    from sklearn.neighbors import KNeighborsClassifier
    def row(text,number):
        return {"ocr":{"text":text},"numeric":[number]*len(features.NUMERIC),"embedding":[number]*512}
    rows=[row("chat message reply",-5),row("chat message hello",-3),
          row("map place directions",-4),row("map route directions",-2)]
    transform=learning.Fusion("text").fit(rows)
    x=transform.transform(rows)
    assert (x.data>=0).all() and all(name.startswith("word:") for name in transform.names)
    for model in [ComplementNB(),KNeighborsClassifier(n_neighbors=1,metric="cosine",algorithm="brute")]:
        model.fit(x,["chat","chat","map_place","map_place"])
        assert np.isfinite(learning.margins(model,x)).all()
        assert model.predict(x).tolist()==["chat","chat","map_place","map_place"]
    blank=learning.Fusion("text").fit([row("",0),row("",0)])
    assert blank.transform([row("",0)]).shape==(1,1)


def test_filter_keeps_latest_run_when_creation_times_match(db,tmp_path):
    for route,count in [("chat",1),("shopping_product",2)]:
        body=dict(directory=str(tmp_path),supported=[route],purpose_available=False,
                  items=[dict(path=f"{i}.png",route=route,status="ok",tasks={},actions=[]) for i in range(count)])
        with db:
            db.execute("INSERT INTO prediction_runs(created,body) VALUES(?,?)",("2026-11-26T12:00:00-05:00",json.dumps(body)))
    app=AppTest.from_file(str(features.ROOT/"app.py")).run()
    next(x for x in app.selectbox if x.label=="Filter images").set_value("shopping_product").run()
    assert not app.exception
    assert next(x for x in app.selectbox if x.label=="Saved classifications").value==2
    assert app.subheader.values==["Products & offers"]*2
