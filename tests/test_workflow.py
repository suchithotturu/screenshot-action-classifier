"""Isolated tests with an injected model; no inference or personal files."""
import csv,io,json
from pathlib import Path
import pytest
from pydantic import ValidationError
from PIL import Image
from classifier import batch,store
from classifier.model import Prediction,policy
from classifier.reports import render

def prediction(**updates):
    fields=dict(image_type="screenshot",category="ATTEND_REGISTER",title="Workshop",evidence="Register button",evidence_text="Workshop December 12, 2026",suggested_action="Register",entities=[],visible_url=None,original_date_text="December 12, 2026",date_role="event",needs_review=False,uncertainty_reasons=[])
    return Prediction(**(fields|updates))

def test_policy():
    with pytest.raises(ValidationError): prediction(category="URGENT")
    assert policy(prediction())["normalized_due_date"] is None
    assert policy(prediction(date_role="deadline"))["normalized_due_date"]=="2026-12-12"
    partial=policy(prediction(original_date_text="December 12",date_role="deadline"))
    assert partial["normalized_due_date"] is None and partial["needs_review"]
    assert policy(prediction(category="REFERENCE"))["suggested_action"] is None
    assert policy(prediction(image_type="ordinary_photo"))["suggested_action"] is None
    assert policy(prediction(visible_url="https://example.com"))["visible_url"] is None

def test_run_recovery_and_exports(tmp_path,monkeypatch):
    monkeypatch.setattr(store,"DATA",tmp_path/"db")
    config={"prompt":"a","side":1200}
    monkeypatch.setattr(batch,"provenance",lambda:config)
    root=tmp_path/"images"; root.mkdir()
    img=Image.new("RGB",(60,50),"green"); img.save(root/"=offer.png")
    (root/"copy.png").write_bytes((root/"=offer.png").read_bytes())
    (root/"bad.jpg").write_bytes(b"broken")
    (root/"sym.png").symlink_to(root/"copy.png")
    c=store.db(); run=batch.create(c,root)
    def stop_after_one(image):
        with c: c.execute("UPDATE runs SET cancel=1 WHERE id=?",(run,))
        return policy(prediction())
    monkeypatch.setattr(batch,"classify",stop_after_one)
    batch.worker(run)
    assert [r["state"] for r in store.records(c,run)]==["analyzed","pending","pending","skipped"]
    c.execute("UPDATE runs SET cancel=0 WHERE id=?",(run,)); c.commit()
    batch.worker(run)
    rows=store.records(c,run)
    assert [r["state"] for r in rows]==["analyzed","error","duplicate","skipped"]
    key=rows[0]["image_id"]
    for status in ("completed","dismissed"):
        store.correct(c,key,"VISIT","<script>test</script>",status)
        c.close(); c=store.db()
        assert store.records(c,run)[0]["correction"]["status"]==status
    def unexpected(_): pytest.fail("cache miss")
    monkeypatch.setattr(batch,"classify",unexpected)
    again=batch.create(c,root); batch.worker(again)
    assert store.records(c,again)[0]["state"]=="cached"
    config["prompt"]="b"
    monkeypatch.setattr(batch,"classify",lambda raw:policy(prediction(category="REFERENCE")))
    changed=batch.create(c,root); batch.worker(changed)
    exported=json.loads(render(c,changed,"json"))["items"][0]
    assert exported["prediction"]["category"]=="REFERENCE" and exported["effective"]["category"]=="VISIT"
    html=render(c,changed,"html")
    assert "<script>" not in html and "&lt;script&gt;" in html and "data:image/jpeg;base64," in html
    first=next(csv.DictReader(io.StringIO(render(c,changed,"csv"))))
    assert first["source_filename"]=="'=offer.png" and str(tmp_path) not in html

def test_invalid_generation_has_bounded_retry(monkeypatch):
    from classifier import model
    calls=[]
    def invalid(*args):
        calls.append(1)
        return {"message":{"content":"{}"}}
    monkeypatch.setattr(model,"api",invalid)
    with pytest.raises(ValueError,match="Invalid structured output"):
        model.classify(b"test fixture")
    assert len(calls)==2
