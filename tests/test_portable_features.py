import numpy as np
import pytest
from PIL import Image
from classifier import features, learning


def test_rapidocr_boxes_normalize_to_shared_feature_contract(monkeypatch):
    monkeypatch.setattr(features,"OCR_BACKEND","rapidocr")
    def engine(image):
        assert image.shape==(100,200,3)
        assert list(image[0,0])==[0,0,255]
        return [([[20,10],[180,10],[180,40],[20,40]],"Add to cart",0.93)],None
    monkeypatch.setattr(features,"portable_ocr",lambda *args:engine)
    lines=features.ocr(Image.new("RGB",(200,100),"red"))
    assert lines[0]["box"]==pytest.approx([0.1,0.1,0.8,0.3])
    assert lines[0]["text"]=="Add to cart" and lines[0]["confidence"]==0.93


def test_ocr_backend_selects_matching_model_and_rejects_mismatch(tmp_path,monkeypatch):
    import joblib
    current=learning.load_model()
    current["features"]=current["features"].replace("letterbox224","changed-ocr-contract")
    p=tmp_path/"wrong-backend.joblib"
    joblib.dump(current,p)
    with pytest.raises(ValueError,match="Incompatible"):
        learning.load_model(p)
    monkeypatch.setattr(learning,"OCR_BACKEND","rapidocr")
    assert learning.default_model_path().parts[-2:]==("rapidocr","classifier.joblib")


def test_source_read_works_without_posix_no_follow_flag(tmp_path,monkeypatch):
    from classifier import ingest
    import os
    p=tmp_path/"image.png"
    payload=b"binary\r\n\x1a\x00content"
    p.write_bytes(payload)
    monkeypatch.delattr(os,"O_NOFOLLOW",raising=False)
    assert ingest.read_image(tmp_path,p)[1]==payload


def test_corrupt_image_errors_are_stable_after_model_reload(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from classifier import dataset, production, store
    from classifier.schema import NAMES
    monkeypatch.setattr(store,"DATA",tmp_path/"library")
    folder=tmp_path/"input"
    folder.mkdir()
    (folder/"broken.jpg").write_bytes(b"not an image")
    bundle=dict(training_id="fixture",tasks={"content_type":{"model":SimpleNamespace(classes_=list(NAMES))}})
    c=dataset.connect()
    for _ in range(2):
        _,body=production.predict_folder(c,folder,bundle=bundle)
        assert body["items"][0]["reasons"]==["Unreadable or corrupt image."]
