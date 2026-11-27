"""Verify a fresh installation without a training database or local Qwen service."""
import json
import sys
from collections import Counter
from classifier import dataset, model, production
from classifier.features import ROOT, OCR_BACKEND
from classifier.learning import load_model
from classifier.schema import NAMES


def main():
    def offline(event, args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise AssertionError("Network connection attempted during shared-install check")
    def no_qwen(*args, **kwargs):
        raise AssertionError("Qwen called during shared-install check")
    sys.addaudithook(offline)
    model.api = no_qwen
    bundle = load_model()
    assert set(bundle["tasks"]["content_type"]["model"].classes_) == set(NAMES)
    c = dataset.connect()
    before = c.execute("SELECT count(*) FROM samples").fetchone()[0]
    _, body = production.predict_folder(c, ROOT / "demo", bundle=bundle)
    counts = Counter(r["status"] for r in body["items"])
    assert counts == {"ok": 10, "error": 1, "skipped": 1}, counts
    assert next(r for r in body["items"] if r["status"] == "error")["path"] == "broken.jpg"
    _, again = production.predict_folder(c, ROOT / "demo", bundle=load_model())
    for first,second in zip(body["items"],again["items"],strict=True):
        assert {k:v for k,v in first.items() if k!="seconds"}=={k:v for k,v in second.items() if k!="seconds"}
    assert c.execute("SELECT count(*) FROM samples").fetchone()[0] == before
    print(json.dumps(dict(training_id=bundle["training_id"], learned_categories=len(NAMES),ocr_backend=OCR_BACKEND,
                         statuses=dict(counts), qwen_calls=0, network_connections=0,
                         training_records_before=before, training_records_after=before,identical_after_reload=True), indent=2))


if __name__ == "__main__":
    main()
