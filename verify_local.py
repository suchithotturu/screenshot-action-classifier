"""Real demo verification. Restrict Python TCP connections to the local model endpoint."""
import csv,io,json,socket,sys
from html.parser import HTMLParser
from collections import Counter
from pathlib import Path
from classifier import batch,store
from classifier.model import ROOT,classify
from classifier.reports import render
def restrict(event,args):
    if event=="socket.connect" and args[1]!=("127.0.0.1",11434):
        raise RuntimeError("Test blocked a non-model connection.")
sys.addaudithook(restrict)
if __name__=="__main__":
    c=store.db()
    print("Uncached restricted-client inference:",classify((ROOT/"demo"/"01-shirt.png").read_bytes())["title"])
    run=batch.create(c,ROOT/"demo"); batch.worker(run)
    rows=store.records(c,run)
    counts=Counter(r["state"] for r in rows)
    print(run,counts)
    assert counts==dict(cached=9,duplicate=1,skipped=1,error=1)
    samples=ROOT/"samples"; samples.mkdir(exist_ok=True)
    for fmt in ("json","csv","html"):
        path=samples/("demo."+fmt)
        path.write_text(render(c,run,fmt))
        content=path.read_text()
        assert str(ROOT) not in content
        if fmt=="json": assert len(json.loads(content)["items"])==12
        if fmt=="csv": assert len(list(csv.DictReader(io.StringIO(content))))==12
        if fmt=="html":
            HTMLParser().feed(content)
            assert "data:image/jpeg;base64," in content and "Café" in content
