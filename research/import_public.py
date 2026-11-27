import json
from concurrent.futures import ThreadPoolExecutor
from classifier import dataset, store
from classifier.features import encoder, extract

def run():
    c = dataset.connect()
    index = json.loads((store.DATA/"raw/public-index.json").read_text())
    for row in index:
        src = row.copy()
        raw = (store.DATA/src.pop("path")).read_bytes()
        dataset.ingest_one(c,raw,src)
    docs = dataset.all_docs(c)
    print(f"{len(docs)} canonical images",flush=True)
    encoder()
    def one(doc):
        f = extract(dataset.image_path(doc).read_bytes(),store.DATA/"features")
        if f["status"]!="ok":
            raise RuntimeError(str({k:f[k].get("error") for k in ("ocr","clip")}))
    with ThreadPoolExecutor(4) as pool:
        for n,_ in enumerate(pool.map(one,docs),1):
            if n%100==0:
                print(n,flush=True)
if __name__=="__main__":
    run()
