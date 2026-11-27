import base64,csv,html,io,json
from pathlib import Path
from . import store
from .model import ROOT

def render(c,run,fmt):
    r=c.execute("SELECT * FROM runs WHERE id=?",(run,)).fetchone()
    meta=dict(run=run,started=r["started"],status=r["status"],dataset="Synthetic demo with NASA sample portrait" if Path(r["root"])==ROOT/"demo" else "Selected folder",provenance=json.loads(r["config"]))
    rows=store.records(c,run)
    if fmt=="json": return json.dumps(dict(**meta,items=rows),indent=2,ensure_ascii=False)
    if fmt=="csv":
        out=io.StringIO()
        fields=["dataset","provenance","source_filename","source_path","image_id","state","note","seconds","analyzed_at","date_status","category","suggested_action","prediction","correction","effective"]
        writer=csv.DictWriter(out,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            data=dict(**row,dataset=meta["dataset"],provenance=meta["provenance"],category=row["effective"].get("category"),suggested_action=row["effective"].get("suggested_action"))
            for k,v in data.items():
                if isinstance(v,dict): data[k]=json.dumps(v,ensure_ascii=False)
                elif isinstance(v,str) and (v.lstrip().startswith(("=","+","-","@")) or v.startswith(("\t","\r","\n"))): data[k]="'"+v
            writer.writerow(data)
        return out.getvalue()
    esc=lambda v:html.escape(str(v))
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Screenshot review</title><style>body{font:16px system-ui;color:#18372d;background:#f5f6f0;max-width:900px;margin:3em auto;padding:1em}article{background:white;border-radius:14px;padding:1.2em;margin:1em 0;overflow:auto}img{float:right;width:100px;margin:1em}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Screenshot review</h1>',esc(meta["dataset"]),'<details><summary>Run and model details</summary><pre>'+esc(json.dumps(meta,indent=2))+'</pre></details>']
    for row in rows:
        p=row["effective"]
        thumb=store.DATA/(str(row["image_id"])+".jpg")
        pic='<img alt="Image preview" src="data:image/jpeg;base64,'+base64.b64encode(thumb.read_bytes()).decode()+'">' if thumb.is_file() else ''
        page.append('<article>'+pic+'<small>'+esc(row["source_path"])+' · '+esc(row["state"])+'</small><h2>'+esc(p.get("title") or row["source_filename"])+'</h2><p>'+esc(p.get("suggested_action") or p.get("category") or row["note"] or "")+'</p><p>'+esc(p.get("evidence",""))+'</p><details><summary>Evidence, prediction and corrections</summary><pre>'+esc(json.dumps(row,indent=2,ensure_ascii=False))+'</pre></details></article>')
    return ''.join(page)+'</html>'
