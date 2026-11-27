import argparse, json, time
from pathlib import Path
from . import store
from .ingest import enumerate_files, read_image, prepare
from .model import provenance, classify
from .locking import exclusive

def create(c,folder):
    root=Path(folder).expanduser().resolve(strict=True)
    if not root.is_dir(): raise ValueError("Select an existing folder.")
    config=json.dumps(provenance(),sort_keys=True)
    with c:
        run=c.execute("INSERT INTO runs(root,started,config,status) VALUES(?,?,?,'queued')",(str(root),store.now(),config)).lastrowid
        c.executemany("INSERT INTO items(run,path,state,note) VALUES(?,?,?,?)",((run,str(p.relative_to(root)),s,n) for p,s,n in enumerate_files(root)))
    return run

def process(run):
    c=store.db()
    with (store.DATA/"worker.lock").open("a") as lock:
        exclusive(lock)
        row=c.execute("SELECT * FROM runs WHERE id=?",(run,)).fetchone()
        cfg=row["config"]
        if json.loads(cfg)!=provenance(): raise ValueError("Settings changed. Start a new analysis.")
        root=Path(row["root"])
        seen={r[0] for r in c.execute("SELECT image_id FROM items WHERE run=? AND state IN ('analyzed','cached')",(run,))}
        c.execute("UPDATE runs SET status='running' WHERE id=?",(run,)); c.commit()
        for item in c.execute("SELECT path FROM items WHERE run=? AND state='pending' ORDER BY path",(run,)).fetchall():
            if c.execute("SELECT cancel FROM runs WHERE id=?",(run,)).fetchone()[0]: break
            key=note=None
            start=time.monotonic()
            try:
                key,raw=read_image(root,root/item[0])
                if key in seen: state="duplicate"
                elif c.execute("SELECT 1 FROM cache WHERE image_id=? AND config=?",(key,cfg)).fetchone(): state="cached"
                else:
                    result=classify(prepare(raw,key,store.DATA))
                    c.execute("INSERT OR IGNORE INTO cache VALUES(?,?,?,?)",(key,cfg,json.dumps(result),store.now()))
                    state="analyzed"
                seen.add(key)
            except Exception as e:
                state="error"
                note=(e.strerror if isinstance(e,OSError) and e.strerror else str(e)).replace(str(root),"[folder]")
                if "BytesIO object" in note: note="Unreadable or corrupt image."
            with c: c.execute("UPDATE items SET state=?,image_id=?,note=?,seconds=? WHERE run=? AND path=?",(state,key,note,round(time.monotonic()-start,3),run,item[0]))
        with c: c.execute("UPDATE runs SET status=CASE WHEN cancel=1 THEN 'cancelled' ELSE 'finished' END WHERE id=?",(run,))
def worker(run):
    try: process(run)
    except Exception as e:
        with store.db() as c: c.execute("UPDATE runs SET status=? WHERE id=?",("Stopped: "+str(e),run))
        raise
if __name__=="__main__":
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--folder"); g.add_argument("--run",type=int)
    args=p.parse_args()
    run=args.run or create(store.db(),args.folder)
    worker(run)
    print("Run",run)
