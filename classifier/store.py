import json, sqlite3, os
from pathlib import Path
from datetime import datetime
from .model import ROOT,TZ
DATA=Path(os.getenv("CLASSIFIER_DATA",ROOT/"data"))
def db():
    DATA.mkdir(exist_ok=True,parents=True)
    c=sqlite3.connect(DATA/"library.db",timeout=20)
    c.row_factory=sqlite3.Row
    c.executescript("""
    CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY,root TEXT,started TEXT,config TEXT,status TEXT, cancel INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS items(run INTEGER,path TEXT,image_id TEXT,state TEXT,note TEXT, seconds REAL, PRIMARY KEY(run,path));
    CREATE TABLE IF NOT EXISTS cache(image_id TEXT,config TEXT,prediction TEXT,created TEXT,PRIMARY KEY(image_id,config));
    CREATE TABLE IF NOT EXISTS edits(image_id TEXT PRIMARY KEY,category TEXT,action TEXT,status TEXT,updated TEXT);
    """)
    return c
def now(): return datetime.now(TZ).isoformat(timespec="seconds")
def correct(c,key,cat,action,status):
    c.execute("INSERT OR REPLACE INTO edits VALUES(?,?,?,?,?)",(key,cat,action,status,now()))
    c.commit()
def records(c,run):
    sql="""SELECT i.*, c.prediction,c.created,e.category,e.action,e.status,e.updated FROM items i JOIN runs r ON r.id=i.run
    LEFT JOIN cache c ON c.image_id=i.image_id AND c.config=r.config
    LEFT JOIN edits e ON e.image_id=i.image_id WHERE run=? ORDER BY path"""
    rows=[]
    for r in c.execute(sql,(run,)):
        p=json.loads(r["prediction"]) if r["prediction"] else None
        edit=dict(category=r["category"],suggested_action=r["action"],status=r["status"],updated=r["updated"]) if r["updated"] else None
        effective=dict(p or {})
        if edit and p: effective.update(category=edit["category"],suggested_action=edit["suggested_action"])
        if effective.get("category") in ("REFERENCE","OTHER"): effective["suggested_action"]=None
        day=effective.get("normalized_date")
        today=datetime.now(TZ).date().isoformat()
        date_status=("past" if day<today else "today" if day==today else "upcoming") if day else None
        rows.append(dict(source_filename=Path(r["path"]).name,source_path=r["path"],image_id=r["image_id"],state=r["state"],note=r["note"],seconds=r["seconds"],analyzed_at=r["created"],prediction=p,correction=edit,effective=effective,date_status=date_status))
    return rows
