from collections import Counter
import fcntl,json,subprocess,sys
from pathlib import Path
import streamlit as st
from classifier import store
from classifier.batch import create
from classifier.model import ROOT,MODEL,CATEGORIES,provenance
from classifier.reports import render

st.set_page_config(page_title="Screenshot inbox",layout="wide")
import os
from classifier import studio
modes = ["Classify Images", "Dataset Annotation", "Legacy inbox"]
requested = st.query_params.get("mode", os.getenv("ASTRA_MODE","classify"))
mode = st.sidebar.radio("Workspace", modes,index={"annotation":1,"legacy":2}.get(requested,0))
if mode == "Classify Images":
    studio.classify()
    st.stop()
if mode == "Dataset Annotation":
    studio.annotation()
    st.stop()
st.title("Screenshot inbox")
st.caption("Turn visible details from saved screens into suggestions you can review.")
c=store.db()

@st.cache_data(ttl=15,show_spinner=False)
def readiness():
    try: return provenance()["quantization"]
    except Exception: return None
ready=readiness()
runs=c.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
def launch(run):
    with (store.DATA/"worker.log").open("a") as log:
        subprocess.Popen([sys.executable,"-m","classifier.batch","--run",str(run)],cwd=ROOT,stdout=log,stderr=log,start_new_session=True)
    st.session_state["selected"]=run
    st.rerun()
with st.sidebar:
    st.header("Image folder")
    st.caption("Analyze an exported phone folder, including subfolders.")
    folder=st.text_input("Local folder path",str(ROOT/"demo"),key="folder")
    def choose_folder():
        choice=subprocess.run(["osascript","-e",'POSIX path of (choose folder with prompt "Choose a folder of exported images")'],capture_output=True,text=True)
        if choice.returncode==0: st.session_state["folder"]=choice.stdout.strip()
    st.button("Choose folder…",on_click=choose_folder)
    st.caption(f"{'Ready' if ready else 'Unavailable'} · {MODEL}")
    if st.button("Analyze",type="primary",disabled=not ready):
        try: launch(create(c,folder))
        except Exception as e: st.error(str(e))
    if not ready: st.warning("Run setup.sh and reopen Launch.command.")
    if not runs: st.stop()
    ids=[r["id"] for r in runs]
    lookup={r["id"]:r for r in runs}
    run=st.selectbox("Saved analyses",ids,key="selected",format_func=lambda i:f"{Path(lookup[i]['root']).name} · Run {i}")
r=lookup[run]
if Path(r["root"])==ROOT/"demo": st.info("Synthetic demo screens and one public NASA portrait")
@st.fragment(run_every=2 if r["status"] in ("queued","running") else None)
def inbox():
    c=store.db()
    info=c.execute("SELECT * FROM runs WHERE id=?",(run,)).fetchone()
    rows=store.records(c,run)
    counts=Counter(x["state"] for x in rows)
    pending=counts["pending"]
    working=False
    with (store.DATA/"worker.lock").open("a") as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: working=True
    st.progress((len(rows)-pending)/max(1,len(rows)),text=f"{info['status'].capitalize()} · {len(rows)-pending} of {len(rows)} files")
    st.caption(f"{len(rows)-counts['skipped']} supported  ·  {counts['duplicate']} duplicates  ·  {counts['skipped']} skipped  ·  {counts['error']} errors  ·  {counts['cached']} reused")
    if working and info["status"] in ("queued","running"):
        if st.button("Cancel after current image"):
            with c: c.execute("UPDATE runs SET cancel=1 WHERE id=?",(run,))
    elif pending or counts["error"]:
        if st.button("Resume / retry failed"):
            with c:
                c.execute("UPDATE runs SET status='queued',cancel=0 WHERE id=?",(run,))
                c.execute("UPDATE items SET state='pending',note=NULL WHERE run=? AND state='error'",(run,))
            launch(run)
    if r["status"]!=info["status"] and info["status"] not in ("queued","running"): st.rerun()
    def group(x):
        p=x["effective"]
        if x["state"] in ("duplicate","error","pending","skipped"): return x["state"].title()
        if p.get("image_type")=="ordinary_photo" and not x["correction"]: return "Photos"
        if p.get("suggested_action"): return "Actions"
        return "References" if p.get("category")=="REFERENCE" else "Other"
    tally=Counter(group(x) for x in rows)
    reviews=sum(bool(x["effective"].get("needs_review")) for x in rows if x["state"]!="duplicate")
    for col,(label,num) in zip(st.columns(4),[("Actions",tally["Actions"]),("References",tally["References"]),("Needs review",reviews),("Photos",tally["Photos"])]): col.metric(label,num)
    a,b,d=st.columns(3)
    view=a.selectbox("Show",["All","Actions","References","Needs review","Photos","Error","Other","Duplicate","Skipped","Pending"])
    category=b.selectbox("Category",["All"]+CATEGORIES)
    order=d.selectbox("Sort by",["Upcoming dates","Category","Needs review","Filename"])
    def sortkey(x):
        p=x["effective"]
        if order=="Upcoming dates": return (x["date_status"] not in ("upcoming","today"),p.get("normalized_date") or "9999",p.get("category",""))
        if order=="Needs review": return not p.get("needs_review",False)
        return p.get("category","") if order=="Category" else x["source_path"]
    visible=[x for x in rows if (view in ("All",group(x)) or view=="Needs review" and x["effective"].get("needs_review") and group(x)!="Duplicate") and category in ("All",x["effective"].get("category"))]
    per_page=24
    pages=max(1,(len(visible)+per_page-1)//per_page)
    page=st.number_input("Page",1,pages,1) if pages>1 else 1
    for x in sorted(visible,key=sortkey)[(page-1)*per_page:page*per_page]:
        p=x["effective"]; key=x["image_id"]
        with st.container(border=True):
            left,right=st.columns([1,5])
            thumb=store.DATA/(str(key)+".jpg")
            if thumb.exists(): left.image(str(thumb),width=105)
            right.text(f"{group(x)}  ·  {x['source_path']}  ·  {x['state']}")
            right.text(p.get("title") or x["source_filename"])
            if p:
                right.caption(p["category"]+"  ·  "+p["image_type"]+("  ·  Needs review" if p["needs_review"] else ""))
                if p["suggested_action"]: right.text(p["suggested_action"])
                right.text(p["evidence"])
                if p["entities"]: right.text(" · ".join(p["entities"]))
                if p["original_date_text"]: right.text(f"{p['date_role'].title()} · {p['original_date_text']} · {x['date_status'] or 'verify date'}")
            elif x["note"]: right.text(x["note"])
            if p and group(x)!="Duplicate":
                with st.expander("Details and correction"):
                    st.json(x,expanded=False)
                    with st.form("edit"+key):
                        cat=st.selectbox("Utility category",CATEGORIES,index=CATEGORIES.index(p["category"]))
                        action=st.text_input("Suggested action",p["suggested_action"] or "")
                        states=["open","reviewed","completed","dismissed"]
                        status=st.selectbox("Status",states,index=states.index((x["correction"] or {}).get("status","open")))
                        if st.form_submit_button("Save correction"):
                            store.correct(c,key,cat,action if cat not in ("REFERENCE","OTHER") else None,status)
                            st.rerun()
    if not visible: st.caption("No items match these filters.")
    with st.expander("Export this analysis"):
        for col,fmt in zip(st.columns(3),["json","csv","html"]):
            col.download_button(fmt.upper(),lambda fmt=fmt: render(store.db(),run,fmt),f"analysis-{run}.{fmt}",mime={"json":"application/json","csv":"text/csv","html":"text/html"}[fmt])
inbox()
