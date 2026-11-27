"""Streamlit views over the same operations as the CLI."""
import json
import hashlib
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import get_args
import pandas as pd
import streamlit as st
from . import dataset, production, store
from .features import ROOT, cache_key, extract
from .schema import ActionType, Draft, NAMES, Purpose

def alert(message):
    st.session_state["studio_notice"] = message
def notice():
    if msg := st.session_state.pop("studio_notice",None):
        st.info(msg)

def progress(bar):
    return lambda n,total: bar.progress(n/max(total,1),text=f"{n} of {total} files")

def mix_review_queue(docs, blind):
    ordered = sorted(docs,key=lambda d:hashlib.sha256(d["id"].encode()).hexdigest())
    if blind:
        return ordered
    categories = defaultdict(deque)
    for doc in ordered:
        label = doc["labels"]["content_type"] or ("Photos" if doc["labels"]["is_screenshot"] is False else "Unknown")
        categories[label].append(doc)
    mixed = []
    while categories:
        for label in list(categories):
            mixed.append(categories[label].popleft())
            if not categories[label]:
                del categories[label]
    return mixed

def classify():
    c = dataset.connect()
    st.title("Your images")
    notice()
    runs = c.execute("SELECT id,created,json_extract(body,'$.directory') AS directory FROM prediction_runs ORDER BY id DESC").fetchall()
    default = ROOT/"demo-unseen"
    if runs:
        default = Path(json.loads(c.execute("SELECT body FROM prediction_runs WHERE id=?",(runs[0]["id"],)).fetchone()[0])["directory"])
    with st.expander("Choose an image folder",expanded=not runs):
        folder = st.text_input("Image folder",str(default if default.exists() else ROOT/"demo"))
        from .features import OCR_BACKEND
        from .learning import load_model
        model_options={"Prepared classifier":None}
        for name,key in [("Public-data experiment","public"),("Synthetic-data experiment","synthetic"),("Mixed-data experiment","mixed")]:
            candidate=ROOT/"models/experiments/feature-study"/OCR_BACKEND/key/"classifier.joblib"
            if candidate.exists():model_options[name]=candidate
        for name,key in [("HOG visual experiment","fusion_char_hog"),("All visual features experiment","fusion_char_visual"),
                         ("Engineered images only experiment","pixels_all")]:
            candidate=ROOT/"models/experiments/visual-study"/OCR_BACKEND/key/"classifier.joblib"
            if candidate.exists():model_options[name]=candidate
        chosen_model=st.selectbox("Model",list(model_options)) if len(model_options)>1 else "Prepared classifier"
        if chosen_model in ("HOG visual experiment","All visual features experiment","Engineered images only experiment"):
            st.caption("Experimental model; the prepared classifier scored higher on validation. OCR rules can still assist uncertain routing.")
        if st.button("Classify folder",type="primary"):
            try:
                run,_ = production.predict_folder(c,folder,progress=progress(st.progress(0)),bundle=load_model(model_options[chosen_model]))
                st.session_state["production_run"] = run
                st.rerun()
            except Exception as e:
                st.error(str(e))
    if not runs:
        st.info("Choose a folder to start a classification run.")
        return
    run_names = {r["id"]:f"{Path(r['directory']).name} · {r['created'].replace('T',' ')[:19]} · {r['id']}" for r in runs}
    with st.sidebar.expander("Previous results"):
        run = st.selectbox("Saved classifications",list(run_names),key="production_run",format_func=run_names.get)
    body = json.loads(c.execute("SELECT body FROM prediction_runs WHERE id=?",(run,)).fetchone()[0])
    tally = Counter(r["route"] for r in body["items"])
    st.caption(f"{Path(body['directory']).name} · {len(body['items'])} images · {tally['other']} miscellaneous")
    filter_col,export_col = st.columns([3,1],vertical_alignment="bottom")
    route = filter_col.selectbox("Filter images",["All",*sorted(tally)],format_func=lambda k:"All images" if k=="All" else f"{NAMES.get(k,k)} ({tally[k]})")
    export_col.download_button("Download results",json.dumps({k:v for k,v in body.items() if k!="directory"},indent=2),
                               f"classification-{run}.json","application/json",use_container_width=True)
    visible = [(i,r) for i,r in enumerate(body["items"]) if route in ("All",r["route"])]
    pages = max(1,(len(visible)+19)//20)
    page = st.number_input("Results page",1,pages,key=f"resultpage{run}{route}") if pages>1 else 1
    review_messages = {
        "Detector margin below validation threshold.":"Check whether this image is a screenshot or a photo.",
        "Content margin below validation threshold.":"Check the suggested category against the image.",
        "Detector unavailable. Confirm screenshots-only input to classify content.":"Choose a screenshots-only folder to classify these images.",
    }
    for index,r in visible[(page-1)*20:page*20]:
        with st.container(border=True):
            st.subheader(NAMES.get(r["route"],r["route"]))
            image,text = st.columns([2,3],gap="large")
            path = Path(body["directory"])/r["path"]
            if path.is_file() and r.get("sha256") and not path.is_symlink():
                try:
                    image.image(str(path),width=320)
                except Exception:
                    image.caption("Preview unavailable")
            text.caption(r["path"])
            if r.get("category_source")=="ocr_rule" and r["route"]!="other":
                text.caption("Category matched from visible text")
            if r["route"]=="Needs review":
                guess = r.get("tasks",{}).get("content_type",{}).get("label")
                if guess:
                    text.write("Suggested category · "+NAMES.get(guess,guess))
                messages = [review_messages[reason] for reason in r.get("reasons",[]) if reason in review_messages]
                for message in messages:
                    text.write(message)
            if r.get("status") in ("failed","error"):
                text.write("This image could not be processed.")
            if r["route"]=="Skipped":
                for reason in r.get("reasons",[]):
                    text.write(reason)
            for j,action in enumerate(r["actions"]):
                text.write(action["description"])
                if action.get("evidence"):
                    text.caption(action["evidence"])
                key = f"action-{run}-{index}-{j}"
                with text.expander("Edit suggestion"):
                    with st.form(key+"form"):
                        value = st.text_input("Suggested action",action["description"],key=key)
                        if st.form_submit_button("Save"):
                            changed = [dict(a) for a in r["actions"]]
                            changed[j]["description"] = value
                            production.save_actions(c,run,index,changed)
                            st.rerun()
            if r.get("ocr") and r["route"]!="Photos":
                with text.expander("Text in image"):
                    st.text(r["ocr"])
            if r.get("status") in ("failed","error"):
                with text.expander("Processing details"):
                    st.text("\n".join(r.get("reasons",[])))

def annotation():
    from .annotation import propose
    c = dataset.connect()
    st.title("Dataset Annotation")
    notice()
    with st.expander("Import an image folder"):
        folder = st.text_input("Local folder path",str(ROOT/"demo"))
        source = st.selectbox("Image source",["personal","public","fixture","memes","masc","screen2words"])
        owner = st.selectbox("Image owner",[None,"A","B","C","D"],disabled=source!="personal",format_func=lambda x:x or "Unknown")
        session = st.text_input("Collection session (optional)",help="Related captures from one session stay in the same split.")
        if st.button("Import folder"):
            try:
                result = dataset.ingest_folder(c,folder,source,owner if source=="personal" else None,session,progress(st.progress(0)))
                st.session_state["source_filter"] = "fixture" if Path(folder).expanduser().resolve()==ROOT/"demo" else source
                alert("Import complete · "+", ".join(f"{k} {v}" for k,v in result.items()))
                st.rerun()
            except Exception as e:
                st.error(str(e))
    docs = dataset.all_docs(c)
    states = Counter(d["annotations"]["review"]["state"] for d in docs)
    a,b,h = st.columns(3)
    a.metric("Images",len(docs))
    b.metric("Reviewed",states["reviewed"])
    h.metric("Auto-labeled",states["auto_labeled"])
    if states["auto_labeled"]:
        st.info("The working dataset is automatically labeled and ready for training. Manual review is optional. Saved model suggestions are marked provisional.")
    with st.expander("Dataset counts and export"):
        sources = Counter(s["dataset"] for d in docs for s in d["source"]["origins"][:1])
        st.write("Sources",dict(sources))
        st.write("Accepted content",dict(Counter(d["labels"]["content_type"] or "Unknown" for d in docs)))
        st.write("Review states",dict(states))
        if st.button("Export validated JSONL and CSV",disabled=st.session_state.get("blind_review",False)):
            try:
                st.success(str(dataset.export(c)))
            except Exception as e:
                st.error(str(e))
    with st.expander("Local Qwen service"):
        st.caption("Start the bundled service for assisted labeling. Manual and blind labeling work without it.")
        if st.button("Start local Qwen"):
            import subprocess
            import sys
            args = (["powershell.exe","-NoProfile","-ExecutionPolicy","Bypass","-File","service.ps1"]
                    if sys.platform=="win32" else ["bash","service.sh"])
            cmd = subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=30)
            if cmd.returncode:
                st.error(cmd.stderr.strip() or cmd.stdout.strip())
            else:
                st.success("Local Qwen service ready.")
    st.subheader("Review queue")
    st.caption("Check each image and choose its content category. Prefilled labels come from the dataset or a saved suggestion. Approve only after checking them.")
    role_col,who_col = st.columns(2)
    role = role_col.selectbox("Reviewer role",["reviewer","owner"])
    reviewer = who_col.selectbox("Reviewer identity",["Reviewer 1","Reviewer 2","A","B","C","D"] if role=="reviewer" else ["A","B","C","D"])
    blind = st.checkbox("Blind labeling",key="blind_review",help="Only records that have never displayed assisted labels are available.")
    a,b,h = st.columns(3)
    sources = sorted({s["dataset"] for d in docs for s in d["source"]["origins"]})
    source_filter = a.selectbox("Filter source",["All",*sources],key="source_filter",index=(1+sources.index("masc") if "masc" in sources else 0))
    state = b.selectbox("Review state",["Pending & drafts","Auto-labeled","Reviewed","Skipped","All"],
                        index=1 if states["auto_labeled"] else 0)
    batch_size = h.number_input("Batch size",1,30,10)
    partition = st.selectbox("Study partition",["Any","Unassigned","train","validation","test_public","test_personal"])
    category_filter = "All"
    if not blind:
        category_filter = st.selectbox("Content filter",["All","Unknown",*NAMES],
                                       format_func=lambda k:NAMES.get(k,k))
    assigned = dict(c.execute("SELECT id,partition FROM splits"))
    def include(d):
        if partition not in ("Any",assigned.get(d["id"],"Unassigned")):
            return False
        r = d["annotations"]["review"]
        if source_filter!="All" and not any(s["dataset"]==source_filter for s in d["source"]["origins"]):
            return False
        if category_filter!="All" and (d["labels"]["content_type"] or "Unknown")!=category_filter:
            return False
        if blind and r["exposed"] or not blind and r["mode"]=="blind" and r["state"]!="reviewed":
            return False
        return (state=="All" or state=="Pending & drafts" and r["state"] in ("pending","draft")
                or state.lower().replace("-","_")==r["state"])
    queue = mix_review_queue([d for d in docs if include(d)],blind)
    st.caption(f"{len(queue)} matching images")
    if not queue:
        return
    page = st.number_input("Batch page",1,max(1,(len(queue)+batch_size-1)//batch_size),key=f"page{source_filter}{state}{blind}{batch_size}{partition}{category_filter}")
    visible = queue[(page-1)*batch_size:page*batch_size]
    if not st.toggle("Open review batch",key=f"open-{blind}-{source_filter}"):
        return
    if not blind:
        batch_counts = Counter(d["labels"]["content_type"] or ("Photos" if d["labels"]["is_screenshot"] is False else "Unknown") for d in visible)
        st.caption("Starting labels in this batch · "+" · ".join(f"{NAMES.get(k,k)} ({v})" for k,v in batch_counts.items()))
    if not blind and st.button("Generate Qwen suggestions for this batch"):
        bar = st.progress(0)
        failures = 0
        for i,d in enumerate(visible,1):
            result = propose(c,d["id"])
            failures += result["status"]!="ok"
            bar.progress(i/len(visible))
        alert(f"Qwen processed {len(visible)} records. {failures} require manual labeling.")
        st.rerun()
    if st.button("Extract automatic OCR for this batch"):
        bar = st.progress(0)
        for i,d in enumerate(visible,1):
            extract(dataset.image_path(d).read_bytes(),store.DATA/"features")
            bar.progress(i/len(visible))
        st.rerun()
    entries = []
    for d in visible:
        sid = d["id"]
        r = d["annotations"]["review"]
        if blind and r["mode"] is None:
            dataset.reserve_blind(c,sid)
        if not blind and not r["exposed"]:
            dataset.expose(c,sid)
        entry = card(c,d,reviewer,role,blind)
        if entry:
            entries.append(entry)
    if entries and st.button(f"Approve {len(entries)} displayed records",type="primary"):
        try:
            dataset.batch_edit(c,entries)
            alert(f"Approved {len(entries)} records.")
            st.rerun()
        except ValueError as e:
            st.error(str(e))

def card(c,d,reviewer,role,blind):
    sid = d["id"]
    a = d["annotations"]
    r = a["review"]
    rev = r["revision"]
    locked = r["state"]=="reviewed"
    initial = dataset.current_draft(d,blind).model_dump()
    prefix = f"{sid}-{rev}-{blind}-{role}-{reviewer}"
    owner = dataset.is_owner(d,reviewer,role)
    if not owner:
        for k in ("purpose","still_useful"):
            initial[k] = d["labels"][k] if d["labels"]["target_provenance"][k]["origin"]=="owner" else None
    def key(k):
        return prefix+"-"+k
    def pick(label,k,options,fmt=None,disabled=False):
        # A real option distinguishes Unknown from the widget's own unselected state.
        choices = ["__unknown__" if v is None else v for v in options]
        def display(v):
            if v=="__unknown__":
                return "Unknown"
            return fmt(v) if fmt else str(v).replace("_"," ")
        selected = st.selectbox(label,choices,index=options.index(initial[k]),key=key(k),
                                disabled=disabled or locked,format_func=display)
        return None if selected=="__unknown__" else selected
    def action(event,fields):
        try:
            dataset.batch_edit(c,[(sid,fields,event,reviewer,role,blind,rev)])
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    with st.container(border=True):
        left,right = st.columns([2,3])
        with left:
            st.image(str(dataset.image_path(d)),width=300)
            st.caption(f"{r['state'].title()} · revision {rev}")
            if not blind:
                st.caption(sid)
        with right:
            if not blind:
                origins = {s.get("content_type") for s in d["source"]["origins"] if s.get("content_type")}
                if origins:
                    st.caption("Dataset label · "+", ".join(NAMES[k] for k in sorted(origins)))
                qwen_label = ((a["qwen"] or {}).get("fields") or {}).get("content_type")
                if qwen_label:
                    st.caption("Saved Qwen suggestion · "+NAMES[qwen_label])
            if not blind and a["qwen"]:
                with st.expander("Original Qwen proposal"):
                    st.json(a["qwen"])
            if locked:
                st.write("Approved labels",d["labels"])
                if st.button("Undo approval",key=key("undo")):
                    action("undo",initial)
                return None
            vals = {}
            vals["is_screenshot"] = pick("Screenshot capture","is_screenshot",[None,True,False])
            vals["content_type"] = pick("Dominant content","content_type",[None,*NAMES],lambda x:NAMES.get(x,"Unknown"))
            vals["purpose"] = pick("Owner-confirmed purpose","purpose",[None,*get_args(Purpose)],disabled=not owner or vals["is_screenshot"] is not True)
            vals["still_useful"] = pick("Still useful to the owner","still_useful",[None,True,False],disabled=not owner or vals["is_screenshot"] is not True)
            cache = store.DATA/"features"/(cache_key(d["image"]["sha256"])+".json")
            with st.expander("Automatic OCR and optional annotations"):
                if cache.is_file():
                    o = json.loads(cache.read_bytes())["ocr"]
                    st.text(o.get("text") or o.get("error") or "OCR succeeded with no readable text.")
                else:
                    st.caption("OCR pending. Use the batch extraction button.")
                for k,label in [("description","Description"),("evidence","Visible evidence"),("corrected_transcript","Corrected transcript")]:
                    vals[k] = st.text_area(label,initial[k],key=key(k))
                vals["tags"] = [v.strip() for v in st.text_input("Tags (comma separated)",", ".join(initial["tags"]),key=key("tags")).split(",") if v.strip()]
                frame = pd.DataFrame(initial["actions"],columns=["type","description","evidence","date_time"])
                actions = st.data_editor(frame,key=key("actions"),num_rows="dynamic",hide_index=True,
                    column_config={"type":st.column_config.SelectboxColumn("Action",options=list(get_args(ActionType)),required=True),
                                   "date_time":st.column_config.TextColumn("Visible date / time")})
                vals["actions"] = actions.astype(object).where(actions.notna(),None).to_dict("records")
                for item in vals["actions"]:
                    for k in ("description","evidence"):
                        item[k] = item.get(k) or ""
            vals["ambiguous"] = st.checkbox("Ambiguous",initial["ambiguous"],key=key("ambiguity"))
            if not blind and a["qwen"] and a["qwen"].get("fields"):
                changed = [k for k,v in a["qwen"]["fields"].items() if vals.get(k)!=v]
                st.caption("Changed from Qwen · "+(", ".join(changed) or "none"))
            if vals!=initial:
                action("save",vals)
            one,two,three = st.columns(3)
            if one.button("Approve",key=key("approve")):
                action("approve",vals)
            if two.button("Skip",key=key("skip")):
                action("skip",vals)
            if not blind and three.button("Reset draft to Qwen",key=key("reset"),disabled=not a["qwen"] or a["qwen"]["status"]!="ok"):
                action("reset",vals)
            return sid,vals,"approve",reviewer,role,blind,rev
