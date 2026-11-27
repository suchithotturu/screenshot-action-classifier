"""Regenerate aggregate figures, report and offline presentation from measured results."""
import base64
import html
import json
import os
from pathlib import Path
from classifier.features import ROOT, atomic, OCR_BACKEND

def build(folder=None):
    folder = Path(folder or ROOT/"reports")
    support = "../" if OCR_BACKEND=="rapidocr" else ""
    ocr_title = "RapidOCR ONNXRuntime" if OCR_BACKEND=="rapidocr" else "Apple Vision OCR"
    ocr_description = ("RapidOCR ONNXRuntime 1.4.4 reads the complete oriented image with its checksum-verified PP-OCRv4 models on CPU."
                       if OCR_BACKEND=="rapidocr" else "Vision revision 3 reads English from the complete oriented image.")
    a = json.loads((folder/"results.json").read_text())
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".runtime"/"matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    content = a["tasks"]["content_type"]
    modes = [m for m in ("text","ocr","clip","fusion","fusion_char","fusion_layout","fusion_char_layout")
             if any(v["features"]==m for v in content["comparisons"])]
    fig,ax = plt.subplots(figsize=(max(10,len(modes)*2),4.8),layout="constrained")
    algorithms = [("#317877","Logistic regression"),("#31496f","Linear SVM"),
                  ("#a26732","k-NN"),("#895486","Complement Naive Bayes")]
    for i,(color,name) in enumerate(algorithms):
        values = [(m,next((v["validation"]["macro_f1"] for v in content["comparisons"]
                   if v["features"]==m and v["algorithm"]==name),None)) for m in modes]
        present = [(m,v) for m,v in values if v is not None]
        bars = ax.bar([modes.index(m)+(i-1.5)*.20 for m,v in present],
                      [v for m,v in present],width=.19,color=color,label=name)
        ax.bar_label(bars,fmt="%.3f",padding=3,fontsize=8)
    base = content["comparisons"][0]["validation"]["macro_f1"]
    ax.axhline(base,color="#936535",linestyle=":",label=f"Most frequent ({base:.3f})")
    names={"text":"OCR text","ocr":"OCR + numeric","clip":"CLIP","fusion":"Fusion",
           "fusion_char":"+ characters","fusion_layout":"+ layout","fusion_char_layout":"+ both"}
    ax.set(ylim=(0,1.08),ylabel="Validation macro F1",xticks=range(len(modes)),xticklabels=[names[m] for m in modes])
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.13),ncol=3,fontsize=9)
    fig.savefig(folder/"content-ablation.png",dpi=170)
    plt.close(fig)
    for task,t in a["tasks"].items():
        for part,m in t["tests"].items():
            fig,ax = plt.subplots(figsize=(6,4),layout="constrained")
            matrix = np.array(m["matrix"])
            ax.imshow(matrix,cmap="Blues")
            labels = [k.replace("_"," ") for k in m["classes"]]
            ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels,
                   xlabel="Predicted",ylabel="Source or approved label")
            ax.tick_params(axis="x",labelrotation=22)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j,i,str(matrix[i,j]),ha="center",va="center",color="white" if matrix[i,j]>matrix.max()/2 else "black")
            fig.savefig(folder/f"{task}-{part}.png",dpi=170)
            plt.close(fig)
    def table(headers,rows):
        return "\n".join(["| "+" | ".join(headers)+" |","|"+"---|"*len(headers)]+["| "+" | ".join(map(str,r))+" |" for r in rows])
    cols = ("train","validation","test_public","test_personal")
    counts = [[k,*[content["counts"][p].get(k,0) for p in cols]] for k in content["classes"]]
    label_count = len(content["classes"])
    comps = [[v["algorithm"],v["features"],v["columns"],f'{v["validation"]["accuracy"]:.3f}',f'{v["validation"]["macro_f1"]:.3f}'] for v in content["comparisons"]]
    scores,figures,details = [],[],[]
    machine_agreement = []
    for k,t in a["tasks"].items():
        if teacher := t.get("machine_labeled_test_agreement"):
            machine_agreement.append([k,teacher["n"],f'{teacher["accuracy"]:.3f}',f'{teacher["macro_f1"]:.3f}'])
        for p,v in t["tests"].items():
            scores.append([k,p,v["n"],f'{v["accuracy"]:.3f}',f'{v["macro_f1"]:.3f}',f'{v["weighted_f1"]:.3f}',f'{v["accepted"]}/{v["n"]}',f'{v["accepted_metrics"]["accuracy"]:.3f}' if v["accepted"] else "—"])
            figures.append(f"![{k} confusion matrix]({k}-{p}.png)")
            details.extend([[k,p,label,*[f'{v["per_class"][label][s]:.3f}' for s in ("precision","recall","f1-score")],int(v["per_class"][label]["support"])] for label in v["classes"]])
    main = content["tests"].get("test_public")
    sel = content["selected"]
    chosen = next(v for v in content["comparisons"] if v["features"]==sel["features"] and v["algorithm"]==sel["algorithm"])
    fusion = max(v["validation"]["macro_f1"] for v in content["comparisons"] if v["features"]=="fusion")
    extra_features = ("Additional conditions add train-fitted character 3–5-gram TF-IDF (up to 1,500 columns), "
                      "22 standardized OCR spatial measurements, or both. Layout uses only automatic text boxes: "
                      "a 3×3 position histogram, left/center/right proportions, box-size statistics, vertical position/span, "
                      "line gaps and OCR confidence. These experiments are documented separately in the feature-study report. "
                      "The deployed public-study winner contains no synthetic training images."
                      if any("_char" in v["features"] or "_layout" in v["features"] for v in content["comparisons"]) else "")
    text = f"""# Screenshot Intelligence · CS 513-B

Can a small supervised classifier recognize the content of a saved screenshot? Does joining OCR and visual features help? A second question concerns why the owner saved it. Only the owner can supply that training target.

## Findings

The validation winner was {sel["algorithm"]} with {sel["features"]} features (macro F1 {chosen["validation"]["macro_f1"]:.3f}). The best fused classifier scored {fusion:.3f} on the same validation images. The tables give pre-routing predictions and performance after applying the validation margin cutoff. Each task is evaluated independently on its eligible records. Deployment uses content confidence and optional OCR rules, then assigns remaining uncertainty to Miscellaneous.

## Data and independence

The study contains {a["total_eligible"]} eligible canonical images. Source counts are {a["sources"]}. The separate demonstration library contributed {a["fixtures_excluded"]} excluded fixtures. The public sample maps MASC Chat, Map and Setting to conversations, places and app screens. Enrico news supplies reading screens. Source labels describe screen function, which can differ from the dominant content a human would choose. The negative examples are photographs from TensorFlow Flowers. Individual photographer, license and source URL are retained in the private manifest.

The expanded experiment learns {label_count} content categories. Additional public MASC candidates were selected using UI hierarchy text, then annotated from their pixels by local Qwen. Hierarchy text and metadata never enter classifier features. CORD supplies real receipt images with source labels under CC BY 4.0. Ordinary flower photos map to Miscellaneous. Content targets now apply independently of capture status. Training label origins are {content.get("training_label_origins",{})}. Validation origins are {content.get("validation_label_origins",{})}. Qwen-labeled validation measures provisional agreement with machine annotations. Source-only test metrics and held-out machine-label agreement are distinguished. Owner purpose/usefulness remain unknown. The acquisition and annotation inventory is in [expansion.json]({support}expansion.json). Previous experiments are backed up in data/backups/.

{table(["Content","Train","Validation","Public test","Personal test"],counts)}

Split sizes are {a["split_counts"]}. The combined app, photographer and duplicate groups per partition are {a["group_counts"]}. Overlapping groups across partitions: {a["group_overlaps"]}. Exact byte matches retain alternate source IDs. Near matches share a group if their aspect ratios are within 3%, 64-bit difference hashes differ by at most four bits, and 8×8 RGB mean absolute difference is at most seven. This groups candidates conservatively. Unidentified crops may survive. Rico IDs are linked across sources. Seed 513 assigns up to seven stratified group folds, reserving one for validation and one for test. Final group assignments and frozen test labels are stored locally.

## Method

```mermaid
flowchart LR
  A[Image pixels] --> B[{ocr_title}]
  A --> C[Frozen CLIP encoder]
  B --> D[Train-fitted TF-IDF and numeric scaler]
  D --> E[Supervised classifier]
  C --> E
  H[Source or human labels] -->|training targets| E
  Q[Local Qwen annotation] --> P[Explicit machine pseudo-labels]
  P -->|provisional training targets| E
  E --> U[Categories and actions]
```

{ocr_description} Tall images use overlapping tiles. CLIP ViT-B/32 uses a fixed quantized ONNX checkpoint, aspect-preserving padding, and L2-normalized 512-dimensional image embeddings. The OCR condition includes 15 numeric features and up to 3,000 word unigrams/bigrams. The text-only condition contains nonnegative TF-IDF features and supports Complement Naive Bayes. The training set determines vocabulary, IDF and standardization. No owner, file name, app ID, Qwen description, suggested action, corrected transcript or human label enters the input matrix. Qwen assists annotation. The supervised classifiers are the project contribution.

{extra_features}

Logistic regression and linear SVM use class weighting and default regularization C=1. k-NN uses five neighbors, cosine distance, and distance-weighted voting. Complement Naive Bayes uses alpha=1 and OCR text only. These are fixed initial settings. Selection maximizes validation macro F1, with smaller dimensionality then fixed candidate order breaking ties. The routing cutoff is the tenth percentile of margins among correct validation predictions. For SVM and logistic regression the margin is computed from decision scores. For k-NN and Naive Bayes it is the difference between the two highest class outputs, whose calibration has not been verified. Empty OCR is a valid observed feature. Extraction errors abstain and can be retried.

## Algorithm and modality comparison

{table(["Algorithm","Features","Columns","Validation accuracy","Validation macro F1"],comps)}

![Content feature comparison](content-ablation.png)

## Current deployment behavior

The eight-category content policy runs the trained content classifier for every decodable image. Confident predictions from the complete supervised head take priority. Screenshot detection is recorded as advisory metadata and cannot block a content prediction. Strong OCR cues can identify message composers, product purchase offers, confirmed payments, issued tickets, restaurant menus, maps and game entertainment. Uncertain content goes to Miscellaneous. Decode or feature-extraction failures retain a separate Processing error state.

Local OCR rules assist uncertain model decisions. Shopping, confirmations, entertainment and Miscellaneous are also learned labels when present in the training data. They remain explicit in each saved prediction through `category_source` and `content_rule`. They use automatic pixel-derived OCR and generic patterns. The fresh images remain outside training. Development examples were inspected to repair this policy, so the aggregate before/after counts in [routing-check.json]({support}routing-check.json) demonstrate routing coverage. They do not measure independent classification accuracy. The test results below evaluate the saved supervised heads and retain their original meaning.

## Frozen test results

{table(["Task","Partition","N","Accuracy","Macro F1","Weighted F1","Accepted","Accepted accuracy"],scores)}

{table(["Task","Partition","Class","Precision","Recall","F1","Support"],details)}

{chr(10).join(figures)}

Median feature latency on this Mac was {a["latency_ms"]["ocr_seconds"]:.1f} ms for OCR and {a["latency_ms"]["clip_seconds"]:.1f} ms for CLIP. These include per-image preparation inside each extractor and concurrent dataset processing. The cached features retain individual times. Model-only batch timings, coverage, accepted-only F1 and detailed validation matrices for both tasks are in [results.json](results.json).

## Held-out machine-label agreement

These rows use machine-generated targets and provide a provisional teacher-agreement check. They are excluded from the source-only test table.

{table(["Task","Machine-labeled rows","Agreement accuracy","Agreement macro F1"],machine_agreement)}

## Interpretation and limits

The supervised head contains {label_count} trained categories. Some newly added categories rely on Qwen pseudo-labels, so teacher errors can propagate into training. CORD receipts are photographs with a different visual style from phone payment confirmations. Miscellaneous includes ordinary photographs and remains a broad heterogeneous category. A closed-set model can confidently assign an unsupported image to a known class. Class margins do not solve open-set recognition. Reading examples come from one source and their test support is small. The detector can exploit image dimensions, source style and flower imagery. Its public score does not establish accuracy on a mixed personal phone library. The blank-crop and portrait demonstration includes uncertain routes.

There are no independently blind human labels covering all categories. A source-only test score measures agreement with dataset mappings for its supported classes. Provisional validation and a separately reported held-out machine-label comparison use Qwen pseudo-labels. Their scores measure teacher agreement. Macro F1 over all trained labels penalizes unsupported test classes with zero scores, while supported_macro_f1 in results.json restricts averaging to classes with test support. Earlier public data have been inspected, and this expanded study uses newly grouped splits. It is exploratory. Fresh independent labels across all eight categories are needed for a confirmatory accuracy claim. No owner-confirmed purpose or usefulness labels were supplied. The course team-size requirement is 3–4.

## Sample records and references

A public record may contain `source.dataset=masc` within an origin, `labels.content_type=chat`, `target_provenance.content_type.origin=source` and `labels.purpose=null`. The validated sample file contains actual records reduced to public identifiers, targets and splits. Private images, OCR and drafts stay in the local export.

- Girgis et al. (2025), MASC dataset and mapping. [Authors’ repository](https://github.com/Ali-Aahmed/MASC-Dataset)
- Leiva, Hota and Oulasvirta (2020), Enrico, derived from Rico. [Authors’ dataset](https://userinterfaces.aalto.fi/enrico/)
- Park et al., CORD receipt data. [Authors’ release](https://github.com/clovaai/cord) and [training data](https://huggingface.co/datasets/naver-clova-ix/cord-v2)
- TensorFlow Team, Flowers. [Dataset and attribution](https://www.tensorflow.org/datasets/catalog/tf_flowers)
- Radford et al. (2021), CLIP. [Original paper](https://proceedings.mlr.press/v139/radford21a.html) and [quantized checkpoint](https://huggingface.co/Xenova/clip-vit-base-patch32/blob/main/onnx/vision_model_quantized.onnx)
- Apple, Vision text recognition. [Documentation](https://developer.apple.com/documentation/vision/recognizing-text-in-images)
- scikit-learn, feature extraction and linear classifiers. [Documentation](https://scikit-learn.org/stable/modules/feature_extraction.html)
- scikit-learn, nearest-neighbor and Naive Bayes classification. [k-NN](https://scikit-learn.org/stable/modules/neighbors.html) and [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- Ollama, Qwen3-VL 4B Instruct and structured outputs. [Model](https://ollama.com/library/qwen3-vl:4b-instruct) and [API](https://docs.ollama.com/capabilities/structured-outputs)
"""
    atomic(folder/"REPORT.md",text.encode())
    slides = [
        ("Screenshot Intelligence","Can a local classifier organize saved screenshots?","A CS 513-B study of lightweight supervised models and human-assisted labels."),
        ("The research sample",f'{a["total_eligible"]} images across public sources.',f'{sum(sum(v.values()) for v in content["counts"].values())} content-labeled screenshots. The negatives are flower photographs. Personal intent needs owner review.'),
        ("The supervised experiment","Logistic regression, linear SVM, k-NN and Complement Naive Bayes.",f"Source labels and explicitly marked training pseudo-labels supply targets. {ocr_title} and frozen CLIP supply features. Fresh-folder classification uses the saved classifiers."),
        ("Testing feature fusion","Which inputs improve validation macro F1?",'<img src="content-ablation.png" alt="Validation macro F1 by algorithm and feature set">'),
        ("Contents of held-out screens",f'{main["accuracy"]:.1%} accuracy across {main["n"]} public screenshots.' if main else "Public test unavailable.",'<img src="content_type-test_public.png" alt="Content confusion matrix">' if main else ""),
        ("What the scores cover",f"{label_count} content categories have trained class boundaries.","Some targets are Qwen pseudo-labels. Source-only tests cover their available classes. Agreement with machine labels is reported separately. Source styles limit generalization."),
        ("Live demonstration","Content categories apply to screenshots and other images.","The saved classifiers and local OCR rules assign categories. Ambiguous content goes to Miscellaneous. Rule-based decisions are recorded separately. Qwen is disabled for classification."),
        ("The next experiment","Collect blind owner-confirmed labels.","Add purchases, receipts, social screenshots and diverse photos. Evaluate plausible purpose on a separate owner holdout. Dataset and software references are in REPORT.md.")
    ]
    stylesheet = """*{box-sizing:border-box}body{margin:0;font:clamp(18px,2.4vw,34px)/1.35 -apple-system,BlinkMacSystemFont,Arial;color:#183438;background:#faf9f4}main{max-width:1100px;margin:auto;padding:7vh 6vw 70px}h1{font-size:clamp(36px,5vw,75px);line-height:1.04;margin:.3em 0 .5em}p{max-width:900px}img{max-width:100%;max-height:46vh;object-fit:contain}nav{position:fixed;bottom:14px;right:4vw}button{font:16px system-ui;padding:9px 17px;margin:3px;border:1px solid #577c77;background:white;border-radius:6px}section{display:none}section.active{display:block}@media print{section,section.active{display:block;break-after:page}nav{display:none}}"""
    content_html = "".join(f'<section><h1>{html.escape(title)}</h1><p>{html.escape(sub)}</p>'+ (detail if detail.startswith("<img") else "<p>"+html.escape(detail)+"</p>")+"</section>" for title,sub,detail in slides)
    for asset in ("content-ablation.png","content_type-test_public.png"):
        if (folder/asset).exists():
            uri = "data:image/png;base64,"+base64.b64encode((folder/asset).read_bytes()).decode()
            content_html = content_html.replace('src="'+asset+'"','src="'+uri+'"')
    script = """const s=[...document.querySelectorAll('section')];let i=0;function show(n){i=Math.min(s.length-1,Math.max(0,n));s.forEach((e,j)=>e.classList.toggle('active',i===j));page.textContent=(i+1)+' / '+s.length;prev.disabled=i===0;next.disabled=i===s.length-1}prev.onclick=()=>show(i-1);next.onclick=()=>show(i+1);onkeydown=e=>{if(e.key==='ArrowRight')show(i+1);if(e.key==='ArrowLeft')show(i-1)};show(0)"""
    atomic(folder/"presentation.html",f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Screenshot Intelligence</title><style>{stylesheet}</style><main>{content_html}</main><nav><span id="page"></span> <button id="prev">Previous</button><button id="next">Next</button></nav><script>{script}</script></html>'.encode())

if __name__=="__main__":
    build()
