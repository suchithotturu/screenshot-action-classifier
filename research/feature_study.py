"""Controlled character/layout ablations on public, synthetic and mixed training.

All inputs use automatic OCR/CLIP. Synthetic generator metadata supplies y and
partitions only. No Qwen/network access; test metrics do not select models.
"""
import argparse
import copy
import hashlib
import io
import json
import sys
import time
import warnings
import platform
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from classifier import dataset, model, splits, store
from classifier.features import ROOT, VERSION, OCR_BACKEND, atomic, encoder, extract
from classifier.learning import Fusion, SEED, load_model, margins, metrics
from classifier.schema import NAMES

CONDITIONS={'Baseline':'fusion','Characters':'fusion_char','Layout':'fusion_layout','Characters + layout':'fusion_char_layout'}


def score(head,rows):
 if not rows:return None
 x=head['transform'].transform([r['f'] for r in rows]);y=[r['label'] for r in rows]
 start=time.perf_counter();p=head['model'].predict(x);elapsed=time.perf_counter()-start
 accepted=margins(head['model'],x)>=head['threshold']
 result=metrics(y,p,list(head['model'].classes_))
 result.update(accepted=int(accepted.sum()),coverage=float(accepted.mean()),inference_ms=1000*elapsed/len(rows),
               accepted_metrics=metrics(np.asarray(y)[accepted],p[accepted],list(head['model'].classes_)) if accepted.any() else None)
 return result


def public_rows(c):
 if not c.execute("SELECT 1 FROM sqlite_master WHERE name='locked_labels'").fetchone():
  raise ValueError('Rebuild the prepared public study, or use --synthetic-only.')
 result=[]
 for body,partition,group,locked in c.execute('SELECT samples.doc,splits.partition,splits.group_id,locked_labels.labels FROM samples JOIN splits USING(id) LEFT JOIN locked_labels USING(id) ORDER BY samples.rowid'):
  d=json.loads(body)
  if locked:d['labels']=json.loads(locked)
  if any(s['dataset']=='personal' for s in d['source']['origins']):continue
  label=splits.eligible(d,'content_type',allow_pseudo=True)
  if label is None:continue
  result.append(dict(id=d['id'],path=dataset.image_path(d),label=label,
                     split='test' if partition.startswith('test') else partition,
                     origin=d['labels']['target_provenance']['content_type']['origin'],group=group))
 return result


def prepare(rows):
 encoder()
 def one(row):
  f=extract(row['path'].read_bytes(),store.DATA/'features')
  if f['status']!='ok':raise RuntimeError('Failed feature extraction: '+row['id'])
  return dict(row,f=f)
 done=[]
 with ThreadPoolExecutor(3) as pool:
  for n,row in enumerate(pool.map(one,rows),1):
   done.append(row)
   if n%200==0:print(f'{OCR_BACKEND} features {n}/{len(rows)}',flush=True)
 return done


def run(public_only=False,synthetic_only=False):
 def offline(event,args):
  if event in ('socket.connect','socket.getaddrinfo'):raise AssertionError('Network disabled during feature study')
 def no_qwen(*args,**kwargs):raise AssertionError('Qwen disabled during feature study')
 model.api=no_qwen;sys.addaudithook(offline)
 out=ROOT/'reports/feature-study'/OCR_BACKEND
 if synthetic_only:out=out/'synthetic-only'
 out.mkdir(parents=True,exist_ok=True)
 c=dataset.connect();public=[] if synthetic_only else public_rows(c)
 if not synthetic_only:
  assert len([r for r in public if r['split']=='train'])==911,'Public partitions changed; use --synthetic-only or rebuild the public study.'
 synthetic=[]
 manifest=ROOT/'data/synthetic-screens/manifest.jsonl'
 if not public_only:
  synthetic=[json.loads(s) for s in manifest.read_text().splitlines()]
  for row in synthetic:row['path']=manifest.parent/row['path']
  groups={p:{r['group'] for r in synthetic if r['split']==p} for p in ('train','validation','test')}
  assert not any(groups[a]&groups[b] for a in groups for b in groups if a<b)
  hashes={p:{r['sha256'] for r in synthetic if r['split']==p} for p in groups}
  assert not any(hashes[a]&hashes[b] for a in hashes for b in hashes if a<b)
 all_rows=prepare(public+synthetic);public=all_rows[:len(public)];synthetic=all_rows[len(public):]
 tests={'public_source':[r for r in public if r['split']=='test' and r['origin']!='pseudo'],
        'public_teacher':[r for r in public if r['split']=='test' and r['origin']=='pseudo'],
        'synthetic':[r for r in synthetic if r['split']=='test']}
 corpora=['Synthetic'] if synthetic_only else ['Public'] if public_only else ['Public','Synthetic','Mixed']
 report=dict(feature_version=VERSION,backend=OCR_BACKEND,tested_host=f'{platform.system()} {platform.machine()}',seed=SEED,
             solver_caps=dict(logistic_regression=1500,linear_svm=30000),character_settings=dict(analyzer='char_wb',ngram_range=[3,5],max_features=1500,min_df=2),
             layout_dimensions=22,corpora={},qwen_calls=0,python_network_connections=0,
             synthetic_inventory=json.loads((manifest.parent/'inventory.json').read_text()) if synthetic else None,
             limitations=['Synthetic labels describe generator intent, not independent human review.','Held-out synthetic configurations share drawing code and phrase banks.','Public validation includes Qwen pseudo-labels; source-only and teacher test rows are separate.','Existing public tests have been inspected previously; all results are exploratory.'])
 base=load_model()
 for corpus in corpora:
  train=[r for r in (public if corpus=='Public' else synthetic if corpus=='Synthetic' else public+synthetic) if r['split']=='train']
  valid=[r for r in (synthetic if corpus=='Synthetic' else public) if r['split']=='validation']
  yt=[r['label'] for r in train];yv=[r['label'] for r in valid]
  result=dict(train_n=len(train),validation_n=len(valid),validation_source='synthetic' if corpus=='Synthetic' else 'public',
              training_origins=dict(Counter(r['origin'] for r in train)),comparisons=[])
  best=None
  for title,mode in CONDITIONS.items():
   transform=Fusion(mode).fit([r['f'] for r in train])
   xt=transform.transform([r['f'] for r in train]);xv=transform.transform([r['f'] for r in valid])
   for algorithm,estimator in [('Logistic regression',LogisticRegression(max_iter=1500,class_weight='balanced',random_state=SEED)),
                               ('Linear SVM',LinearSVC(class_weight='balanced',dual='auto',random_state=SEED,max_iter=30000))]:
    start=time.perf_counter()
    with warnings.catch_warnings():
     warnings.simplefilter('error',ConvergenceWarning)
     estimator.fit(xt,yt)
    fit_seconds=time.perf_counter()-start
    pred=estimator.predict(xv);val=metrics(yv,pred,sorted(NAMES))
    correct=margins(estimator,xv)[np.asarray(yv)==pred]
    threshold=float(np.quantile(correct,.1)) if len(correct) else float(np.finfo(float).max)
    head=dict(algorithm=algorithm,features=mode,transform=transform,model=estimator,threshold=threshold)
    comparison=dict(condition=title,features=mode,algorithm=algorithm,columns=xt.shape[1],fit_seconds=fit_seconds,
                    solver_iterations=int(np.max(estimator.n_iter_)),solver_converged=True,
                    validation=val,threshold=threshold,tests={k:score(head,v) for k,v in tests.items() if v})
    result['comparisons'].append(comparison)
    key=(val['macro_f1'],-xt.shape[1])
    if best is None or key>best[0]:best=(key,head,comparison)
    print(f"{corpus} / {title} / {algorithm}: validation F1={val['macro_f1']:.4f}",flush=True)
  _,head,chosen=best
  ident=hashlib.sha256(json.dumps(dict(version='feature-study-v1',features=VERSION,corpus=corpus,
      rows=[(r['id'],r['label']) for r in train],selected=(head['features'],head['algorithm'])),sort_keys=True).encode()).hexdigest()[:16]
  artifact=copy.copy(base);artifact['tasks']=dict(base['tasks']);artifact['tasks']['content_type']=head
  artifact.update(training_id=ident,study=dict(corpus=corpus,synthetic_train=corpus in ('Synthetic','Mixed'),seed=SEED))
  dest=ROOT/'models/experiments/feature-study'/OCR_BACKEND/corpus.lower()/'classifier.joblib'
  buf=io.BytesIO();joblib.dump(artifact,buf,compress=3);atomic(dest,buf.getvalue())
  again=load_model(dest)
  assert np.array_equal(again['tasks']['content_type']['model'].predict(head['transform'].transform([r['f'] for r in valid])),head['model'].predict(head['transform'].transform([r['f'] for r in valid])))
  result.update(selected={k:chosen[k] for k in ('condition','features','algorithm','columns','validation','threshold','tests')},
                training_id=ident,model=str(dest.relative_to(ROOT)),identical_after_reload=True)
  report['corpora'][corpus]=result
  atomic(out/'results.json',json.dumps(report,indent=2,allow_nan=False).encode())
 write_report(report,out)
 print('Report: '+str(out/'REPORT.md'),flush=True)
 return report


def write_report(r,out):
 import os
 preview_relative=os.path.relpath(ROOT/'reports/feature-study/synthetic-preview.jpg',out).replace(os.sep,'/')
 os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.runtime/matplotlib'))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rows=[]
 for corpus,data in r['corpora'].items():
  for x in data['comparisons']:
   tests=x['tests'];pub=tests.get('public_source');syn=tests.get('synthetic')
   rows.append(f"| {corpus} | {x['condition']} | {x['algorithm']} | {x['columns']} | {x['validation']['macro_f1']:.3f} | "+(f"{pub['supported_macro_f1']:.3f}" if pub else '—')+' | '+(f"{syn['macro_f1']:.3f}" if syn else '—')+' |')
 text=f'''# Character and OCR-layout feature study

Backend: **{r['backend']}**. Seed {SEED}. Character features use train-fitted character 3–5-gram TF-IDF (up to 1,500 columns); layout adds 22 train-standardized spatial measurements from automatic OCR boxes. Baseline is word TF-IDF + 15 numeric features + frozen CLIP. No Qwen or network calls were allowed during extraction, training or evaluation.

## Planned comparison

All four feature conditions use identical partitions and fixed classifier settings. When present, Public and Mixed models are selected on the same 189-row public validation set; Synthetic models use synthetic validation. Thresholds use the tenth percentile of correct validation margins. Test measurements do not select models. The prepared public source test includes 155 rows across six supported categories; 25 machine-labeled rows are measured separately in results.json. The source-test column below averages over supported classes only. Synthetic test covers all eight classes. A synthetic-only run does not evaluate public data.

| Training corpus | Features | Algorithm | Columns | Validation macro F1 | Public source test supported F1 | Synthetic test macro F1 |
|---|---|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'\n\n'
 if r['synthetic_inventory']:
  inv=r['synthetic_inventory']
  text+=f'''## Synthetic corpus

{inv['images']} fictional UI captures, {inv['per_category']} per category. Split counts: {inv['splits']}. There are {inv['groups']} category/configuration groups. Families 0–5 are training, 6–7 validation and 8–9 test. Configuration groups and exact image hashes do not cross partitions. Screens vary device dimensions, padding, alignment, colors, dark/light themes, text, prices and illustrated content. Category labels come directly from the generator. Text, group IDs, filenames and label metadata never enter X except text re-extracted from image pixels by OCR.

The renderer resembles phone interfaces, but its shared drawing logic and phrase banks make this an easier, artificial distribution. Neither a high synthetic score nor an improvement from synthetic augmentation establishes accuracy on real screenshots. The public source test and teacher-agreement limitations still apply.

![Balanced synthetic screenshot preview]({preview_relative})

'''
 text+='## Selected models\n\n'
 for corpus,data in r['corpora'].items():
  x=data['selected'];text+=f"- **{corpus}:** {x['algorithm']} / {x['condition']}; validation macro F1 {x['validation']['macro_f1']:.3f}. Local artifact: `{data['model']}`. Reload predictions identical.\n"
 text+='\n## Limits\n\n'+'\n'.join('- '+s for s in r['limitations'])+'\n\nComplete matrices, per-class results, teacher agreement and timings are in [results.json](results.json). Production content rules are excluded from these classifier metrics.\n'
 atomic(out/'REPORT.md',text.encode())
 fig,axes=plt.subplots(1,len(r['corpora']),figsize=(6*len(r['corpora']),4.8),squeeze=False,layout='constrained')
 for ax,(corpus,data) in zip(axes[0],r['corpora'].items()):
  for j,alg in enumerate(['Logistic regression','Linear SVM']):
   vals=[next(x['validation']['macro_f1'] for x in data['comparisons'] if x['condition']==name and x['algorithm']==alg) for name in CONDITIONS]
   ax.bar(np.arange(4)+(j-.5)*.36,vals,.35,label=alg)
  ax.set_xticks(range(4),['Baseline','Characters','Layout','Both'],rotation=15);ax.set_ylim(0,1.05)
  ax.set_title(corpus+' training');ax.set_ylabel('Validation macro F1');ax.legend(fontsize=8)
 fig.savefig(out/'validation-ablation.png',dpi=160);plt.close(fig)
 # Selected synthetic model's real eight-class synthetic holdout confusion matrix.
 if 'Synthetic' in r['corpora']:
  m=r['corpora']['Synthetic']['selected']['tests']['synthetic'];fig,ax=plt.subplots(figsize=(9,7),layout='constrained')
  ax.imshow(m['matrix'],cmap='Blues');ax.set_xticks(range(8),m['classes'],rotation=45,ha='right');ax.set_yticks(range(8),m['classes'])
  for i,row in enumerate(m['matrix']):
   for j,n in enumerate(row):ax.text(j,i,str(n),ha='center',va='center',color='white' if n>20 else 'black')
  ax.set_xlabel('Predicted');ax.set_ylabel('Generator label');ax.set_title('Synthetic-only model: held-out synthetic configurations')
  fig.savefig(out/'synthetic-confusion.png',dpi=160);plt.close(fig)


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
 g.add_argument('--public-only',action='store_true');g.add_argument('--synthetic-only',action='store_true');args=p.parse_args()
 run(args.public_only,args.synthetic_only)
