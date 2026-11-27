"""Pixel-feature ablation on the existing eight-category public partitions."""
import argparse,copy,hashlib,io,json,platform,sys,time,warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
from classifier import dataset,model,store
from classifier.features import ROOT,VERSION,OCR_BACKEND,atomic,extract
from classifier.pixels import VERSION as PIXEL_VERSION,NAMES as PIXEL_NAMES
from classifier.learning import Fusion,SEED,load_model,requires_pixels
from research.feature_study import public_rows,score

CONDITIONS={
 'Current classifier':'fusion_char',
 '+ HOG':'fusion_char_hog',
 '+ HSV colors':'fusion_char_color',
 '+ Edge grid':'fusion_char_edges',
 '+ All visual':'fusion_char_visual',
 'HOG only':'pixels_hog',
 'Colors only':'pixels_color',
 'Edge grid only':'pixels_edges',
 'Engineered visuals only':'pixels_all',
 'CLIP only':'clip',
 'CLIP + engineered visuals':'clip_visual',
}


def run(include_synthetic=True):
 def offline(event,args):
  if event in ('socket.connect','socket.getaddrinfo'):raise AssertionError('Network blocked during visual study')
 def no_qwen(*args,**kwargs):raise AssertionError('Qwen blocked during visual study')
 model.api=no_qwen;sys.addaudithook(offline)
 folder=ROOT/'reports/visual-study'/OCR_BACKEND;folder.mkdir(parents=True,exist_ok=True)
 c=dataset.connect();rows=public_rows(c)
 assert len([r for r in rows if r['split']=='train'])==911
 synthetic=[]
 manifest=ROOT/'data/synthetic-screens/manifest.jsonl'
 if include_synthetic and manifest.exists():
  for line in manifest.read_text().splitlines():
   r=json.loads(line)
   if r['split']=='test':r['path']=manifest.parent/r['path'];synthetic.append(r)
 start=time.perf_counter()
 def one(row):
  f=extract(row['path'].read_bytes(),store.DATA/'features',False,True)
  if f['status']!='ok':raise RuntimeError('Automatic features unavailable for '+row['id'])
  return dict(row,f=f)
 with ThreadPoolExecutor(3) as pool:
  full=[]
  for n,row in enumerate(pool.map(one,rows+synthetic),1):
   full.append(row)
   if n%200==0:print(f'Visual descriptors {n}/{len(rows)+len(synthetic)}',flush=True)
 data=full[:len(rows)];synthetic=full[len(rows):]
 train=[r for r in data if r['split']=='train'];valid=[r for r in data if r['split']=='validation']
 tests={'public_source':[r for r in data if r['split']=='test' and r['origin']!='pseudo'],
        'public_teacher':[r for r in data if r['split']=='test' and r['origin']=='pseudo'],
        'synthetic_transfer':synthetic}
 baseline=load_model()
 report=dict(backend=OCR_BACKEND,tested_host=f'{platform.system()} {platform.machine()}',base_feature_version=VERSION,
             pixel_feature_version=PIXEL_VERSION,pixel_dimensions={k:len(v) for k,v in PIXEL_NAMES.items()},seed=SEED,
             train_n=len(train),validation_n=len(valid),test_n={k:len(v) for k,v in tests.items()},
             training_origins={k:sum(r['origin']==k for r in train) for k in ('source','pseudo')},
             validation_origins={k:sum(r['origin']==k for r in valid) for k in ('source','pseudo')},
             feature_preparation_seconds=time.perf_counter()-start,
             median_pixel_extraction_ms=float(np.median([r['f']['engineered_visual']['seconds']*1000 for r in full])),
             group_overlaps=0,qwen_calls=0,python_network_connections=0,synthetic_training=False,comparisons=[])
 groups={p:{r['group'] for r in data if r['split']==p} for p in ('train','validation','test')}
 assert not any(groups[a]&groups[b] for a in groups for b in groups if a<b)
 best=None
 report['condition_models']={}
 for condition,mode in CONDITIONS.items():
  condition_best=None
  transform=Fusion(mode).fit([r['f'] for r in train]);xt=transform.transform([r['f'] for r in train]);xv=transform.transform([r['f'] for r in valid])
  yt=[r['label'] for r in train];yv=[r['label'] for r in valid]
  for algorithm,estimator in [('Logistic regression',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=SEED)),
                              ('Linear SVM',LinearSVC(class_weight='balanced',dual='auto',max_iter=30000,random_state=SEED))]:
   start=time.perf_counter()
   with warnings.catch_warnings(),threadpool_limits(limits=2):
    warnings.simplefilter('error',ConvergenceWarning);estimator.fit(xt,yt)
   fit_seconds=time.perf_counter()-start
   pred=estimator.predict(xv)
   from classifier.learning import metrics,margins
   val=metrics(yv,pred,list(estimator.classes_));correct=margins(estimator,xv)[np.asarray(yv)==pred]
   threshold=float(np.quantile(correct,.1)) if len(correct) else float(np.finfo(float).max)
   head=dict(algorithm=algorithm,features=mode,transform=transform,model=estimator,threshold=threshold)
   x=dict(condition=condition,features=mode,algorithm=algorithm,columns=xt.shape[1],fit_seconds=fit_seconds,
          solver_iterations=int(np.max(estimator.n_iter_)),solver_converged=True,validation=val,threshold=threshold,
          tests={k:score(head,v) for k,v in tests.items() if v})
   report['comparisons'].append(x)
   if condition_best is None or val['macro_f1']>condition_best[0]:condition_best=val['macro_f1'],head,x
   # Select only additions to the prepared pipeline. Pure image candidates are
   # diagnostic controls; they do not discard the useful existing text branch.
   if condition=='Current classifier' or condition.startswith('+'):
    key=(val['macro_f1'],-xt.shape[1])
    if best is None or key>best[0]:best=key,head,x
   print(f"{condition} / {algorithm}: validation macro F1={val['macro_f1']:.4f}",flush=True)
   atomic(folder/'results.json',json.dumps(report,indent=2,allow_nan=False).encode())
  _,condition_head,condition_result=condition_best
  condition_artifact=copy.copy(baseline);condition_artifact['tasks']=dict(baseline['tasks'])
  condition_artifact['tasks']['content_type']=condition_head
  condition_artifact['training_id']=hashlib.sha256((PIXEL_VERSION+VERSION+mode+condition_head['algorithm']).encode()).hexdigest()[:16]
  if requires_pixels(condition_artifact):condition_artifact['pixel_features_version']=PIXEL_VERSION
  else:condition_artifact.pop('pixel_features_version',None)
  condition_path=ROOT/'models/experiments/visual-study'/OCR_BACKEND/mode/'classifier.joblib'
  buf=io.BytesIO();joblib.dump(condition_artifact,buf,compress=3);atomic(condition_path,buf.getvalue())
  report['condition_models'][condition]=str(condition_path.relative_to(ROOT))
 _,head,chosen=best
 ident=hashlib.sha256(json.dumps(dict(experiment='visual-study-v1',pixels=PIXEL_VERSION,base=VERSION,
               ids=[(r['id'],r['label']) for r in train],mode=head['features'],algorithm=head['algorithm']),sort_keys=True).encode()).hexdigest()[:16]
 artifact=copy.copy(baseline);artifact['tasks']=dict(baseline['tasks']);artifact['tasks']['content_type']=head
 artifact.update(training_id=ident,study=dict(kind='visual-only-feature-improvements',synthetic_training=False))
 if requires_pixels(artifact):artifact['pixel_features_version']=PIXEL_VERSION
 else:artifact.pop('pixel_features_version',None)
 path=ROOT/'models/experiments/visual-study'/OCR_BACKEND/'classifier.joblib'
 buf=io.BytesIO();joblib.dump(artifact,buf,compress=3);atomic(path,buf.getvalue())
 reloaded=load_model(path)
 assert np.array_equal(reloaded['tasks']['content_type']['model'].predict(xv if head['features']==mode else head['transform'].transform([r['f'] for r in valid])),head['model'].predict(head['transform'].transform([r['f'] for r in valid])))
 report.update(selected={k:chosen[k] for k in ('condition','features','algorithm','columns','validation','threshold','tests')},
               training_id=ident,model=str(path.relative_to(ROOT)),identical_after_reload=True)
 atomic(folder/'results.json',json.dumps(report,indent=2,allow_nan=False).encode());write_report(report,folder)
 print(json.dumps({k:report[k] for k in ('training_id','model','median_pixel_extraction_ms')},indent=2),flush=True)
 return report


def write_report(r,folder):
 import os
 os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.runtime/matplotlib'))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rows=[]
 for x in r['comparisons']:
  source=x['tests']['public_source'];syn=x['tests'].get('synthetic_transfer')
  rows.append(f"| {x['condition']} | {x['algorithm']} | {x['columns']} | {x['validation']['macro_f1']:.3f} | {source['accuracy']:.3f} | {source['supported_macro_f1']:.3f} | "+(f"{syn['accuracy']:.3f}" if syn else '—')+' |')
 s=r['selected']
 text=f'''# Engineered visual feature study

Backend: {r['backend']}; host: {r['tested_host']}; seed {SEED}. No Enrico expansion or new datasets were used. All candidates use the existing eight-category public corpus and the same frozen app/duplicate groups: 911 train, 189 validation, 155 source-labeled test rows and 25 machine-labeled test rows. Zero tracked group overlap. Public validation includes machine labels. Scores remain exploratory because public data have been inspected previously.

## Pixel descriptors

- **HOG: 3,780** values. Grayscale 128×256 aspect-preserving portrait canvas, neutral padding, 16×16 cells, 2×2-cell blocks, nine orientations, L2-Hys normalization.
- **HSV colors: 24** values. Eight bins per channel from an unpadded thumbnail; each channel histogram sums to one.
- **Edge grid: 16** values. Canny thresholds 100/200, L2 gradient, a 4×4 grid containing edge-pixel proportions.

These descriptors use no OCR, annotations, file names, app IDs or learned model. Train-fitted standardization followed by L2 normalization per descriptor block keeps the 3,780-dimensional HOG block from dominating simply through its size. Pixel extraction has a separate versioned cache. Median initial descriptor extraction on this Mac: **{r['median_pixel_extraction_ms']:.2f} ms** per image (decoding/cache I/O not included).

The baseline is the current word/character TF-IDF + numeric + CLIP model. Its text pipeline stays fixed. Pure engineered-image and CLIP candidates are diagnostic controls. The selected deployment candidate is chosen only from baseline and its visual additions, using validation macro F1 then fewer columns. Test metrics do not select the model. Synthetic screenshots, when available, are transfer tests only and never enter training.

| Features | Algorithm | Columns | Validation macro F1 | Source-test accuracy | Source-test supported F1 | Synthetic-transfer accuracy |
|---|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

## Selected candidate

**{s['condition']} / {s['algorithm']}**, {s['columns']} columns; validation macro F1 **{s['validation']['macro_f1']:.3f}**. Artifact: `{r['model']}`. Reload predictions are identical. All fits converged. Qwen calls and Python network connections were blocked.

The source test has six source-supported categories; shopping and entertainment lack independent test labels. Machine-label test agreement is reported separately in results.json. The synthetic-transfer images share renderer logic and are not an independent real-image benchmark. No source-only score establishes broad eight-category accuracy. Model metrics precede confidence routing and OCR rules.

![Visual-feature ablation](ablation.png)
'''
 atomic(folder/'REPORT.md',text.encode())
 fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
 groups=[list(CONDITIONS)[:5],list(CONDITIONS)[5:]]
 for ax,names,title in zip(axes,groups,['Additions to the current classifier','Image-only controls']):
  for j,alg in enumerate(['Logistic regression','Linear SVM']):
   vals=[next(x['validation']['macro_f1'] for x in r['comparisons'] if x['condition']==name and x['algorithm']==alg) for name in names]
   ax.bar(np.arange(len(names))+(j-.5)*.36,vals,.35,label=alg)
  ax.set_xticks(range(len(names)),names,rotation=25,ha='right');ax.set_ylim(0,1)
  ax.set_ylabel('Validation macro F1');ax.set_title(title);ax.legend(fontsize=8)
 fig.savefig(folder/'ablation.png',dpi=160);plt.close(fig)


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--no-synthetic',action='store_true',help='Omit the optional synthetic transfer test.')
 args=parser.parse_args();run(not args.no_synthetic)
