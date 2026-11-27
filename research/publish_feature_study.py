"""Promote only the public-data study winner when public validation improves.

Synthetic-only and mixed study artifacts stay separate. Test scores are not
consulted by this publication decision. Backups preserve the prepared baseline.
"""
import copy
import hashlib
import json
import shutil
from datetime import datetime,timezone
from classifier import dataset
from classifier.features import ROOT,OCR_BACKEND,atomic
from classifier.learning import default_model_path,default_report_dir,load_model
from research.report import build
from research.replay import replay


def main():
 folder=ROOT/'reports/feature-study'/OCR_BACKEND
 study=json.loads((folder/'results.json').read_text())['corpora']['Public']
 chosen=study['selected'];model_path=default_model_path();report_dir=default_report_dir()
 current=json.loads((report_dir/'results.json').read_text())
 previous=current['tasks']['content_type'];old=previous['selected']
 old_score=next(x['validation']['macro_f1'] for x in previous['comparisons'] if x['algorithm']==old['algorithm'] and x['features']==old['features'])
 if current['training_id']==study['training_id'] or chosen['validation']['macro_f1']<=old_score+1e-10:
  print(json.dumps(dict(published=False,reason='Already published or validation did not improve.',training_id=current['training_id'])))
  return
 candidate=ROOT/study['model'];bundle=load_model(candidate)
 assert len(bundle['tasks']['content_type']['model'].classes_)==8
 backup=ROOT/'data/backups'/('before-feature-study-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))/OCR_BACKEND
 backup.mkdir(parents=True)
 shutil.copy2(model_path,backup/'classifier.joblib')
 shutil.copy2(model_path.parent/'classifier.sha256',backup/'classifier.sha256')
 for p in report_dir.iterdir():
  if p.is_file():shutil.copy2(p,backup/p.name)
 audit=copy.deepcopy(current);t=audit['tasks']['content_type']
 for x in study['comparisons']:
  if x['condition']=='Baseline':continue
  t['comparisons'].append(dict(algorithm=x['algorithm'],features=x['features'],columns=x['columns'],
                              validation=x['validation'],ms=x['tests']['public_source']['inference_ms'],
                              settings={'study':'character-layout-v1','timing_partition':'public_source_test','fit_seconds':x['fit_seconds']}))
 t['selected']={k:chosen[k] for k in ('algorithm','features','threshold')}
 t['tests']={'test_public':chosen['tests']['public_source']}
 t['machine_labeled_test_agreement']=chosen['tests']['public_teacher']
 t['machine_labeled_test_agreement']['interpretation']='Held-out machine-label agreement, without independent human verification.'
 audit['training_id']=study['training_id']
 audit['feature_extensions']=dict(character_ngrams='3-5, char_wb, up to 1500 train-fitted features',
                                  spatial_dimensions=22,synthetic_training=False,baseline_validation_f1=old_score,
                                  selected_validation_f1=chosen['validation']['macro_f1'])
 try:
  atomic(model_path,candidate.read_bytes())
  atomic(model_path.parent/'classifier.sha256',(hashlib.sha256(candidate.read_bytes()).hexdigest()+'  classifier.joblib\n').encode())
  atomic(report_dir/'results.json',json.dumps(audit,indent=2,allow_nan=False).encode())
  checked=replay(dataset.connect())
  build(report_dir)
 except Exception:
  for p in backup.iterdir():
   target=(model_path if p.name=='classifier.joblib' else model_path.parent/'classifier.sha256' if p.name=='classifier.sha256' else report_dir/p.name)
   atomic(target,p.read_bytes())
  raise
 summary=dict(published=True,backend=OCR_BACKEND,training_id=study['training_id'],algorithm=chosen['algorithm'],features=chosen['features'],
              columns=chosen['columns'],previous_validation_f1=old_score,new_validation_f1=chosen['validation']['macro_f1'],
              public_test_accuracy=chosen['tests']['public_source']['accuracy'],synthetic_training=False,
              identical_source_test_replay=checked,backup=str(backup.relative_to(ROOT)))
 atomic(folder/'publication.json',json.dumps(summary,indent=2).encode())
 print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
