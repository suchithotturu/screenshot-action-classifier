"""Exercise a fitted pixel-feature artifact with real OCR/CLIP and no networking."""
import json,sys,tempfile,time
from pathlib import Path
from classifier import dataset,model,production,store
from classifier.features import ROOT,OCR_BACKEND
from classifier.learning import load_model,requires_pixels


def main():
 artifact=ROOT/'models/experiments/visual-study'/OCR_BACKEND/'fusion_char_visual/classifier.joblib'
 def offline(event,args):
  if event in ('socket.connect','socket.getaddrinfo'):raise AssertionError('Network forbidden during visual inference')
 def no_qwen(*args,**kwargs):raise AssertionError('Qwen forbidden during visual inference')
 model.api=no_qwen;sys.addaudithook(offline)
 bundle=load_model(artifact);assert requires_pixels(bundle)
 previous=store.DATA
 with tempfile.TemporaryDirectory() as temp:
  try:
   store.DATA=Path(temp);c=dataset.connect();start=time.perf_counter()
   _,first=production.predict_folder(c,ROOT/'demo',bundle=bundle);cold=time.perf_counter()-start
   _,again=production.predict_folder(c,ROOT/'demo',bundle=load_model(artifact))
   for a,b in zip(first['items'],again['items'],strict=True):
    assert {k:v for k,v in a.items() if k!='seconds'}=={k:v for k,v in b.items() if k!='seconds'}
   assert sum(r['status']=='ok' for r in first['items'])==10
   assert c.execute('SELECT count(*) FROM samples').fetchone()[0]==0
  finally:store.DATA=previous
 head=bundle['tasks']['content_type']
 result=dict(backend=OCR_BACKEND,training_id=bundle['training_id'],classifier_columns=len(head['transform'].names),
             pixel_columns=sum(n.startswith(('hog_','hsv_','edge_grid_')) for n in head['transform'].names),
             valid_images=10,expected_corrupt_images=1,expected_unsupported_files=1,empty_cache=True,
             cold_run_seconds=cold,identical_after_reload=True,qwen_calls=0,python_network_connections=0,
             limitation='Workflow check on macOS, not native Windows verification or an accuracy benchmark.')
 target=ROOT/'reports/visual-study'/OCR_BACKEND/'offline-check.json'
 target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
