import io,json
import joblib
import numpy as np
import pytest
from PIL import Image,ImageDraw
from sklearn.linear_model import LogisticRegression
from classifier import features,learning,pixels,production


def test_pixel_shapes_colors_and_spatial_information():
 red=pixels.describe(Image.new('RGB',(128,256),'red'))
 blue=pixels.describe(Image.new('RGB',(128,256),'blue'))
 assert {k:len(red[k]) for k in pixels.NAMES}=={'hog':3780,'color':24,'edges':16}
 assert np.allclose(np.asarray(red['color']).reshape(3,8).sum(axis=1),1)
 assert red['color'][0]==1 and blue['color'][5]==1
 assert not any(red['hog']) and not any(red['edges'])
 im=Image.new('RGB',(128,256),'black');d=ImageDraw.Draw(im)
 d.rectangle((5,5,26,52),fill='white')
 textured=pixels.describe(im)
 assert np.linalg.norm(textured['hog'])>0
 assert textured['edges'][0]>0 and textured['edges'][-1]==0
 assert all(np.isfinite(textured[k]).all() for k in pixels.NAMES)


def test_pixel_cache_is_versioned_and_avoids_recomputation(tmp_path,monkeypatch):
 im=Image.new('RGB',(128,256),'green');buf=io.BytesIO();im.save(buf,'PNG');raw=buf.getvalue()
 first=pixels.extract_pixels(raw,tmp_path)
 def unexpected(_):raise AssertionError('Cached descriptors recomputed')
 monkeypatch.setattr(pixels,'describe',unexpected)
 assert pixels.extract_pixels(raw,tmp_path)==first
 p=next(tmp_path.glob('*.json'));doc=json.loads(p.read_text());doc['version']='old';p.write_text(json.dumps(doc))
 with pytest.raises(AssertionError,match='recomputed'):pixels.extract_pixels(raw,tmp_path)


def pixel_rows():
 result=[]
 for color in ['red','blue','green','white']:
  im=Image.new('RGB',(128,256),color)
  ImageDraw.Draw(im).rectangle((12,20,60,90),outline='black',width=3)
  result.append(dict(engineered_visual=pixels.describe(im),embedding=[1.]+[0.]*511))
 return result


def test_image_only_modes_do_not_read_ocr_and_fit_scaling_on_train_only():
 rows=pixel_rows()
 for mode,n in [('pixels_hog',3780),('pixels_color',24),('pixels_edges',16),('pixels_all',3820),('clip_visual',4332)]:
  transform=learning.Fusion(mode).fit(rows[:3])
  mean=transform.pixel_scaler.mean_.copy()
  x=transform.transform(rows[3:])
  assert x.shape==(1,n) and len(transform.names)==n
  assert np.array_equal(mean,transform.pixel_scaler.mean_) and np.isfinite(x.data).all()
  assert transform.vectorizer is None and transform.scaler is None


def test_pixel_artifact_reload_and_production_request_the_pixel_pipeline(tmp_path,monkeypatch):
 rows=pixel_rows();transform=learning.Fusion('pixels_all').fit(rows)
 model=LogisticRegression().fit(transform.transform(rows),['chat','other','chat','other'])
 head=dict(model=model,transform=transform,threshold=0,algorithm='Logistic regression',features='pixels_all')
 bundle=dict(format=1,features=features.VERSION,sklearn=learning.sklearn.__version__,training_id='test',
             pixel_features_version=pixels.VERSION,tasks={'content_type':head})
 path=tmp_path/'model.joblib';joblib.dump(bundle,path)
 loaded=learning.load_model(path)
 assert learning.requires_pixels(loaded)
 row={**rows[0], 'status':'ok','sha256':'fixture','ocr':{'text':'Hello','blocks':[]}}
 calls=[]
 def fake(*args):calls.append(args);return row
 monkeypatch.setattr(production,'extract',fake)
 result=production.predict_image(loaded,b'image')
 assert calls[0][3] is True and result['tasks']['content_type']['label'] in ('chat','other')
 bundle['pixel_features_version']='wrong';joblib.dump(bundle,path)
 with pytest.raises(ValueError,match='pixel feature'):learning.load_model(path)
