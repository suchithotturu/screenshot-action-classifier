import numpy as np
from classifier.layout import NAMES,spatial_features
from classifier.learning import Fusion,load_model
from classifier.features import NUMERIC


def row(text,boxes=None):
 return dict(ocr=dict(text=text,blocks=boxes or []),numeric=[0.0]*len(NUMERIC),embedding=[1.0]+[0.0]*511)


def test_layout_positions_and_empty_input_are_finite():
 f=row('hello there',[dict(box=[0.0,0.1,0.2,0.1],confidence=.8),dict(box=[.7,.7,.2,.1],confidence=1)])
 v=spatial_features(f)
 assert len(v)==len(NAMES)==22 and np.isfinite(v).all()
 assert v[:9].sum()==1 and v[9:12].sum()==1
 assert v[0]==.5 and v[8]==.5
 assert v[-1]==.9
 assert np.array_equal(spatial_features(row('')),np.zeros(22))
 assert np.isfinite(spatial_features(row('',[dict(box=[float('nan'),0,1,1])]))).all()


def test_character_and_layout_modules_fit_train_only_and_add_columns():
 train=[row('receipt total paid'),row('message reply today'),row('receipt order paid'),row('message sent today')]
 base=Fusion('fusion').fit(train)
 both=Fusion('fusion_char_layout').fit(train)
 names=both.names.copy();vocabulary=both.char_vectorizer.vocabulary_.copy()
 test=row('zzzzzz validation only',[dict(box=[.8,.8,.1,.1],confidence=.7)])
 x=both.transform([test])
 assert both.names==names and both.char_vectorizer.vocabulary_==vocabulary
 assert 'zzz' not in vocabulary
 assert x.shape[1]==len(names)>base.transform([test]).shape[1]+22
 assert any(n.startswith('char:') for n in names) and 'text_right' in names
 assert np.isfinite(x.data).all()
 # Old pickled Fusion instances have no extension attributes.
 del base.char_vectorizer;del base.layout_scaler
 assert base.transform([test]).shape[1]==len(base.names)


def test_existing_deployed_artifact_remains_loadable():
 bundle=load_model();head=bundle['tasks']['content_type']
 x=head['transform'].transform([row('Hello send a message')])
 assert x.shape[1]==len(head['transform'].names)
 assert head['model'].predict(x).shape==(1,)


def test_synthetic_renderer_is_reproducible_and_varied():
 from research.synthetic_screens import render
 a=render('chat',8,0)
 b=render('chat',8,0)
 c=render('chat',8,1)
 assert a.size==b.size and a.tobytes()==b.tobytes()
 assert a.tobytes()!=c.tobytes()
 assert render('map_place',8,0).tobytes()!=a.tobytes()
