import joblib
import pytest
from classifier import learning


def test_model_accepts_other_macos_but_rejects_changed_features(tmp_path):
    model = learning.load_model()
    model["features"] = model["features"].replace("macOS 15.7.9", "macOS 26.1")
    path = tmp_path / "classifier.joblib"
    joblib.dump(model, path)
    assert learning.load_model(path)["training_id"] == model["training_id"]
    model["features"] = model["features"].replace("letterbox224", "different_preprocessing")
    joblib.dump(model, path)
    with pytest.raises(ValueError, match="Incompatible"):
        learning.load_model(path)
