import joblib
import pytest
from classifier import learning


def test_model_accepts_same_contract_but_rejects_changed_features(tmp_path):
    model = learning.load_model()
    model["features"] = model["features"].replace("macOS 15.7.9", "macOS 26.1")
    path = tmp_path / "classifier.joblib"
    joblib.dump(model, path)
    assert learning.load_model(path)["training_id"] == model["training_id"]
    model["features"] = model["features"].replace("letterbox224", "different_preprocessing")
    joblib.dump(model, path)
    with pytest.raises(ValueError, match="Incompatible"):
        learning.load_model(path)


def test_macos_release_is_provenance_but_ocr_engine_is_a_contract():
    from classifier.features import compatible_version
    first="v1/Apple Vision rev3 accurate en-US macOS 15.7.9/cliphash"
    second="v1/Apple Vision rev3 accurate en-US macOS 26.1/cliphash"
    assert compatible_version(first)==compatible_version(second)
    assert compatible_version(first)!=compatible_version("v1/RapidOCR/cliphash")
