import os
import joblib
from typing import Tuple, Any

def _normalize_loaded(obj):
    """
    If saved as dict {'model':..., 'preprocessor':...}, return (model, preprocessor).
    Otherwise return (obj, None).
    """
    if isinstance(obj, dict):
        model = obj.get('model') or obj.get('estimator') or list(obj.values())[0]
        pre = obj.get('preprocessor') or obj.get('preprocessing') or None
        return model, pre
    else:
        return obj, None

def load_model_and_preprocessor(mode: str) -> Tuple[Any, Any]:
    """
    Try loading model + preprocessor from models/ folder for given mode.
    Mode 'fisik' expects xgboost_fisik.pkl & preprocessor_fisik.pkl
    Mode 'akurat' expects lasso_akurat_model.pkl & preprocessor_akurat.pkl
    Returns (model_obj, preprocessor_obj) or (None, None) if not found.
    """
    base = "models"
    model_obj = None
    preprocessor_obj = None

    if mode == "fisik":
        candidates = [
            ("xgboost_fisik.pkl", "preprocessor_fisik.pkl"),
            ("xgboost.pkl", "preprocessor_fisik.pkl"),
            ("xgb_fisik.pkl", "preprocessor_fisik.pkl"),
        ]
    else:
        candidates = [
            ("lasso_akurat_model.pkl", "preprocessor_akurat.pkl"),
            ("lasso.pkl", "preprocessor_akurat.pkl"),
            ("ridge.pkl", "preprocessor_akurat.pkl"),
        ]

    for mname, pname in candidates:
        mpath = os.path.join(base, mname)
        ppath = os.path.join(base, pname)
        if os.path.exists(mpath):
            loaded = joblib.load(mpath)
            model_obj, pre_tmp = _normalize_loaded(loaded)
            # prefer explicit preprocessor file
            if os.path.exists(ppath):
                try:
                    preprocessor_obj = joblib.load(ppath)
                except:
                    preprocessor_obj = pre_tmp or None
            else:
                preprocessor_obj = pre_tmp or None
            break

    return model_obj, preprocessor_obj

def load_from_uploaded(uploaded_file):
    """
    Load object from uploaded file (Streamlit UploadedFile).
    """
    return joblib.load(uploaded_file)
