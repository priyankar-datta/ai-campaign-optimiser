from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json, os
import random
import joblib
import xgboost as xgb
import numpy as np

app = FastAPI(title='AI Campaign Optimizer - Demo API')

# simple brief model
class Brief(BaseModel):
    product: str
    audience: str
    objective: str
    tone: str = "trustworthy"
    channel: str = "facebook"

# LLM stub / replace with actual API call
def llm_generate_variants(brief, n=6):
    # This is a placeholder. Replace with OpenAI/Mistral/GPT call.
    variants = []
    for i in range(n):
        tone = brief.tone
        variants.append({
            "variant_id": f"v_{random.randint(1000,9999)}",
            "headline": f"{brief.product} - {tone[:6]} {i}",
            "description": f"Buy {brief.product} now. Tailored for {brief.audience}.",
            "cta": random.choice(["Buy","Learn More","Sign Up"]),
            "tone": tone
        })
    return variants

# Load model if available
MODEL_PATH = 'models/ctr_xgb.json'
FEATURE_META = 'models/feature_metadata.json'
model = None
feature_meta = None
if os.path.exists(MODEL_PATH):
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
if os.path.exists(FEATURE_META):
    with open(FEATURE_META,'r') as f:
        feature_meta = json.load(f)

def fe_for_variants(variants, brief):
    # create dummy features aligned with training
    # This function must be adapted when you have real model features
    feats = []
    for v in variants:
        d = {
            'headline_len': len(v['headline']),
            'sentiment': 0.1
        }
        # one-hot placeholders
        d['channel_'+brief.channel]=1
        feats.append(d)
    # align to feature_meta if available
    if feature_meta:
        features = []
        for d in feats:
            row = []
            for f in feature_meta['features']:
                row.append(d.get(f,0))
            features.append(row)
        return np.array(features)
    else:
        # return simple array
        return np.array([[d['headline_len'], d['sentiment']] for d in feats])

@app.post('/generate')
def generate(brief: Brief):
    variants = llm_generate_variants(brief)
    X = fe_for_variants(variants, brief)
    preds = []
    if model is not None and feature_meta:
        dmat = xgb.DMatrix(X)
        preds = model.predict(dmat).tolist()
    else:
        # mock preds: base CTR per channel + some noise
        base = {'facebook':0.01,'instagram':0.012,'google_search':0.03,'youtube':0.008}.get(brief.channel,0.01)
        preds = [max(0.0001, base + (len(v['headline'])-30)/10000 + random.gauss(0,0.001)) for v in variants]
    for v,p in zip(variants,preds):
        v['predicted_ctr'] = round(p,6)
    return {'variants': variants}

if __name__=='__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
