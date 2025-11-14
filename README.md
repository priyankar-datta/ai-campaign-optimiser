# AI Campaign Optimizer (Prototype)
An end-to-end prototype for an **AI Campaign Optimizer**: generates ad copy with LLM prompts, predicts CTR/CVR with XGBoost on synthetic data, simulates A/B tests (uplift-style), and exposes a FastAPI + Streamlit demo.

## Contents
- `data/` - synthetic data generation script and sample CSV
- `notebooks/` - a Jupyter notebook that runs EDA + trains a CTR model
- `models/` - saved model artifacts (after running training)
- `service/fastapi_app.py` - API that accepts a campaign brief and returns LLM-generated creatives + predicted KPIs (predictions mocked if model missing)
- `ui/streamlit_app.py` - simple Streamlit UI to demo the pipeline locally
- `prompts/prompt_library.json` - prompt templates and few-shot examples
- `train/train_xgb.py` - script to train XGBoost on synthetic data
- `Dockerfile` - container for the service
- `requirements.txt` - Python deps

## Quickstart (local)
1. Clone or download this repo.
2. Create a Python venv (Python 3.10+):
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Generate synthetic data & train:
   ```
   python data/generate_synthetic_data.py --rows 50000
   python train/train_xgb.py
   ```
   This produces `models/ctr_xgb.json` and `models/feature_metadata.json`.
4. Run demo:
   ```
   uvicorn service.fastapi_app:app --reload --port 8000
   streamlit run ui/streamlit_app.py
   ```
5. The Streamlit UI sends briefs to the FastAPI endpoint and displays generated creatives + predictions.

## Notes
- LLM calls are stubbed to avoid API keys in the repo. Replace the `llm_client` function in `service/fastapi_app.py` with your OpenAI/Azure/Mistral client.
- This is a prototype scaffold meant for course/portfolio use — adapt datasets & model as you plug in real platform data.

