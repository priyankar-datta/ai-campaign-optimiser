import streamlit as st
import requests
st.set_page_config(page_title='AI Campaign Optimizer Demo')

st.title('AI Campaign Optimizer — Demo')

with st.form('brief'):
    product = st.text_input('Product', 'Organic Face Cream')
    audience = st.text_input('Audience', 'Women 25-40, skincare-interested')
    objective = st.selectbox('Objective', ['purchase','lead','awareness'])
    tone = st.selectbox('Tone', ['trustworthy','playful','urgent','luxury'])
    channel = st.selectbox('Channel', ['facebook','instagram','google_search','youtube'])
    submitted = st.form_submit_button('Generate')

if submitted:
    payload = {'product':product,'audience':audience,'objective':objective,'tone':tone,'channel':channel}
    try:
        resp = requests.post('http://localhost:8000/generate', json=payload, timeout=5).json()
        variants = resp.get('variants',[])
        for v in variants:
            st.subheader(v['headline'])
            st.write(v['description'])
            st.write('CTA:', v['cta'],'| Tone:', v['tone'],'| Pred CTR:', v['predicted_ctr'])
    except Exception as e:
        st.error('API call failed: '+str(e))
        st.info('Make sure the FastAPI server is running (uvicorn service.fastapi_app:app --reload --port 8000)')
