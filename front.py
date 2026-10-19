import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict/"
TITLE = "SPEECHCOMMANDS"  # можно заменить на "Urban Sounds Classifier"

st.set_page_config(page_title=TITLE, page_icon="🎧")

st.title(TITLE)
st.write("Upload audio (.wav) to recognize sound")

uploaded = st.file_uploader("Upload a file", type=["wav"])

if uploaded is not None:
    audio_bytes = uploaded.getvalue()
    st.audio(audio_bytes, format="audio/wav")

    if st.button("check"):
        try:
            with st.spinner("Recognizing..."):
                response = requests.post(
                    API_URL,
                    files={"file": (uploaded.name, audio_bytes, "audio/wav")},
                    timeout=30,
                )
        except requests.exceptions.ConnectionError:
            st.error(
                "Не удалось подключиться к backend. "
                "Убедитесь, что main.py запущен на http://127.0.0.1:8000"
            )
        except requests.exceptions.Timeout:
            st.error("Backend не ответил вовремя. Попробуйте ещё раз.")
        else:
            if response.ok:
                result = response.json()
                st.success(f"Index: {result['index']}, Sound: {result['class']}")
            else:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                st.error(f"Ошибка {response.status_code}: {detail}")