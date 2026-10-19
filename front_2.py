import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict/"

st.set_page_config(page_title="Audio Model", page_icon="🎧")

st.title("SPEECHCOMMANDS")
st.write("Upload the file and model will try to recognize it")

# --- Запись с микрофона ---
recorded = st.audio_input("Record audio")

# --- Загрузка файла ---
uploaded = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg"])

# Приоритет: запись с микрофона, иначе загруженный файл
audio_bytes = None
filename = None
content_type = "audio/wav"

if recorded is not None:
    audio_bytes = recorded.getvalue()
    filename = "recording.wav"
elif uploaded is not None:
    audio_bytes = uploaded.getvalue()
    filename = uploaded.name
    content_type = uploaded.type or "audio/wav"
    st.audio(audio_bytes, format=content_type)

if st.button("Predict"):
    if audio_bytes is None:
        st.warning("Сначала запишите звук или загрузите файл.")
    else:
        try:
            with st.spinner("Recognizing..."):
                response = requests.post(
                    API_URL,
                    files={"file": (filename, audio_bytes, content_type)},
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
                st.success(f"**Label:** {result['class']}  \n**Index:** {result['index']}")
            else:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                st.error(f"Ошибка {response.status_code}: {detail}")