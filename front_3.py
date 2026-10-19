import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict/"

st.set_page_config(page_title="SPEECHCOMMANDS", page_icon="🎧")

st.title("SPEECHCOMMANDS")

mode = st.radio("Тандоо кыл:", ["WAV файл жүктөө", "Голос жазуу"])

audio_bytes = None
filename = None

if mode == "WAV файл жүктөө":
    uploaded = st.file_uploader("WAV файл жүктө", type=["wav"])
    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        filename = uploaded.name
        st.audio(audio_bytes, format="audio/wav")
else:
    recorded = st.audio_input("Голос жазуу")
    if recorded is not None:
        audio_bytes = recorded.getvalue()
        filename = "recording.wav"

if st.button("Проверка"):
    if audio_bytes is None:
        st.warning("Алгач WAV файл жүктөңүз же үн жазыңыз.")
    else:
        try:
            with st.spinner("..."):
                response = requests.post(
                    API_URL,
                    files={"file": (filename, audio_bytes, "audio/wav")},
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
                st.success(f"Модель думает, что это жанр: {result['class']}")
            else:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                st.error(f"Ошибка {response.status_code}: {detail}")