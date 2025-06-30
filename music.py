import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import webbrowser

# Load model and labels with error handling
try:
    model = tf.keras.models.load_model("model.h5")
    label = np.load("labels.npy")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.title("🎶 Emotion Based Music Recommender 🎶")

# Initialize session state variables
if "lang" not in st.session_state:
    st.session_state.lang = ""
if "singer" not in st.session_state:
    st.session_state.singer = ""
if "emotion" not in st.session_state:
    st.session_state.emotion = ""
if "emotion_captured" not in st.session_state:
    st.session_state.emotion_captured = False


class EmotionProcessor:
    def __init__(self):
        self.last_emotion = ""

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            lst = np.array(lst).reshape(1, -1)

            if model is not None:
                pred = label[np.argmax(model.predict(lst))]
                cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
                self.last_emotion = pred

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1,
                                                                         circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


# Input fields
st.session_state.lang = st.text_input("Language 🌐", value=st.session_state.lang)
st.session_state.singer = st.text_input("Singer 🎤", value=st.session_state.singer)

# Emotion capture
if st.session_state.lang and st.session_state.singer:
    st.subheader("Capture Your Emotion")
    emotion_processor = EmotionProcessor()
    webrtc_ctx = webrtc_streamer(
        key="emotion_capture",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=lambda: emotion_processor,
        async_processing=True,
    )

    if webrtc_ctx.state.playing:
        st.write("Capturing emotion... Stop the webcam when ready.")
    elif webrtc_ctx.state.playing == False and emotion_processor.last_emotion:
        st.session_state.emotion = emotion_processor.last_emotion
        st.success(f"Emotion '{st.session_state.emotion}' captured successfully!")

        # Recommend button
        if st.button("Recommend Songs", key="recommend_songs"):
            search_query = f"{st.session_state.lang}+{st.session_state.emotion}+song+{st.session_state.singer}"
            webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
            st.success(
                f"Opening YouTube with {st.session_state.emotion} {st.session_state.lang} songs by {st.session_state.singer}")

# Display current inputs
if st.session_state.emotion:
    st.subheader("Current Inputs")
    st.write(f"Language: {st.session_state.lang}")
    st.write(f"Singer: {st.session_state.singer}")
    st.write(f"Captured Emotion: {st.session_state.emotion}")

# Start Over button
if st.button("Start Over"):
    for key in ["lang", "singer", "emotion", "emotion_captured"]:
        st.session_state[key] = ""
    st.experimental_rerun()