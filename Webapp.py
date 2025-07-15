import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode


# Load your trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'K',
               10:'L', 11:'M', 12:'N', 13:'O', 14:'P', 15:'Q', 16:'R', 17:'S',
               18:'T', 19:'U', 20:'V', 21:'W', 22:'X', 23:'Y', 24:'Z'}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)


class GestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.predicted_character = ""

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        H, W, _ = img.shape
        data_aux, x_, y_ = [], [], []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)

                x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10

                if len(data_aux) > model.n_features_in_:
                    data_aux = data_aux[:model.n_features_in_]

                if len(data_aux) == model.n_features_in_:
                    prediction = model.predict([np.asarray(data_aux)])
                    self.predicted_character = labels_dict[int(prediction[0])]

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(img, self.predicted_character, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                else:
                    self.predicted_character = ""

        return img


st.set_page_config(page_title="Adaptify - Gesture Recognition", page_icon="🤟", layout="centered")
st.title("🤟 Adaptify - Real-time ASL Gesture Recognition")
st.write("Press **Start** below to begin your webcam-based gesture recognition demo.")

webrtc_ctx = webrtc_streamer(
    key="gesture-recognition",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=GestureTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

if webrtc_ctx.video_transformer:
    st.markdown(f"### Last Detected Gesture: **{webrtc_ctx.video_transformer.predicted_character}**")
else:
    st.markdown("### Waiting for camera...")
