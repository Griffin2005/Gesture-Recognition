import cv2
import mediapipe as mp
import pickle
import numpy as np

#loading the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

#initializing the webcam
cap = cv2.VideoCapture(0)

#initializing mediapipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


x1, x2, y1, y2 = 0, 0, 0, 0

#label dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'}
while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame from webcam")
        break

    H, W, _ = frame.shape

    #convert the frame to rgb
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #process the frame with mediapipe
    results = hands.process(frame_rgb)

    #check for hand detections
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            #extract hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)

        #calculate bounding box
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10


        #limit data_aux to model's expected input size
        if len(data_aux) > model.n_features_in_:
            data_aux = data_aux[:model.n_features_in_]

        #perform prediction
        if len(data_aux) == model.n_features_in_:
            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]
            #print(predicted_character)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        else:
            print(f"Feature size mismatch. Expected {model.n_features_in_}, but got {len(data_aux)}")

    else:
        cv2.putText(frame, "No hand detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)

    #display the frame
    cv2.imshow('Gesture Recognition', frame)

    #breaking on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




    '''cv2.rectangle(frame, (x1, y1),(x2, y2), (0, 0, 0), 4 )
    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)'''


    ''''cv2.imshow('frame', frame)
    cv2.waitKey(1)'''

cap.release()
cv2.destroyAllWindows()
