import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def recognize_gesture(hand_landmark):
    ujung_jempol = hand_landmark.landmark[mp_hands.HandLandmark.THUMB_TIP]
    ujung_telunjuk = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    ujung_jariTengah = hand_landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ujung_jarimanis = hand_landmark.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ujung_kelingking = hand_landmark.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pangkal_telunjuk = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pangkal_jariTengah = hand_landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    pangkal_jarimanis = hand_landmark.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pangkal_kelingking = hand_landmark.landmark[mp_hands.HandLandmark.PINKY_MCP]
    
    if (ujung_jempol.y < ujung_telunjuk.y and
        ujung_jempol.y < ujung_jariTengah.y and
        ujung_jempol.y < ujung_jarimanis.y and
        ujung_jempol.y < ujung_kelingking.y):
        return "Hebat"
    if (ujung_jempol.y < ujung_telunjuk.y and
        ujung_jempol.y < ujung_jariTengah.y and
        ujung_jempol.y < ujung_jarimanis.y and
        ujung_jempol.y > ujung_kelingking.y):
        return "semut"

    if (ujung_telunjuk.y < ujung_jempol.y and
        ujung_jariTengah.y < ujung_jempol.y and
        ujung_jarimanis.y > ujung_jempol.y and
        ujung_kelingking.y > ujung_jempol.y):
        return "Peace Sign"
    
    if (ujung_telunjuk.y < ujung_jempol.y and
        ujung_jariTengah.y < ujung_jempol.y and
        ujung_jarimanis.y < ujung_jempol.y and
        ujung_kelingking.y > ujung_jempol.y):
        return "angka 3"
  
    if (ujung_telunjuk.y < ujung_jempol.y and
        ujung_kelingking.y < ujung_jempol.y and
        ujung_jariTengah.y > ujung_jempol.y and
        ujung_jarimanis.y > ujung_jempol.y):
        return "I Love You Sign"
    
    if (ujung_telunjuk.y < pangkal_telunjuk.y and
        ujung_jariTengah.y < pangkal_jariTengah.y and
        ujung_jarimanis.y < pangkal_jarimanis.y and
        ujung_kelingking.y < pangkal_kelingking.y):
        return "Buka Tangan"

    if (ujung_telunjuk.y > pangkal_telunjuk.y and
        ujung_jariTengah.y > pangkal_jariTengah.y and
        ujung_jarimanis.y > pangkal_jarimanis.y and
        ujung_kelingking.y > pangkal_kelingking.y):
        return "Mengepalkan tangan"

    if (ujung_telunjuk.y < pangkal_telunjuk.y and
        ujung_jariTengah.y < pangkal_jariTengah.y and
        ujung_jarimanis.y < pangkal_jarimanis.y and
        ujung_kelingking.y < pangkal_kelingking.y and
        ujung_telunjuk.x > ujung_jempol.x and
        ujung_jariTengah.x > ujung_jempol.x and
        ujung_jarimanis.x > ujung_jempol.x and
        ujung_kelingking.x > ujung_jempol.x):
        return "Terima Kasih"
    
    if (abs(ujung_jempol.x - ujung_telunjuk.x) < 0.05 and
        abs(ujung_jempol.y - ujung_telunjuk.y) < 0.05 and
        abs(ujung_jariTengah.y - ujung_telunjuk.y) > 0.1 and
        abs(ujung_jarimanis.y - ujung_telunjuk.y) > 0.1 and
        abs(ujung_kelingking.y - ujung_telunjuk.y) > 0.1):
        return "OK"

    return "Gesture Tidak Diketahui"

def detect_hand_gesture(image, hand):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hand.process(image_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            gesture = recognize_gesture(hand_landmarks)
            

            h, w, _ = image.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
            hand_type = "kiri" if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < 0.5 else "Kanan"
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            cv2.putText(image, f"{gesture} ({hand_type})", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    return image

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Tidak dapat membuka kamera")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        print("Gagal menangkap frame")
        break
    
    frame = cv2.flip(frame, 1)
    
    frame = detect_hand_gesture(frame, hands)
    cv2.imshow("Hand Gesture", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
