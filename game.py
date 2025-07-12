import cv2
import mediapipe as mp
import random
import time
import math
import pygame
import os

# 📥 Load high score from file if exists
def load_high_score(filename="high_score.txt"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return int(f.read().strip())
    return 0

# 💾 Save high score to file
def save_high_score(score, filename="high_score.txt"):
    with open(filename, "w") as f:
        f.write(str(score))

high_score = load_high_score()

# 🎵 Load music and sound effects
pygame.mixer.init()
pop_sound = pygame.mixer.Sound("pop.mp3")
lose_sound = pygame.mixer.Sound("sad.mp3")
pygame.mixer.music.load("bg_music.mp3")
pygame.mixer.music.set_volume(0.3)
pygame.mixer.music.play(-1)  # Loop background music

# 🎨 Balloon colors
colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 165, 0)]

# 🎈 Balloon class
class Balloon:
    def __init__(self, x, y, radius=30, color=(0, 0, 255), speed=2):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.speed = speed
        self.popped = False

    def splash(self, frame):
        cv2.circle(frame, (self.x, self.y), self.radius + 10, (255, 255, 255), 3)

    def move(self):
        self.y -= self.speed

    def draw(self, frame):
        if not self.popped:
            cv2.circle(frame, (self.x, self.y), self.radius, self.color, -1)

    def is_off_screen(self, height):
        return self.y + self.radius <= 0  # Ensures balloon is completely off-screen

    def check_pop(self, fingertip_pos):
        fx, fy = fingertip_pos
        distance = math.hypot(self.x - fx, self.y - fy)
        return distance <= self.radius

# 🖐️ MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

def detect_thumb_gesture(hand_landmarks, w, h):
    lm = hand_landmarks.landmark

    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = lm[mp_hands.HandLandmark.THUMB_IP]
    wrist = lm[mp_hands.HandLandmark.WRIST]
    index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = lm[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    # Convert to pixel values
    thumb_tip_y = thumb_tip.y * h
    thumb_ip_y = thumb_ip.y * h
    wrist_y = wrist.y * h
    index_tip_y = index_tip.y * h
    index_mcp_y = index_mcp.y * h

    # 👍 Thumbs Up: thumb_tip above IP and wrist, fingers curled (index tip lower than MCP)
    if thumb_tip_y < thumb_ip_y and thumb_tip_y < wrist_y and index_tip_y > index_mcp_y:
        return "thumbs_up"

    # 👎 Thumbs Down: thumb_tip below IP and wrist, fingers curled (index tip lower than MCP)
    elif thumb_tip_y > thumb_ip_y and thumb_tip_y > wrist_y and index_tip_y > index_mcp_y:
        return "thumbs_down"

    return None

# 📷 Webcam capture
cap = cv2.VideoCapture(0)

# 🕹️ Game state
score = 0

while True:
    # 🌟 Show Ready-Set-Go!
    for msg in ["Ready", "Set", "Go!"]:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame[:] = (0, 0, 0)
        cv2.putText(frame, msg, (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)
        cv2.imshow("Balloon Pop Game 🎈", frame)
        cv2.waitKey(1000)

    # 🔁 Game loop
    balloons = []
    score = 0
    spawn_interval = 1.2
    last_spawn_time = time.time()
    game_over = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # 🧠 Hand detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        index_fingertips = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                x_tip = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
                y_tip = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
                cv2.circle(frame, (x_tip, y_tip), 15, (0, 255, 255), -1)
                index_fingertips.append((x_tip, y_tip))

        # 🎈 Spawn balloon
        if time.time() - last_spawn_time > spawn_interval:
            num_new = random.randint(2, 3)  # 2 or 3 balloons per interval
            for _ in range(num_new):
                x = random.randint(50, w - 50)
                color = random.choice(colors)
                speed = random.randint(5, 9)
                balloons.append(Balloon(x, h, color=color, speed=speed))
            last_spawn_time = time.time()

        # 🎯 Update balloons
        for balloon in balloons[:]:
            balloon.move()

            for tip in index_fingertips:
                if not balloon.popped and balloon.check_pop(tip):
                    balloon.popped = True
                    score += 1
                    balloon.splash(frame)
                    pop_sound.play()
                    break

            balloon.draw(frame)

            if not balloon.popped and balloon.is_off_screen(h):
                lose_sound.play()
                game_over = True
                if score > high_score:
                    high_score = score
                    save_high_score(high_score)
                break

        # 🧾 Score display
        cv2.putText(frame, f"Score: {score}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f"High Score: {high_score}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        if game_over:
            # 🎥 Game Over display with gesture-based control
            gesture_detected = None
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    for handLms in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                        gesture = detect_thumb_gesture(handLms, w, h)
                        if gesture == "thumbs_up":
                            gesture_detected = "restart"
                        elif gesture == "thumbs_down":
                            gesture_detected = "quit"
                s="Score: "+str(score)
                # Game Over message
                cv2.putText(frame, "Game Over!", (w//2 - 150, h//2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv2.putText(frame, "ThumbUp Restart  ThumbDown Quit", (w//2 - 305, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                cv2.putText(frame, s, (w//2 - 150, h//2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow("Balloon Pop Game 🎈", frame)

                if gesture_detected:
                    break

                if cv2.waitKey(10) & 0xFF == 27:
                    break  # Escape fallback

            # 🚦 Take action based on gesture
            if gesture_detected == "restart":
                balloons.clear()
                score = 0
                game_over = False
                break  # Restart loop
            elif gesture_detected == "quit":
                cap.release()
                cv2.destroyAllWindows()
                exit()
        # 🎥 Show frame
                # 🎥 Show frame
        cv2.imshow("Balloon Pop Game 🎈", frame)
        if cv2.waitKey(100) & 0xFF == 27:
            break
