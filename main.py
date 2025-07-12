import cv2
import mediapipe as mp
import numpy as np
import time
import datetime
import os
import random

# --- Constants ---
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
BUTTON_WIDTH = 120
BUTTON_HEIGHT = 60
BUTTON_COLOR = (255, 178, 102)  # Orange-ish
BUTTON_TEXT_COLOR = (255, 255, 255)  # White
POLAROID_WIDTH = 400
POLAROID_HEIGHT = 500
POLAROID_MARGIN = 50
POLAROID_FONT = cv2.FONT_HERSHEY_SIMPLEX
POLAROID_FONT_SCALE = 0.7
POLAROID_FONT_COLOR = (0, 0, 0)
SNAP_GESTURE_THRESHOLD = 30  # Distance in pixels to trigger snap
FIST_GESTURE_THRESHOLD = 50 # Sum of distances for all fingers to trigger fist

# --- Helper Functions ---
def distance(point1, point2):
    return int(((point1.x - point2.x)**2 + (point1.y - point2.y)**2)**0.5 * 100) #Scaled up to avoid small decimals


def draw_button(frame, x, y, width, height, text, color, text_color):
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, -1)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = x + (width - text_size[0]) // 2
    text_y = y + (height + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)


def add_date_time(image):
    now = datetime.datetime.now()
    date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(image, date_time_str, (20, POLAROID_HEIGHT - 30), POLAROID_FONT, POLAROID_FONT_SCALE, POLAROID_FONT_COLOR, 2)
    return image


def stack_polaroids(image1, image2, image3):
    # Resize images to standard Polaroid size
    resized_image1 = cv2.resize(image1, (POLAROID_WIDTH, POLAROID_HEIGHT))
    resized_image2 = cv2.resize(image2, (POLAROID_WIDTH, POLAROID_HEIGHT))
    resized_image3 = cv2.resize(image3, (POLAROID_WIDTH, POLAROID_HEIGHT))

    # Add date/time stamp
    resized_image1 = add_date_time(resized_image1)
    resized_image2 = add_date_time(resized_image2)
    resized_image3 = add_date_time(resized_image3)

    # Create composite image
    composite_image = np.vstack((resized_image1, resized_image2, resized_image3))
    return composite_image


def save_image(image, folder="snaps"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{folder}/snap_{timestamp}.png"
    cv2.imwrite(filename, image)
    print(f"Image saved to {filename}")


# --- Filters ---
def apply_normal(frame):
    return frame

def apply_sketch(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray_blurred, 10, 70)  # Experiment with thresholds
    sketch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return sketch

def apply_glow_up(frame):
    # Brighten the image
    brightened = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)

    # Apply a bilateral filter for skin smoothing
    smoothed = cv2.bilateralFilter(brightened, d=15, sigmaColor=75, sigmaSpace=75)

    return smoothed

def apply_soft_glow(frame):
    # Convert to float for calculations
    frame_float = frame.astype(np.float32) / 255.0

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(frame_float, (15, 15), 0)

    # Blend the original with the blurred version
    soft_glow = cv2.addWeighted(frame_float, 0.8, blurred, 0.5, 0)

    # Convert back to uint8
    soft_glow = (soft_glow * 255).astype(np.uint8)

    # Apply a warm tone
    soft_glow[:, :, 0] = np.clip(soft_glow[:, :, 0] * 1.1, 0, 255)
    soft_glow[:, :, 1] = np.clip(soft_glow[:, :, 1] * 1.05, 0, 255)

    return soft_glow

def apply_pop_art(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Boost saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8, 0, 255)

    # Increase contrast
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Convert back to BGR and return
    pop_art = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    pop_art = cv2.addWeighted(pop_art, 0.7, enhanced_frame, 0.3, 0)
    return pop_art


# --- Main Application ---
def main():
    # --- Initialize MediaPipe ---
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                            max_num_hands=1,
                            min_detection_confidence=0.7,
                            min_tracking_confidence=0.7)

    # --- Webcam Setup ---
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # --- UI State ---
    filter_options = ["Normal", "Sketch", "Glow Up", "Soft Glow", "Pop Art"]
    current_filter_index = 0
    filter_functions = [apply_normal, apply_sketch, apply_glow_up, apply_soft_glow, apply_pop_art]

    taking_photo = False
    countdown_start_time = 0
    countdown_duration = 3  # seconds
    countdown_text = ""
    photo_taken = False
    photo_stack = []

    polaroid_prompt = False
    yes_button_active = False
    no_button_active = False
    show_final_polaroid = False
    final_polaroid_image = None

    # --- Button Positions ---
    button_x = CAMERA_WIDTH
    button_y_start = 50
    button_spacing = 80
    yes_button_pos = (CAMERA_WIDTH // 2 - 70, CAMERA_HEIGHT // 2 + 80)
    no_button_pos = (CAMERA_WIDTH // 2 + 70, CAMERA_HEIGHT // 2 + 80)

    # --- Start Screen (Optional) ---
    background_color = (255, 255, 255)  # Default white
    start_screen = True
    color_circles = [(100, 100, (255, 0, 0)), (200, 100, (0, 255, 0)), (300, 100, (0, 0, 255))]  # Red, Green, Blue
    color_circle_radius = 30

    while start_screen:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)

        for x, y, color in color_circles:
            cv2.circle(frame, (x, y), color_circle_radius, color, -1)

        cv2.putText(frame, "Hover over a circle to choose background color", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, "Press 's' to start!", (50, CAMERA_HEIGHT - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Hand tracking for color selection
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Fix: Convert to RGB *ONCE* here
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                ih, iw, _ = frame.shape
                x, y = int(index_finger_tip.x * iw), int(index_finger_tip.y * ih)

                for cx, cy, color in color_circles:
                    if (x - cx)**2 + (y - cy)**2 < color_circle_radius**2:
                        background_color = color
                        break

        cv2.imshow("Photo Booth - Choose Background Color", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            start_screen = False
        elif key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            return

    # --- Main Loop ---
    last_snap_gesture_time = 0
    snap_gesture_cooldown = 0.5  # seconds

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)  # Mirror the frame

        # Set the background color
        frame[:] = background_color

        # --- Hand Tracking ---
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Fix: Convert to RGB *ONCE* here
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # --- Gesture Detection ---

                # Snap Gesture (Index and Middle fingers close and apart)
                index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
                dist = distance(index_tip, middle_tip)

                if dist < SNAP_GESTURE_THRESHOLD and time.time() - last_snap_gesture_time > snap_gesture_cooldown:
                    current_filter_index = (current_filter_index + 1) % len(filter_options)
                    last_snap_gesture_time = time.time()
                    print(f"Filter changed to: {filter_options[current_filter_index]}")

                # Fist Gesture (all fingers closed)
                thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
                ring_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]

                fist_distance_sum = distance(index_tip, thumb_tip) + distance(middle_tip, thumb_tip) + distance(ring_tip, thumb_tip) + distance(pinky_tip, thumb_tip)
                if fist_distance_sum < FIST_GESTURE_THRESHOLD and not taking_photo and not photo_taken and not polaroid_prompt and not show_final_polaroid: #added check on showing polaroid
                    taking_photo = True
                    countdown_start_time = time.time()
                    countdown_text = "Taking photo in..."
                    print("Taking photo sequence started")

                # Button Hover / Pinch Gesture for Polaroid Prompt
                ih, iw, _ = frame.shape
                index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * iw), int(index_finger_tip.y * ih)

                # Hover detection
                if polaroid_prompt:
                    if (yes_button_pos[0] < x < yes_button_pos[0] + BUTTON_WIDTH and
                            yes_button_pos[1] < y < yes_button_pos[1] + BUTTON_HEIGHT):
                        yes_button_active = True
                    else:
                        yes_button_active = False

                    if (no_button_pos[0] < x < no_button_pos[0] + BUTTON_WIDTH and
                            no_button_pos[1] < y < yes_button_pos[1] + BUTTON_HEIGHT): #Fixed bug: was checking yes button pos
                        no_button_active = True
                    else:
                        no_button_active = False

                # Pinch detection (example implementation, adjust threshold)
                if yes_button_active and (distance(index_tip, thumb_tip) < 40):
                    print("Yes selected!")
                    polaroid_prompt = False
                    for i in range(2):  # Take 2 more photos
                        time.sleep(1)  # Short delay between photos
                        success, frame = cap.read() #added success check
                        if not success:
                            print("Ignoring empty camera frame during polaroid.")
                            continue
                        frame = cv2.flip(frame, 1)
                        frame = filter_functions[current_filter_index](frame)
                        photo_stack.append(frame)

                    # Create Polaroid Stack
                    final_polaroid_image = stack_polaroids(photo_stack[0], photo_stack[1], photo_stack[2])
                    save_image(final_polaroid_image)
                    photo_stack = []  # Clear the stack
                    show_final_polaroid = True #show final polaroid

                elif no_button_active and (distance(index_tip, thumb_tip) < 40):
                    print("No selected.")
                    polaroid_prompt = False
                    show_final_polaroid = False  # Ensure the final polaroid is not shown if No is pressed
                    photo_stack = []  # Clear the stack
                    # Reset the state so it can take more pictures
                    photo_taken = False  # Allow new photo to be taken
                    taking_photo = False


        # --- Apply Filter ---
        filtered_frame = filter_functions[current_filter_index](frame)

        # --- Countdown Timer ---
        if taking_photo:
            remaining_time = countdown_duration - (time.time() - countdown_start_time)
            if remaining_time > 0:
                countdown_text = f"Taking photo in... {int(remaining_time) + 1}"
            else:
                countdown_text = "Smile!"
                if not photo_taken:
                    taking_photo = False
                    photo = filtered_frame.copy() # Take photo from filtered frame
                    photo_stack.append(photo)
                    photo_taken = True
                    print("Photo taken!")

                    # After first photo, prompt for Polaroid
                    polaroid_prompt = True

        # --- UI Elements ---
        # Filter Buttons
        for i, filter_name in enumerate(filter_options):
            button_y = button_y_start + i * button_spacing
            color = BUTTON_COLOR if i != current_filter_index else (51, 153, 255) #Highlight selected button
            draw_button(frame, button_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT, filter_name, color, BUTTON_TEXT_COLOR)

        # Countdown Text
        cv2.putText(frame, countdown_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Polaroid Prompt
        if polaroid_prompt:
            overlay = frame.copy()
            alpha = 0.7  # Transparency factor.
            cv2.rectangle(overlay, (0, CAMERA_HEIGHT // 2 - 100), (CAMERA_WIDTH + BUTTON_WIDTH, CAMERA_HEIGHT // 2 + 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame) #Apply overlay

            cv2.putText(frame, "Do you want a Polaroid?", (CAMERA_WIDTH//2 - 150, CAMERA_HEIGHT // 2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            yes_color = (0,255,0) if yes_button_active else BUTTON_COLOR
            no_color = (0,0,255) if no_button_active else BUTTON_COLOR

            draw_button(frame, yes_button_pos[0], yes_button_pos[1], BUTTON_WIDTH, BUTTON_HEIGHT, "Yes", yes_color, BUTTON_TEXT_COLOR)
            draw_button(frame, no_button_pos[0], no_button_pos[1], BUTTON_WIDTH, BUTTON_HEIGHT, "No", no_color, BUTTON_TEXT_COLOR)

        # Show final Polaroid Image
        if show_final_polaroid and final_polaroid_image is not None:
            # Display the final polaroid
            resized_polaroid = cv2.resize(final_polaroid_image, (CAMERA_WIDTH, CAMERA_HEIGHT))
            cv2.imshow("Final Polaroid", resized_polaroid)

            # Reset states after showing the polaroid
            show_final_polaroid = False
            final_polaroid_image = None
            photo_taken = False # Allow new photo to be taken

        # --- Display ---
        cv2.imshow("Photo Booth", frame)

        # --- Exit Condition ---
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):
            save_image(filtered_frame)  # Save current frame for debugging/testing

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()