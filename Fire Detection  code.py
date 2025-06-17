import cv2         # Library for OpenCV
import threading   # Library for threading (run code in background)
import playsound   # Library for playing alarm sound
import smtplib     # Library for email sending
import numpy as np  # Library for numerical operations
import matplotlib.pyplot as plt  # Library for plotting graphs

# Load Haar Cascade fire detection model
fire_cascade = cv2.CascadeClassifier('C:/Users/user/Downloads/fire_detection_cascade_model (1).xml') # To access xml file which includes positive and negative images of fire. (Trained images)
                                                                                                     # File is also provided with the code.
# Open camera (0 for built-in webcam, 1 for USB camera)
vid = cv2.VideoCapture(0)
runOnce = False  # Boolean flag to prevent multiple email alerts
fire_detections = []  # Store fire detection count
frame_count = 0  # Frame counter

def play_alarm_sound_function():
    """Plays alarm sound when fire is detected."""
    playsound.playsound('C:/Users/user/Downloads/Fire alram sound.mp3', True) # to play alarm # mp3 audio file is also provided with the code.
    print("⏰ Fire alarm end! ⏰") # to print in console

def send_mail_function(): # defined function to send mail post fire detection using threading
    """Sends an email alert when fire is detected."""
    recipientmail = "adeshjainhp@gmail.com" # recipients mail
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login("adeshjainhp@gmail.com", 'gfhb upno ffbj cpuo') # Senders mail ID and password
        server.sendmail('adeshjainhp@gmail.com', recipientmail, "Warning fire accident has been reported") # recipients mail with mail message
        print(f"Alert mail sent successfully to {recipientmail}")  # to print in consol to whome mail is sent
        server.close()
    except Exception as e:
        print(f"Error sending email: {e}")

def calculate_confidence(frame, x, y, w, h):
    """Calculates fire confidence score based on fire-like colors in the detected region."""
    roi = frame[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    lower_fire = np.array([0, 50, 50])  # Lower bound for fire colors
    upper_fire = np.array([35, 255, 255])  # Upper bound for fire colors

    mask = cv2.inRange(hsv, lower_fire, upper_fire)  # Create mask for fire colors
    fire_pixels = cv2.countNonZero(mask)
    total_pixels = w * h
    return (fire_pixels / total_pixels) * 100  # Confidence score in percentage

while True:
    ret, frame = vid.read()  # Read video frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    fire = fire_cascade.detectMultiScale(frame, 1.2, 5)  # Detect fire
    frame_count += 1
    fire_detected = 0

    for (x, y, w, h) in fire:
        confidence = calculate_confidence(frame, x, y, w, h)

        if confidence > 40:  # If confidence is above 40%, consider fire detected
            cv2.rectangle(frame, (x-20, y-20), (x+w+20, y+h+20), (255, 0, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            print(f"🔥 Fire detected! Confidence: {confidence:.2f}% 🔥")
            fire_detected += 1
            threading.Thread(target=play_alarm_sound_function).start()  # Play alarm
            
            if not runOnce:
                print("📧 Sending fire alert email...")
                threading.Thread(target=send_mail_function).start()
                runOnce = True
    
    fire_detections.append(fire_detected)
    cv2.imshow('Fire Detection System', frame)  # Display camera feed
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

vid.release()
cv2.destroyAllWindows()

# Plot separate graphs after detection
plt.figure()
plt.plot(range(len(fire_detections)), fire_detections, marker='o', linestyle='-')
plt.title('Fire Detections Over Time')
plt.xlabel('Frame Count')
plt.ylabel('Number of Fires Detected')
plt.show()

plt.figure()
plt.hist(fire_detections, bins=5, color='red', alpha=0.7)
plt.title('Fire Detection Frequency')
plt.xlabel('Number of Fires Detected')
plt.ylabel('Frequency')
plt.show()

plt.figure()
window_size = 5
moving_avg = np.convolve(fire_detections, np.ones(window_size)/window_size, mode='valid')
plt.plot(range(len(moving_avg)), moving_avg, marker='^', linestyle='-', color='blue')
plt.title('Moving Average of Fire Detections')
plt.xlabel('Frame Count')
plt.ylabel('Average Detections')
plt.show()