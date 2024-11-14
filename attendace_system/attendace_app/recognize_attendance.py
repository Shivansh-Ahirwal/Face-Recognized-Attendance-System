# recognize_attendance.py
import face_recognition
import cv2
import pickle
from .models import Student, Attendance
from django.utils import timezone
import numpy as np

def mark_attendance(student_name):
    try:
        student = Student.objects.get(full_name=student_name)
        attendance, created = Attendance.objects.get_or_create(
            student=student,
            timestamp__date=timezone.now().date(),
            defaults={'status': 'Present'}
        )
        if created:
            print(f"{student_name}'s attendance marked.")
    except Student.DoesNotExist:
        print(f"Student {student_name} not found.")

# Load known faces
known_face_encodings = []
known_face_names = []

with open("known_faces.dat", "rb") as f:
    while True:
        try:
            name, encoding = pickle.load(f)
            known_face_names.append(name)
            known_face_encodings.append(encoding)
        except EOFError:
            break

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Start processing video for attendance
while True:
    ret, frame = video_capture.read()
    
    if not ret:
        print("Failed to grab video frame.")
        break
    
    # Resize the frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    # Mark attendance for recognized students
    for name in face_names:
        if name != "Unknown":
            mark_attendance(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Label the face
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Show the video frame
    cv2.imshow('Attendance', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close window
video_capture.release()
cv2.destroyAllWindows()
