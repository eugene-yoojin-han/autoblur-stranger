import face_recognition
import cv2
import numpy as np

vid = cv2.VideoCapture(0)

target_image = face_recognition.load_image_file("target.jpeg")
target_face_encoding = face_recognition.face_encodings(target_image)[0]

known_face_encodings = [
    target_face_encoding
]
known_face_names = [
    "Target"
]

face_locations = []
face_encodings = []
face_names = []
process = True

frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while True:
    ret, frame = vid.read()
    if process:
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
    process = not process
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        if name != "Target":
            center_x = left+(right-left)//2
            center_y = top+(bottom-top)//2
            temp_img = frame.copy()
            mask_shape = (frame.shape[0], frame.shape[1], 1)
            mask = np.full(mask_shape, 0, dtype=np.uint8)
            temp_img[top:bottom, left:right] = cv2.blur(temp_img[top:bottom, left:right], (99, 99))
            cv2.circle(mask, (center_x, center_y), 1, (255, 255, 255), 500)
            inverted_mask = cv2.bitwise_not(mask)
            background = cv2.bitwise_and(frame, frame, mask=inverted_mask)
            foreground = cv2.bitwise_and(temp_img, temp_img, mask=mask)
            combined = cv2.add(background, foreground)
        else:
            combined = frame
    out.write(combined)
    cv2.imshow('Video', combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
out.release()
cv2.destroyAllWindows()

