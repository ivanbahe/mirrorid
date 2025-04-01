import cv2
import os

# Ruta donde se guardaron los clasificadores adicionales
custom_cascade_path = r'C:\Users\34640\OneDrive - Stucom, S.A\Documentos\Reconocimiento_facial\codigo'

# Cargar los clasificadores preentrenados para la detección de caras (frontal)
face_cascade_default = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
face_cascade_alt_tree = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt_tree.xml')

# Cargar los clasificadores preentrenados para la detección de caras (perfil)
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Cargar los clasificadores preentrenados para la detección de partes específicas del rostro
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') # ojos
eyeglasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml') # ojos con gafas
left_ear_cascade = cv2.CascadeClassifier(os.path.join(custom_cascade_path, 'haarcascade_mcs_leftear.xml')) # oreja izquierda
right_ear_cascade = cv2.CascadeClassifier(os.path.join(custom_cascade_path, 'haarcascade_mcs_rightear.xml')) # oreja derecha
nose_cascade = cv2.CascadeClassifier(os.path.join(custom_cascade_path, 'haarcascade_mcs_nose.xml')) #nariz
mouth_cascade = cv2.CascadeClassifier(os.path.join(custom_cascade_path, 'haarcascade_mcs_mouth.xml')) #boca

# Comprobar si los clasificadores se han cargado correctamente
if (face_cascade_default.empty() or face_cascade_alt.empty() or face_cascade_alt_tree.empty() or
    profile_cascade.empty() or eye_cascade.empty() or eyeglasses_cascade.empty() or left_ear_cascade.empty() or
    right_ear_cascade.empty() or nose_cascade.empty() or mouth_cascade.empty()):
    print("Error al cargar uno o más clasificadores")
    exit()

# Capturar video desde la cámara (usualmente 0 es el índice de la cámara por defecto)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Inicializar el primer frame para la detección de movimiento
ret, first_frame = cap.read()
if not ret:
    print("Error: No se pudo leer el frame inicial.")
    cap.release()
    exit()

first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_frame_gray = cv2.GaussianBlur(first_frame_gray, (21, 21), 0)

while True:
    # Leer un frame del video
    ret, frame = cap.read()

    if not ret:
        print("Error: No se pudo leer el frame.")
        break

    # Convertir el frame a escala de grises (necesario para la detección de Haar cascades)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    # Detectar movimiento
    frame_delta = cv2.absdiff(first_frame_gray, gray_blurred)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Detectar caras y perfiles en el frame usando múltiples clasificadores
    all_faces = []
    for face_cascade in [face_cascade_default, face_cascade_alt, face_cascade_alt_tree]:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        all_faces.extend(faces)
    
    profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    eyeglasses = eyeglasses_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    left_ears = left_ear_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    right_ears = right_ear_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    nose = nose_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    mouth = mouth_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar un rectángulo alrededor de cada cara detectada
    for (x, y, w, h) in all_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Dibujar un rectángulo alrededor de cada perfil detectado
    for (x, y, w, h) in profiles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Dibujar un rectángulo alrededor de cada parte específica detectada
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    for (x, y, w, h) in eyeglasses:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    for (x, y, w, h) in left_ears:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

    for (x, y, w, h) in right_ears:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

    for (x, y, w, h) in nose:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    for (x, y, w, h) in mouth:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Mostrar el frame con las caras, perfiles, partes específicas y movimiento detectados
    cv2.imshow('Face, Profile, Specific, and Motion Detection', frame)

    # Salir del bucle si se presiona la tecla 'p'
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
