import cv2

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

    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Calcular la diferencia absoluta entre el primer frame y el frame actual
    frame_delta = cv2.absdiff(first_frame_gray, gray)
    
    # Aplicar un umbral a la diferencia para obtener las regiones en movimiento
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Encontrar contornos en la imagen umbralizada
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Mostrar el frame con las áreas en movimiento detectadas
    cv2.imshow('Motion Detection', frame)

    # Salir del bucle si se presiona la tecla 'p'
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
