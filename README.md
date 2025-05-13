# MirrorID 👁️👂👃

Sistema de reconocimiento facial que detecta rostros y características como ojos, nariz y orejas en tiempo real.

## 🔧 Requisitos
- Python 3.8+
- OpenCV
- dlib
- numpy

## 🚀 Instalación rápida
```bash
git clone https://github.com/ivanbahe/mirrorid.git
cd mirrorid
pip install -r requirements.txt
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat models/

▶️ Cómo usar
bash
# Usar con cámara web
python mirrorid.py --video 0

# Usar con archivo de video
python mirrorid.py --video ruta/al/video.mp4

📌 Ejemplo básico
python
from mirrorid import FaceDetector

detector = FaceDetector()
results = detector.detect_faces(frame)

for face in results:
    print(f"Puntos de la cara: {face.landmarks}")
    print(f"Ojos detectados: {face.eyes}")
