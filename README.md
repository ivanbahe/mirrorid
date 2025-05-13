# MirrorID 👁️👂👃  
**Sistema de reconocimiento facial que detecta rostros y características como ojos, nariz y orejas en tiempo real.**  

---

## 🔧 Requisitos  
- Python 3.8+  
- OpenCV  
- dlib  
- numpy  

---

## 🚀 Instalación rápida  
```bash
git clone https://github.com/ivanbahe/mirrorid.git  
cd mirrorid  
pip install -r requirements.txt  
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2  
bunzip2 shape_predictor_68_face_landmarks.dat.bz2  
mv shape_predictor_68_face_landmarks.dat models/

python mirrorid.py --video 0

python mirrorid.py --video ruta/al/video.mp4

from mirrorid import FaceDetector  

# Inicializar detector  
detector = FaceDetector()  

# Detectar rostros en un frame (ej. captura de OpenCV)  
results = detector.detect_faces(frame)  
