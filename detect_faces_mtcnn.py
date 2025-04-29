import cv2
from facenet_pytorch import MTCNN
import torch

# Iniciar detector
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

# Abrir webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converter BGR para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar faces
    boxes, _ = mtcnn.detect(rgb_frame)

    # Desenhar as caixas
    if boxes is not None:
        for box in boxes:
            (x1, y1, x2, y2) = [int(b) for b in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Mostrar imagem
    cv2.imshow('MTCNN Face Detection', frame)

    # Pressione ESC para sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
