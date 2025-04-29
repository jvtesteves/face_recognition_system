import os
import cv2
import torch
from facenet_pytorch import MTCNN

# Parâmetros do usuário
person_name = "joao"
save_dir    = f"dataset/{person_name}"
max_imgs    = 20

# Cria pasta se não existir
os.makedirs(save_dir, exist_ok=True)

# Inicializa o detector MTCNN
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn  = MTCNN(keep_all=True, device=device)

# Abre a câmera
cap = cv2.VideoCapture(0)
count = 0

while count < max_imgs:
    ret, frame = cap.read()
    if not ret:
        break

    # Detecta as faces no frame BGR
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None and len(boxes):
        # Pega a primeira face detectada
        x1, y1, x2, y2 = boxes[0].astype(int)

        # Corta a região da face direto do frame original
        face_crop = frame[y1:y2, x1:x2]

        # (Opcional) Normaliza o tamanho para 160×160, padrão FaceNet
        face_resized = cv2.resize(face_crop, (160, 160))

        # Salva a imagem
        filename = os.path.join(save_dir, f"{count}.jpg")
        cv2.imwrite(filename, face_resized)
        print(f"[✔] Rosto {count} salvo em {filename}")
        count += 1

    # Exibe o vídeo com a detecção marcada (só para referência)
    # Desenha retângulo verde na primeira face
    if boxes is not None and len(boxes):
        x1, y1, x2, y2 = boxes[0].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Capturando rostos", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
