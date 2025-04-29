import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from PIL import Image

# --- Configurações ---
device    = 'cuda' if torch.cuda.is_available() else 'cpu'
threshold = 0.8   # ajuste entre 0.6 (estrito) e 1.0 (tolerante) conforme necessário

# Modelos
mtcnn   = MTCNN(keep_all=True, device=device)
resnet  = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Pré-processamento para o ResNet
preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
])

# Carrega embeddings pré-computados
embeddings_dict = torch.load('embeddings.pt', map_location='cpu')

# Calcula o vetor médio (template) de cada pessoa
templates = {}
for name, embs in embeddings_dict.items():
    templates[name] = embs.mean(dim=0)
print(f"Templates carregados: {list(templates.keys())}")

# Função de distância Euclidiana
def euclidean(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).norm().item()

# Inicia captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detecta todas as faces no frame BGR
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]

            # Garanta coordenadas dentro dos limites do frame
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Verifica se a ROI é válida
            if x2 <= x1 or y2 <= y1:
                continue

            # Corta o rosto diretamente do frame original
            face_crop = frame[y1:y2, x1:x2]

            # Converte para PIL e aplica preprocessamento
            pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            img_tensor = preprocess(pil_img).unsqueeze(0).to(device)

            # Extrai embedding
            with torch.no_grad():
                emb = resnet(img_tensor).squeeze().cpu()

            # Calcula distâncias para cada template
            distances = {name: euclidean(emb, tpl) for name, tpl in templates.items()}
            name, dist = min(distances.items(), key=lambda x: x[1])

            # Escolhe label com base no threshold
            if dist < threshold:
                label, color = name, (0, 255, 0)
            else:
                label, color = "Desconhecido", (0, 0, 255)

            # Desenha retângulo e label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame,
                        f"{label} ({dist:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)

    # Exibe resultado
    cv2.imshow("Reconhecimento Facial", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
