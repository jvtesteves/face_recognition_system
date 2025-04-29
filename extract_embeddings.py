import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from PIL import Image

# --- Configurações ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Modelo FaceNet pré-treinado (vggface2)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Diretório com subpastas de pessoas (ex: dataset/joao)
dataset_dir = 'dataset'

# Transformações: resize → tensor [0,1] → normalize [-1,1]
preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# Dicionário para armazenar embeddings
embeddings_dict = {}

# --- Pipeline de extração ---
for person_name in os.listdir(dataset_dir):
    person_folder = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_folder):
        continue

    embeds = []
    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)
        # Carrega e preprocura
        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        # Extrai embedding
        with torch.no_grad():
            emb = model(img_tensor)                # shape [1, 512]
        embeds.append(emb.squeeze().cpu())         # shape [512]

    # Empilha todos os embeddings dessa pessoa: shape [n_imgs, 512]
    embeddings_stack = torch.stack(embeds)
    embeddings_dict[person_name] = embeddings_stack
    print(f"→ extraídos {embeddings_stack.shape[0]} embeddings para “{person_name}”")

# --- Salva tudo no disco ---
save_path = 'embeddings.pt'
torch.save(embeddings_dict, save_path)
print(f"\n✅ Embeddings salvos em: {save_path}")
