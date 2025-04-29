# Face Recognition System

Um sistema completo de detecção e reconhecimento facial em tempo real, construído do zero com Python, OpenCV e `facenet-pytorch`. Permite:

- Detecção de múltiplas faces em vídeo ao vivo (webcam).
- Captura automática de rolos de rostos para treinamento.
- Extração de embeddings com modelo pré-treinado FaceNet (InceptionResnetV1).
- Reconhecimento em tempo real comparando embeddings com um banco de templates.
- Classificação como “conhecido” ou “desconhecido” com limiar ajustável.

---

## 📁 Estrutura de Diretórios

```
face_recognition_system/
│
├── dataset/                  # Diretório gerado em runtime (não comitado)
│   └── joao/                 # Imagens de treino por pessoa
│
├── detectors/                # (Opcional) Seus módulos personalizados de detecção
├── embeddings/               # (Opcional) Seus módulos de extração/comparação
├── utils/                    # (Opcional) Funções auxiliares (logs, I/O, etc)
│
├── .gitignore                # Ignora venv, dataset, embeddings, caches
├── requirements.txt          # Dependências do projeto
├── capture_faces.py          # Captura cortes de rosto do frame original
├── detect_faces_mtcnn.py     # Teste de detecção em tempo real com MTCNN
├── extract_embeddings.py     # Extrai e salva embeddings em `embeddings.pt`
├── recognize_faces.py        # Reconhecimento em tempo real usando templates
└── README.md                 # Este arquivo
```

---

## ⚙️ Pré-requisitos

- Python 3.8+
- Git
- Webcam integrada ou USB
- (Opcional) GPU com CUDA para acelerar inferência

### Dependências

```bash
pip install -r requirements.txt
```

**`requirements.txt`** deve conter, por exemplo:

```text
opencv-python
torch
torchvision
facenet-pytorch
pillow
```

---

## 🚀 Guia de Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/face_recognition_system.git
   cd face_recognition_system
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\activate      # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure certificados (macOS):
   - Caso haja erro SSL ao baixar pesos, execute o script **Install Certificates.command** em `/Applications/Python 3.x/`.

---

## 🛠️ Uso

### 1. Testar detecção com MTCNN

```bash
python detect_faces_mtcnn.py
```

— Abre a webcam e desenha quadros verdes em todas as faces detectadas.

---

### 2. Capturar imagens de treinamento

```bash
python capture_faces.py
```

- Captura 20 cortes de rosto (160×160) da webcam e salva em `dataset/<nome>/`.  
- Pressione `ESC` para interromper antes de completar 20.

---

### 3. Extrair embeddings

```bash
python extract_embeddings.py
```

- Carrega imagens em `dataset/` e extrai embeddings 512-dim com FaceNet.  
- Salva dicionário de tensores em `embeddings.pt`.

---

### 4. Reconhecimento em tempo real

```bash
python recognize_faces.py
```

- Carrega `embeddings.pt`, calcula template médio por pessoa.  
- Detecta rosto, extrai embedding e compara com templates.  
- Exibe nome (verde) ou “Desconhecido” (vermelho) com a distância.

**Parâmetro de sensibilidade**  
Edite no topo de `recognize_faces.py`:

```python
threshold = 0.8  # 0.6 = mais estrito, 1.0 = mais tolerante
```

---

## 🔧 Como adicionar novas pessoas

1. Altere `person_name` em `capture_faces.py` (e crie `dataset/<nome>/`).
2. Rode o script para capturar rostos dessa nova identidade.  
3. Execute `extract_embeddings.py` novamente para atualizar `embeddings.pt`.  
4. Agora `recognize_faces.py` reconhecerá também esse novo nome.

---

## 💡 Boas práticas e dicas

- **Alinhamento**: utilize landmarks (olhos, nariz) para rotacionar e padronizar cada rosto antes de extrair embedding.  
- **Similaridade de cosseno**: troque distância Euclidiana por `F.cosine_similarity()` para mais robustez.  
- **Persistência de logs**: registre eventos de reconhecimento em arquivos CSV para análise posterior.  
- **Interface gráfica**: experimente Streamlit ou Flask para uma interface web amigável.

---

## 📄 Licença

Este projeto está licenciado sob a **MIT License**. Veja [LICENSE](LICENSE) para detalhes.

---

## 🤝 Contribuindo

1. Faça um fork deste repositório.  
2. Crie uma branch nova: `git checkout -b my-feature`.  
3. Commit suas alterações: `git commit -m "Add my feature"`.  
4. Push para branch: `git push origin my-feature`.  
5. Abra um Pull Request aqui no GitHub.

---

## 📫 Contato

João Victor Tavares Esteves  
👤 [joaovtesteves2002@gmail.com](mailto:joaovtesteves2002@gmail.com)  
🌐 [GitHub](https://github.com/jvtesteves)  

---
