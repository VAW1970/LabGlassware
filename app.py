import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import io

# ==============================================================================
# 🧪 CONFIGURAÇÕES
# ==============================================================================
# O Streamlit Cloud geralmente roda em CPU. Forçando o fallback para evitar erros.
DEVICE = torch.device('cpu') 
PATH_MODELO = "IA/Glassware/Github/modelo_labglassware.pth"  # Arquivo deve estar no diretório

# Classes do modelo (O background é a classe 0)
CLASSES = [
    "background",
    "beaker",
    "compass",
    "digital_balance",
    "erlenmeyer_flask",
    "funnel",
    "graduated_cylinder",
    "horseshoe_magnet",
    "objects",
    "stirring_rod",
    "test_tube",
    "test_tube_rack",
    "thermometer",
]
NUM_CLASSES = len(CLASSES)

# ==============================================================================
# 🧠 CARREGAMENTO E CACHE DO MODELO (Otimização Streamlit)
# ==============================================================================
@st.cache_resource
def load_detection_model():
    """Carrega o modelo de detecção treinado uma única vez."""
    st.info("Carregando o modelo... Por favor, aguarde.")
    
    # Cria a arquitetura do modelo
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=NUM_CLASSES)
    
    # Carrega os pesos (força o carregamento para CPU para compatibilidade com Streamlit Cloud)
    try:
        model.load_state_dict(torch.load(PATH_MODELO, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        st.success("Modelo de Vidraria Carregado com Sucesso!")
        return model
    except FileNotFoundError:
        st.error(f"Erro: O arquivo de modelo '{PATH_MODELO}' não foi encontrado. Certifique-se de que ele está na pasta do aplicativo.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None


# ==============================================================================
# 🎨 FUNÇÃO PRINCIPAL DE DETECÇÃO E VISUALIZAÇÃO
# ==============================================================================
def detect_and_draw(model, pil_image, threshold):
    """Aplica a detecção no PIL Image e desenha as bounding boxes."""
    
    # 1. Transformar a imagem para Tensor
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(pil_image).to(DEVICE)

    # 2. Fazer a predição
    with torch.no_grad():
        outputs = model([img_tensor])

    # 3. Processar resultados
    output = outputs[0]
    boxes = output['boxes']
    labels = output['labels']
    scores = output['scores']

    # 4. Desenhar Bounding Boxes usando Matplotlib
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(pil_image)
    
    detected_items = []

    # Filtrar e desenhar
    indices = [i for i, s in enumerate(scores) if s > threshold]

    if not indices:
        ax.text(50, 50, "Nenhuma detecção acima do limite.", color='red', fontsize=16, backgroundcolor='white')

    for i in indices:
        box = boxes[i].cpu().numpy()
        x1, y1, x2, y2 = box
        class_id = labels[i].item()
        classe_nome = CLASSES[class_id]
        conf = scores[i].item()

        # Desenhar retângulo
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, color='lime', linewidth=2))
        
        # Adicionar texto
        ax.text(x1, y1 - 5, f"{classe_nome} ({conf:.2f})",
                color='yellow', fontsize=12, backgroundcolor='black')
        
        detected_items.append({"Classe": classe_nome, "Confiança": f"{conf:.2f}"})

    plt.axis("off")
    return fig, detected_items


# ==============================================================================
# 🖥️ INTERFACE STREAMLIT
# ==============================================================================

# 1. Configuração da Página
st.set_page_config(
    page_title="Vidraria Lab - Detecção de Objetos",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Detecção de Vidraria de Laboratório")
st.markdown("Faça o upload de uma imagem para aplicar o modelo de reconhecimento (Faster R-CNN).")
st.markdown("Este modelo de detecção foi treinado com um **dataset público** no ambiente Google Colaboratory, aproveitando a aceleração de **GPU (CGU)**, porém dentro das limitações de tempo e recursos do ambiente gratuito.")
st.markdown("---")


# 2. Carregar o modelo
modelo = load_detection_model()

if modelo is None:
    st.stop() # Parar o aplicativo se o modelo não puder ser carregado

# 3. Sidebar para Controles
st.sidebar.header("⚙️ Controles de Detecção")
st.image("Taleh azul 3D ícone.png", width=128)

threshold = st.sidebar.slider(
    'Limite de Confiança (Threshold)',
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05,
    help="Apenas detecções com confiança acima deste valor serão exibidas."
)

# 4. Upload de Imagem
uploaded_file = st.file_uploader(
    "🖼️ **Selecione uma Imagem (.jpg, .png)**",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    # Ler a imagem e converter para o formato PIL
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)

    with col1:
        st.header("Imagem Original")
        st.image(image, caption='Imagem de Entrada', use_column_width=True)

    with col2:
        st.header("Resultado da Detecção")
        
        # 5. Aplicar e Exibir Resultados
        with st.spinner('Processando imagem e detectando objetos...'):
            fig_result, detections = detect_and_draw(modelo, image, threshold)
            
            # Exibe o resultado do Matplotlib no Streamlit
            st.pyplot(fig_result, use_container_width=True)
            
            st.subheader("Itens Detectados")
            if detections:
                st.dataframe(detections, use_container_width=True)
            else:
                st.warning("Nenhum objeto de vidraria detectado com a confiança atual.")
