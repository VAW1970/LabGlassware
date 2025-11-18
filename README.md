# 🔬 LabGlassware: Detecção de Objetos em Tempo Real

Este projeto apresenta um aplicativo Streamlit para a **Detecção de Objetos de Vidraria e Equipamentos de Laboratório** (Lab Glassware Detection) utilizando o modelo de Deep Learning **Faster R-CNN com ResNet-50 FPN** e a biblioteca **PyTorch**.

## ✨ Funcionalidades

* **Upload de Imagem:** Permite que o usuário faça o upload de uma imagem contendo vidraria de laboratório (Béqueres, Erlenmeyers, Tubos de Ensaio, etc.).
* **Detecção de Objetos:** Aplica o modelo treinado para identificar a localização exata e a classe dos objetos.
* **Visualização:** Exibe a imagem processada com **bounding boxes** e rótulos de classe/confiança.
* **Controle de Confiança:** Permite ajustar o limite (*threshold*) de confiança para filtrar as detecções.
