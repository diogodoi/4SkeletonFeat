# Documentação do Pipeline de Extração de Movimento (4SkeletonFeat / FitTube)

Este repositório contém o *framework* modular desenvolvido para extração, engenharia de características e análise visual de biomecânica. O objetivo do sistema é converter fluxos de vídeo brutos ou dados de sensores espaciais em múltiplos paradigmas de dados estruturados comprimidos (`.npz`), prontos para o treinamento de modelos de Machine Learning e Deep Learning.

---

## 🏗️ Arquitetura do Sistema

```text
📂 4SkeletonFeat
│
├── 📄 main.py                       # Orquestrador principal do pipeline
├── 📄 app.py                        # Dashboard Web Multimodal (Streamlit)
├── 📄 topology_registry.py          # Registo Central de Topologias (MediaPipe, NTU, COCO)
├── 📂 extractors
    ├── 📄 ntu_ingester.py               # Exemplo de Tradutor de Dados Externos (Kinect)
    ├── 📄 media_pipe_extractor.py       # Extração de pontos brutos a partir de vídeo
    ├── 📄 bone_extractor.py             # Cálculo de vetores anatómicos
    ├── 📄 joint_motion_extractor.py     # Cálculo de diferenciais de articulações
    ├── 📄 bone_motion_extractor.py      # Cálculo de diferenciais de ossos
    ├── 📄 pose_tokenizer.py             # Quantizador vetorial (Tokens de esqueleto)
    ├── 📄 skeleton_image_extractor.py   # Gerador de matrizes espaço-temporais para CNNs
    ├── 📄 graph_extractor.py            # Modelagem topológica e estrutural para GCNs

```

---

## 📦 1. Preparação dos Dados (Como utilizar o Framework)

Uma das maiores inovações deste *framework* é ser **totalmente agnóstico à fonte dos dados**. Ele não está preso ao MediaPipe. Pode processar dados do YOLO-Pose, OpenPose, Kinect (NTU RGB+D), Vicon ou qualquer outro sistema de captura de movimento.

Para utilizar os seus próprios dados neste ecossistema, basta seguir três regras de ouro:

### 1.1 O Formato Universal (`.npz`)

Independentemente da origem, os dados em bruto das poses devem ser convertidos e guardados na pasta `/joints` como um ficheiro NumPy Comprimido (`.npz`). A matriz no interior deve, obrigatoriamente, ter a estrutura tridimensional:

* **Shape**: `(Frames, Número_de_Nós, 4)`
* **Eixo 0 (Frames)**: A dimensão temporal do vídeo.
* **Eixo 1 (Nós)**: As articulações do esqueleto (ex: 33 para MediaPipe, 25 para NTU, 17 para COCO).
* **Eixo 2 (Atributos)**: Exatamente 4 valores por nó, na ordem `[X, Y, Z, Visibilidade]`.
* *Nota:* A Visibilidade (Confiança) deve ser um valor normalizado entre `0.0` e `1.0`. Se o seu sensor não fornecer a visibilidade, preencha esta coluna com `1.0`.



### 1.2 O Registo de Topologia

O *framework* precisa de saber "como os ossos se ligam" para calcular matrizes de adjacência (Grafos) e vetores cinemáticos (Ossos).
Antes de correr o `main.py`, abra o arquivo **`topology_registry.py`** e garanta que a anatomia do seu modelo está mapeada.

Exemplo para um modelo customizado de 5 pontos:

```python
TOPOLOGIES = {
    'meu_modelo': {
        'num_nodes': 5,
        'edges': [(0, 1), (1, 2), (2, 3), (3, 4)]
    }
}

```

### 1.3 Criar um "Ingestor" (Para Datasets Externos)

Se já possui um dataset descarregado da internet (como o *NTU RGB+D* em arquivos `.skeleton` de texto, ou *COCO* em `.json`), não precisa de reescrever o pipeline de IA.
Basta criar uma classe "Ingestor" (veja o exemplo `ntu_ingester.py`) que lê esses arquivos de texto, constrói a matriz NumPy `(Frames, Nós, 4)` e os guarda na pasta `/joints`. A partir daí, o resto do *framework* assume o controlo automaticamente.

---

## 🖥️ 2. Guia do Dashboard de Visualização Multimodal

Para facilitar a análise qualitativa e a inspeção de erros no dataset, o *framework* inclui um painel Web interativo desenvolvido em **Streamlit**.

### 2.1 Instalação e Execução

Certifique-se de que possui as bibliotecas gráficas instaladas no seu ambiente virtual:

```bash
pip install streamlit plotly networkx matplotlib opencv-python

```

Para iniciar a interface de visualização, abra o terminal na pasta do projeto e execute:

```bash
streamlit run app.py

```

O seu navegador predefinido abrirá automaticamente em `http://localhost:8501`.

### 2.2 Como Operar a Interface

#### Barra Lateral (Painel de Controlo)

1. **Topologia:** Selecione o padrão de esqueleto que os dados atuais utilizam (ex: `mediapipe`, `ntu`). O Dashboard autoajusta-se a esta topologia para não gerar erros de leitura.
2. **Seleção de Arquivos:** O painel detecta automaticamente os vídeos processados. Se desejar, pode clicar em *Browse/Procurar* para mapear uma nova pasta com arquivos de vídeo (`.mp4`) caso queira a sobreposição em tempo real.
3. **Motor de Visualização:** Alterne livremente entre os 5 paradigmas de Machine Learning.

#### Player de Animação Sincronizada

No topo da área principal, encontra o **Controlador de Tempo**. Pode arrastar o *slider* com o mouse para saltar para momentos específicos do vídeo, ou clicar no botão **▶️ Reproduzir** para que a interface anime automaticamente. O sistema de estado dinâmico garante que todas as abas estejam perfeitamente sincronizadas com o *frame* atual.

### 2.3 Paradigmas de Visualização Disponíveis

1. **Pose 3D (Estática/Animada):** Utiliza o *Plotly* para renderizar o esqueleto num plano tridimensional. Totalmente interativo (permite *zoom*, rotação e identificação das coordenadas exatas ao passar o mouse). Excelente para investigar anomalias de profundidade (Eixo Z).
2. **Movimento de Junções (Vídeo 2D):** Sobrepõe vetores de velocidade direcional sobre o vídeo RGB original usando OpenCV. Setas vermelhas disparam a partir das articulações indicando intensidade e direção do deslocamento temporal.
3. **Movimento de Ossos (Gráfico Dinâmico):** Exibe um osciloscópio de velocidades combinadas. Conta com um seletor múltiplo (`multiselect`) permitindo comparar matematicamente a dinâmica entre múltiplos membros (ex: cruzar os dados do Braço Esquerdo contra o Braço Direito em simultâneo).
4. **Topologia de Grafos (NetworkX):** Renderiza a matriz de adjacência usada por modelos *GCN*. Os nós são coloridos dinamicamente do Vermelho (oculto/erro) ao Verde (visível) com base na margem de confiança, permitindo avaliar visualmente a qualidade do *tracking* da pose no momento atual.
5. **Sequência de Tokens:** Exibe o mapeamento no formato de *Linguagem Corporal* (*NLP Transformers*). Um gráfico de série temporal onde uma linha vermelha percorre as "palavras de postura" (IDs Inteiros) identificadas pelo quantizador K-Means.


# Documentação do Pipeline de Extração de Movimento (FitTube Dataset)

O pipeline é composto por 7 etapas independentes e complementares:

1. **Pose Bruta**: Coordenadas tridimensionais absolutas das articulações.
2. **Vetores de Ossos (Bones)**: Segmentos corporais invariantes à translação.
3. **Joint Motion**: Velocidade e dinâmica temporal das articulações.
4. **Bone Motion**: Velocidade e dinâmica temporal dos segmentos de ossos.
5. **Pose Tokens**: Discretização quantizada (linguagem corporal) para Transformers.
6. **Skeleton-to-Image**: Representação espaço-temporal em matrizes 2D para CNNs.
7. **Skeleton-to-Graph**: Modelagem topológica de esqueleto para redes de grafos (GCNs).

Para ter acesso ao dataset preencha o seguinte formulário: https://forms.gle/vzC35kwWLqeHJrJC7
---
---

## 📊 Detalhes Técnicos dos Novos Componentes

### 3. `JointMotionExtractor` & `BoneMotionExtractor` (Dinâmica Temporal)

* **Conceito**: Calculam a diferença de posição entre frames consecutivos ($t$ e $t-1$). Capturam a velocidade e aceleração angular das articulações e ossos, fornecendo a assinatura dinâmica do exercício.
* **Formatos de Saída**: `(Frames - 1, 33, 4)` para Junções e `(Frames - 1, 12, 4)` para Ossos.

### 4. `PoseTokenizer` (Linguagem Corporal Discreta)

* **Conceito**: Utiliza algoritmos como `KMeans` ou `MiniBatchKMeans` para criar um dicionário (*Codebook*) de 512 poses universais coletadas de forma estratificada e equilibrada entre as classes de exercício. Converte matrizes contínuas pesadas em sequências simples de IDs inteiros.
* **Formato de Saída**: Um vetor 1D de inteiros `(Frames,)`.

### 5. `SkeletonImageExtractor` (Matrizes Espaço-Temporais)

* **Conceito**: Achata as dimensões espaciais do esqueleto e empilha o tempo no eixo vertical. O vídeo vira uma imagem 2D onde linhas são frames e colunas são características, permitindo o uso de texturas por CNNs.
* **Formato de Saída**: Uma matriz 2D `(Frames, Características_Achatadas)`.

### 6. `GraphExtractor` (Topologia Corpórea)

* **Conceito**: Cria um grafo não-direcionado que explica matematicamente a conectividade biológica do corpo através de uma Matriz de Adjacência com conexões reflexivas (*self-loops*).
* **Formato de Saída**: Salva os Atributos dos Nós `x`, a matriz `adjacency` e a lista de arestas em formato COO `edge_index`.

---

## 🧠 Guia de Implementação no Machine Learning

Abaixo estão os templates de código demonstrando como carregar e alimentar modelos de IA utilizando cada um dos formatos gerados pelo pipeline:

### 1. Consumindo os Dados Contínuos (Poses, Bones e Motions)

Ideal para redes neurais sequenciais clássicas como **LSTMs, GRUs e RNNs**.

```python
import numpy as np
import torch
import torch.nn as nn

# Carregamento do arquivo de ossos ou poses
data = np.load("data/processed/bones/video_exemplo.npz")
X_sequence = data['bones']  # Shape: (Frames, 12, 4)

# Preparação para PyTorch (Adicionando dimensão de Batch)
# O formato esperado por camadas RNN no PyTorch geralmente é (Batch, Sequence_Length, Features)
tensor_in = torch.tensor(X_sequence, dtype=torch.float32).unsqueeze(0) 
flat_tensor_in = tensor_in.view(tensor_in.size(0), tensor_in.size(1), -1) # Shape: (1, Frames, 48)

# Exemplo de arquitetura de modelo
lstm_layer = nn.LSTM(input_size=48, hidden_size=64, batch_first=True)
output, (hn, cn) = lstm_layer(flat_tensor_in)

```

### 2. Consumindo Pose Tokens

Ideal para modelos de Processamento de Linguagem Natural baseados em **Transformers (BERT, GPT) ou camadas de Embedding**.

```python
import numpy as np
import torch
import torch.nn as nn

# Carrega os tokens discretos
data = np.load("data/processed/tokens/video_exemplo.npz")
tokens = data['tokens']  # Shape: (Frames,) -> Ex: [42, 42, 107, 312, ...]

# Conversão para Tensor de inteiros compativeis com NLP
tensor_tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0) # Shape: (1, Frames)

# Implementação na camada de incorporação (Embedding)
# Transforma IDs inteiros (0-511) em representações vetoriais densas de 128 dimensões
embedding_layer = nn.Embedding(num_embeddings=512, embedding_dim=128)
embedded_sequence = embedding_layer(tensor_tokens) # Shape: (1, Frames, 128)

```

### 3. Consumindo Imagens Espaço-Temporais

Ideal para **Redes Neurais Convolucionais 2D (CNNs)** de imagem tradicional, como ResNet, MobileNet ou EfficientNet.

```python
import numpy as np
import torch
import torch.nn as nn

# Carrega a imagem do movimento
data = np.load("data/processed/skeleton_images/video_exemplo.npz")
motion_image = data['motion_image']  # Shape: (Frames, Características)

# Transforma em tensor adicionando a dimensão de canais (Channel = 1, escala de cinza)
# Formato padrão para Conv2D: (Batch, Channels, Height, Width)
tensor_img = torch.tensor(motion_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) 

# Exemplo de Convolução 2D varrendo o tempo e o espaço simultaneamente
conv_layer = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
feature_maps = conv_layer(tensor_img)

```

### 4. Consumindo Estruturas de Grafos

Ideal para **Redes de Convolução em Grafos (GCN / ST-GCN)** implementadas via **PyTorch Geometric (PyG)**.

```python
import numpy as np
import torch
from torch_geometric.data import Data

# Carrega o grafo do esqueleto
data_graph = np.load("data/processed/graphs/video_exemplo.npz")

x_nodes = data_graph['x']          # Atributos dos nós. Shape: (Frames, 33, 4)
edge_index = data_graph['edge_index']  # Lista de arestas em formato COO. Shape: (2, Num_Arestas)

# Converte para Tensores nativos
x_tensor = torch.tensor(x_nodes, dtype=torch.float32)
edge_tensor = torch.tensor(edge_index, dtype=torch.long)

# Monta o objeto Data do PyTorch Geometric
# Nota científica: Para ST-GCN, a dimensão de frames em 'x' é processada iterativamente pela rede
graph_data = Data(x=x_tensor, edge_index=edge_tensor)

```

---

## 🚀 Como Executar o Pipeline Completo

Configure as pastas de destino desejadas no script `main.py` e execute o comando:

```bash
python main.py

```

O script gerenciará automaticamente o fluxo sequencial de dependências, garantindo que as características estatísticas, temporais, discretas, convolucionais e topológicas sejam geradas sem colisões de cache ou erros de memória.
