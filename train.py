# Liveness detector - https://github.com/alexcamargoweb/liveness-detection.
# Prova de vida através de biometria facial utilizando OpenCV.
# Adrian Rosebrock, Liveness Detection with OpenCV. PyImageSearch.
# Disponível em: https://www.pyimagesearch.com/2019/03/11/liveness-detection-with-opencv/.
# Acessado em: 23/03/2021.
# Arquivo: train.py
# Execução via PyCharm/Linux (Python 3.8)
# $ conda activate tensorflow_keras

# importa os pacotes necessários
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from imutils import paths
import matplotlib

matplotlib.use("Agg")  # salvar figuras em background
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os


# cria a CNN LivenessNet
def LivenessNet(width, height, depth, classes):
    # inicializa o modelo junto com o input shape "channels last"
    # e a própria dimensão dos canais
    model = Sequential()  # instancia o modelo
    inputShape = (height, width, depth)
    chanDim = -1
    # se está sendo utilizado "channels first",
    # atualiza o input shape e a dimensão dos canais
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    # DEFINIÇÃO DA CNN

    # 1 CONJUNTO DE CAMADAS: CONV => RELU => CONV => RELU => POOL
    model.add(Conv2D(16, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(16, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 2 CONJUNTO DE CAMADAS: CONV => RELU => CONV => RELU => POOL
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # CONJUNTO DE CAMADAS DENSAS: FC => RELU
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # classificador softmax
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # retorna a arquitetura da rede construida
    return model


# PARÂMETROS DE TREINAMENTO

# dataset dos exemplos gerados
DATASET = './dataset'
# output do modelo a ser treinado
MODEL = './model/liveness.model'
# output do label encoder a ser gerado
ENCODER = './encoder/le.pickle'
# gráfico de treinamento
PLOT = './plot/train.png'
# taxa de aprendizado inicial
INIT_LR = 1e-4
# tamanho do lote (batch size)
BS = 8
# épocas de treinamento
EPOCHS = 50

print("[INFO] carregando imagens...")
# pega a lista de imagens do dataset criado
imagePaths = list(paths.list_images(DATASET))
data = []  # imagens
labels = []  # rótulos

# percorre todas as imagens do diretório
for imagePath in imagePaths:
    # extrai o rótulo com base na localização do arquivo
    label = imagePath.split(os.path.sep)[-2]
	# armazena a imagem
    image = cv2.imread(imagePath)
    # redimensiona a imagem para 32x32 pixels, ignorando a proporção
    image = cv2.resize(image, (32, 32))
    # atualiza as listas de imagens e rótulos, respectivamente
    data.append(image)
    labels.append(label)

# dimensiona os pixels da imagem de entrada para o intervalo [0, 1],
# utilizando um numpy array
data = np.array(data, dtype = "float") / 255.0

# codifica os rótulos (que atualmente são strings) como inteiros
le = LabelEncoder()
labels = le.fit_transform(labels)
# realiza o one-hot encoding
labels = to_categorical(labels, 2)

# particiona os dados em 75% para treinamento e 25% para teste
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size = 0.25,
                                                  random_state = 777)

# constrói o gerador de imagens de treinamento para o
# data augmentation (aumento de dados)
aug = ImageDataGenerator(rotation_range = 20,
                         zoom_range = 0.15,
                         width_shift_range = 0.2,
                         height_shift_range = 0.2,
                         shear_range = 0.15,
                         horizontal_flip = True,
                         fill_mode = "nearest")

print("[INFO] compilando o modelo...")
# inicializa o otimizador e o modelo
opt = Adam(lr = INIT_LR, decay = INIT_LR / EPOCHS)
model = LivenessNet(width = 32, height = 32,
                    depth = 3, classes = len(le.classes_))
model.compile(loss = "binary_crossentropy",
              optimizer = opt, metrics = ["accuracy"])

print("[INFO] treinando a rede para {} épocas...".format(EPOCHS))
# treina a rede
H = model.fit(x = aug.flow(trainX, trainY, batch_size = BS),
              validation_data = (testX, testY),
              steps_per_epoch = len(trainX) // BS,
              epochs = EPOCHS)

print("\n[INFO] avaliando a rede...")
# avalia a rede
predictions = model.predict(x = testX, batch_size = BS)
print(classification_report(testY.argmax(axis = 1),
                            predictions.argmax(axis = 1),
                            target_names = le.classes_))

print("[INFO] salvando a rede em '{}'...".format(MODEL))
# salva a rede (.h5)
model.save(MODEL, save_format = "h5")
# salva o label encoder (.pickle)
f = open(ENCODER, "wb")
f.write(pickle.dumps(le))
f.close()
# plota a perda e a acurácia do treinamento
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label = "train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label = "val_acc")
plt.title("Loss (perda) e Accuracy (acurácia) no dataset de treinamento")
plt.xlabel("Época #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(PLOT)
