# Liveness detector - https://github.com/alexcamargoweb/liveness-detection.
# Prova de vida através de biometria facial utilizando OpenCV.
# Adrian Rosebrock, Liveness Detection with OpenCV. PyImageSearch.
# Disponível em: https://www.pyimagesearch.com/2019/03/11/liveness-detection-with-opencv/.
# Acessado em: 23/03/2021.
# Arquivo: predict.py
# Execução via PyCharm/Linux (Python 3.8)
# $ conda activate tensorflow_keras

# importa os pacotes necessários
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import pickle
import time
import cv2
import os

# input do modelo treinado
MODEL = './model/liveness.model'
# input do label encoder gerado
ENCODER = './encoder/le.pickle'
# detector de faces deep e learning opencv caffemodel
DETECTOR = './detector'
# probabilidade mínima para filtrar "weak predictions"
CONFIDENCE = 0.5

# carrega o face detector
print("[INFO] carregando o detector de faces...")
protoPath = os.path.sep.join([DETECTOR, "deploy.prototxt"])
modelPath = os.path.sep.join([DETECTOR, "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("[INFO] carregado o liveness detector...")
# carrega o modelo liveness (liveness.model)
model = load_model(MODEL)
# carrega o modelo liveness (liveness.model) detector e o label encoder (le.pickle)
le = pickle.loads(open(ENCODER, "rb").read())

# inicializa o stream de vídeo e permite o acesso a câmera
print("[INFO] iniciando stream de vídeo...")
vs = VideoStream(src = 0).start()
time.sleep(2.0)

# faz um loop sobre os frames do vídeo
while True:
    # pega um frame do vídeo e redimensiona para um tamanho máximo de 600px
    frame = vs.read()
    frame = imutils.resize(frame, width = 600)
    # pega as dimensões do frame e constrói um blob da imagem
    (h, w) = frame.shape[:2]
    # 300x300 devido ao Caffe face detector
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    # passa o blob pela rede e obtém as detecções
    net.setInput(blob)
    detections = net.forward()

    # faz um loop sobre as detecções
    for i in range(0, detections.shape[2]):
        # extrai a probabilidade (confidence) associada à predição
        confidence = detections[0, 0, i, 2]
        # garante que a detecção com a maior probabilidade também
        # é maior que a probabilidade mínima definida
        if confidence > CONFIDENCE:
            # processa as coordenadas X e Y da bouding box da face e extrai o ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # certifica-se que a bounding box não está fora das dimensões do frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            # extrai o ROI de face e o pré-processa
            # da mesma maneira que os dados de treinamento
            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (32, 32))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis = 0)
            # passa o ROI da face pela rede liveness detector treinada
            # e o modelo determina se o rosto é "real" or "fake"
            preds = model.predict(face)[0]
            j = np.argmax(preds)
            label = le.classes_[j]
            # desenha o rótulo e a bouding box no frame
            label = "{}: {:.4f}".format(label, preds[j])
            # define a cor da bouding box com base na classe atribuída
            if le.classes_[j] == 'fake':
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            if le.classes_[j] == 'real':
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # exibe o título do frame
    cv2.imshow("Liveness detector", frame)
    key = cv2.waitKey(1) & 0xFF

    # se a tecla 'q' for pressionada, sai do loop
    if key == ord("q"):
        break

# limpa a execução
cv2.destroyAllWindows()
vs.stop()
