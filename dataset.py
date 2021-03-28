# Liveness detector - https://github.com/alexcamargoweb/liveness-detection.
# Prova de vida através de biometria facial utilizando OpenCV.
# Adrian Rosebrock, Liveness Detection with OpenCV. PyImageSearch.
# Disponível em: https://www.pyimagesearch.com/2019/03/11/liveness-detection-with-opencv/.
# Acessado em: 23/03/2021.
# Arquivo: dataset.py
# Execução via PyCharm/Linux (Python 3.8)
# $ conda activate tensorflow_keras

# importa os pacotes necessários
import numpy as np
import cv2
import os

# inputs dos vídeos "real" ou "fake" (executar um de cada vez)
DATASET = 'real'
# DATASET = 'fake'

if DATASET == 'real':
    INPUT = './input/real.mp4'  # vídeo de entrada "real"
    OUTPUT = './dataset/real'  # saída das imagens do ROI

if DATASET == 'fake':
    INPUT = './input/fake.mp4'  # vídeo de entrada "fake"
    OUTPUT = './dataset/fake'  # saída das imagens do ROI

# detector de faces deep learning e opencv caffemodel
DETECTOR = './detector'
# probabilidade mínima para filtrar "weak predictions"
CONFIDENCE = 0.5
# frames para ignorar antes de aplicar o face detection (não armazena frames parecidos)
SKIP = 9

# carrega o face detector
print("[INFO] carregando o detector de faces...")
protoPath = os.path.sep.join([DETECTOR, "deploy.prototxt"])
modelPath = os.path.sep.join([DETECTOR, "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# abre um ponteiro para o stream do arquivo de vídeo
vs = cv2.VideoCapture(INPUT)

# inicializa o total de frames lidos e salvos até o momento
read = 0
saved = 0

# faz um loop sobre o stream de vídeo
while True:

    # pega um frame do arquivo
    (grabbed, frame) = vs.read()
    # se o frame não foi carregado, então é o fim do vídeo
    if not grabbed:
        break

    # incrementa o total de frames lidos
    read += 1
    # verifica se o frame atual precisa ser processado
    if read % SKIP != 0:
        continue

    # pega as dimensões do frame e constrói um blob da imagem
    (h, w) = frame.shape[:2]
    # 300x300 devido ao Caffe face detector model
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # passa o blob pela rede e obtém as detecções
    net.setInput(blob)
    detections = net.forward()

    # certifica-se que pelo menos um rosto foi encontrado
    if len(detections) > 0:

        # aqui é assumido que cada imagem tem somente um rosto,
        # então é buscado a bounding box com maior probabilidade
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # garante que a detecção com a maior probabilidade também
        # é maior que a probabilidade mínima definida
        if confidence > CONFIDENCE:
            # processa as coordenadas X e Y da bouding box da face e extrai o ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
            # salva o frame em imagem no disco
            p = os.path.sep.join([OUTPUT, "{}.png".format(saved)])
            cv2.imwrite(p, face)
            # incrementa o total de imagens salvas
            saved += 1
            print("[INFO] salvando em {} ...".format(p))

# limpa a execução
vs.release()
cv2.destroyAllWindows()
