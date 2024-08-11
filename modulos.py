import cv2
import numpy as np
from tensorflow.keras.models import load_model

def cargar():
    modelo1 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    modelo2 = load_model('datos/modelo2_90ep.keras')
    emotions = ['ENFADADO', 'DISGUSTADO', 'MIEDO', 'FELIZ', 'NEUTRAL', 'TRISTE', 'SORPRENDIDO']
    return modelo1,modelo2,emotions

def get_face(image, val,modelo1):
    foto = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = modelo1.detectMultiScale(foto, scaleFactor=1.5, minNeighbors=val, minSize=(30, 30))
    
    if len(coords) > 0:
        x, y, z, t = coords[0]
        face = foto[y:y+t, x:x+z]
        return face, [x, y, z, t]
    else:
        return None, None
    

def predict(modelo2, foto, emotions):
    try:
        array = cv2.resize(foto, (70, 70)) 
        array = array.astype('float32') / 255 
        array = array.reshape(70, 70, 1)
        img = np.expand_dims(array, axis=0)
        pred = np.argmax(modelo2.predict(img, verbose=0))
        emotion = emotions[pred]
        return emotion
    except:
        return None
    
def show_txt(image,text,color, coords):
    x,y,z,t = coords
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    size = cv2.getTextSize(text, fuente, 1, 6)
    anc, lar = size[0]
    t_x = x + (z-anc) // 2
    t_y = y -10
    cv2.putText(image, text, (t_x, t_y + lar), fuente, 1, color, 6)
    cv2.line(image, (x, y), (t_x - 10, y), color, 4)
    cv2.line(image, (t_x + anc + 10, y), (x + z, y), color, 4)


def show_yes(image,coords,pred):
    if pred is not None:
        color = (0, 255, 0)
        texto = pred
    else:
        color = (0, 0, 255)
        texto = '???'

    x,y,z,t = coords
    cv2.line(image, (x, y), (x, y+t), color, 4)
    cv2.line(image, (x+z, y), (x+z, y+t), color, 4)
    cv2.line(image, (x, y+t), (x+z, y+t), color, 4)
    show_txt(image,texto,color,coords)

'''

def launch(ruta,modelo1,modelo2,emotions):
    image = cv2.imread(ruta)
    val = 9
    face = None
    while face is None and val>=0:
        face, coords = get_face(image,val,modelo1)
        val = val - 1

    if face is not None and coords is not None:
        pred = predict(modelo2, face,emotions)
        show_yes(image,coords,pred)
    
    cv2.imshow('Emotion Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''


