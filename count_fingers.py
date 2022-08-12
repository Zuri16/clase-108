import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

#Son las identificaciones del numero de las puntas de los dedos
tipIds = [4, 8, 12, 16, 20]

# Definir una función para contar dedos
def countFingers(image, hand_landmarks, handNo=0):
    #Verifica si llegaron los puntos de referencia
    if hand_landmarks:
        #Guarda todos las posiciones de los puntos de referencia de la primer mano visible
        landMarks=hand_landmarks[handNo].landmark
        print(landMarks)
        fingers=[]
        #Para recorrer toda la lista de identificaciones 
        for lm_index in tipIds:
            punta_y=landMarks[lm_index].y
            dedo_abajo_y=landMarks[lm_index-2].y
            if lm_index != 4:
                if punta_y < dedo_abajo_y:
                    fingers.append(1)
                    print("El dedo",lm_index,"esta abierto")
                if punta_y > dedo_abajo_y:
                    fingers.append(0)
                    print("El dedo ",lm_index, "esta cerrado")
        totalFingers=fingers.count(1)
        text=f'dedos:{totalFingers}'
        cv2.putText(image,text,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

# Definir una función para
def drawHandLanmarks(image, hand_landmarks):

    # Dibujar conexiones entre los puntos de referencia
    if hand_landmarks:

      for landmarks in hand_landmarks:
               
        mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)


while True:
    success, image = cap.read()

    image = cv2.flip(image, 1)
    
    # Detectar los puntos de referencia de las manos
    results = hands.process(image)

    # Obtener la posición de los puntos de referencia del resultado procesado
    hand_landmarks = results.multi_hand_landmarks

    # Dibujar puntos de referencia
    drawHandLanmarks(image, hand_landmarks)

    # Obtener la posición de los dedos de la mano
    countFingers(image,hand_landmarks)

    cv2.imshow("Controlador de medios", image)

    # Cerrar la ventana al presionar la barra espaciadora
    key = cv2.waitKey(1)
    if key == 32:
        break

cv2.destroyAllWindows()
