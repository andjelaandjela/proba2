import os
import cv2
import math
import numpy as np
    
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils import np_utils    

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# Neuronska mreza

def model_neuronske_mreze():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

def treniraj_neuronsku_mrezu():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    model = model();
    model.fit(X_train, Y_train, batch_size=128, epochs=17,verbose=2,validation_data=(X_test, Y_test))

    model.save('model.h5')
    return model

# Pronalazak linije 

def nadji_liniju(boja):
    zamucena = cv2.GaussianBlur(boja, (5, 5), 0) 
    erozija = cv2.erode(zamucena, np.ones((3, 3))) 
    ret, binarna = cv2.threshold(erozija,68,255,cv2.THRESH_BINARY)
    linije = cv2.HoughLinesP(binarna, 1, np.pi / 180, 110,240, 55)
    
    odabrana_linija = linije[0][0]
    for linija in linije:
        crta = linija[0]
        if(crta[0] < odabrana_linija[0] or crta[1] > odabrana_linija[1] ):
            odabrana_linija[0] = crta[0]
            odabrana_linija[1] = crta[1]
        if(crta[2] > odabrana_linija[2] or crta[3] < odabrana_linija[3] ):
            odabrana_linija[2] = crta[2]
            odabrana_linija[3] = crta[3]
    return odabrana_linija

# Klasa koja sadrzi sve potrebno informacije o broju

class Broj:
    def __init__(self,vrednost, x, y, sirina, visina, identifikator):
        self.vrednost = vrednost
        self.x = x
        self.y = y 
        self.sirina = sirina 
        self.visina = visina 
        self.identifikator = identifikator
        self.presao_plavu = False
        self.presao_zelenu = False
        self.broj_bez_promena = 0

# Pronalazak brojeva, njihivo pracenje i racunanje 

def izracunaj(video):

    ret, frame = video.read()
    if ret is None or ret is False:
        return
    plava, zelena, crvena = cv2.split(frame)
    plava_linija = nadji_liniju(plava)
    zelena_linija = nadji_liniju(zelena)
    brojac = 0
    brojevi = []
    za_sabiranje = []
    za_oduzimanje = []
    # cv2.line(frame,(plava_linija[0],plava_linija[1]),(plava_linija[2],plava_linija[3]),(0,0,255),3)
    # cv2.line(frame,(zelena_linija[0],zelena_linija[1]),(zelena_linija[2],zelena_linija[3]),(0,0,255),3)
    # cv2.imshow('Oznacene linije', frame)
    # if cv2.sirinaaitKey(0) & 0xFF == ord('q'):
    #     pass
    while True:
        ret, frame = video.read()
        if ret is None or ret is False:
            break
        plava, zelena, crvena = cv2.split(frame)

        # pronadji regione od interesa
        regioni_interesa = []
        zamucena = cv2.GaussianBlur(crvena, (5, 5), 0)
        erozija = cv2.erode(zamucena,np.ones((2,2)))
        dilatacija = cv2.dilate(erozija,np.ones((3,3)))
        binarna = cv2.threshold(dilatacija,68,255,cv2.THRESH_BINARY)[1] 
        konture, nesto =  cv2.findContours(binarna, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for kontura in konture:
            x, y, sirina, visina = cv2.boundingRect(kontura) 
            if sirina > 2 and visina > 13:
                regioni_interesa.append((x,y,sirina,visina))
        
        ne_pomereni_brojevi = brojevi.copy()
        for region in regioni_interesa:
            x, y, sirina, visina = region         
            isecen_region = binarna[y-1:y+visina+1, x-1:x+sirina+1]
            kopija_isecenog_regiona = isecen_region.copy();   
            region_spreman_za_mrezu = cv2.resize(kopija_isecenog_regiona , (28,28)).reshape(1,28,28,1)
            
            rezultat = neuronska_mreza.predict(region_spreman_za_mrezu)
            vrednost_regiona = 0
            for i in range(len(rezultat[0])):
                if vrednost_regiona < rezultat[0][i]:
                     vrednost_regiona = i
            
            pomereni_broj = None
            for broj in ne_pomereni_brojevi:
                if  math.sqrt( math.pow(x-broj.x,2) + math.pow(y-broj.y,2) ) < 20 :
                    pomereni_broj = broj
                    break   
            if pomereni_broj is None:                       
                brojac += 1
                pomereni_broj =Broj(vrednost_regiona,x,y,sirina,visina,brojac) 
                brojevi.append(pomereni_broj) 
            else:
                ne_pomereni_brojevi.remove(pomereni_broj) 
                for broj in brojevi:
                    if broj.identifikator == pomereni_broj.identifikator:
                        broj.x = x
                        broj.y = y
                        break   
                if not pomereni_broj.presao_plavu:
                    if (x + pomereni_broj.sirina >= plava_linija[0] and x + pomereni_broj.sirina <= plava_linija[2] and y <= plava_linija[1] and y >= plava_linija[3]
                        or( x  >= plava_linija[0] and x <= plava_linija[2] and y + pomereni_broj.visina <= plava_linija[1] and y +pomereni_broj.visina >= plava_linija[3])):
                        if ( y - plava_linija[1] >= ((plava_linija[3] - plava_linija[1]) / (plava_linija[2] - plava_linija[0])) * (x +pomereni_broj.sirina - plava_linija[0])
                            or y +pomereni_broj.visina - plava_linija[1] >= ((plava_linija[3] - plava_linija[1]) / (plava_linija[2] - plava_linija[0])) * (x - plava_linija[0]) ):
                            za_sabiranje.append(pomereni_broj.vrednost)
                            pomereni_broj.presao_plavu = True
                if not pomereni_broj.presao_zelenu:
                    if ((x + pomereni_broj.sirina  >= zelena_linija[0] and x + pomereni_broj.sirina <= zelena_linija[2] and y <= zelena_linija[1] and y >= zelena_linija[3]) 
                        or (x >= zelena_linija[0] and x <= zelena_linija[2] and y + pomereni_broj.visina  <= zelena_linija[1] and y +pomereni_broj.visina >= zelena_linija[3])):
                        if (y - zelena_linija[1] >= ((zelena_linija[3] - zelena_linija[1]) / (zelena_linija[2] - zelena_linija[0])) * (x +pomereni_broj.sirina - zelena_linija[0])
                        or y + pomereni_broj.visina - zelena_linija[1] >= ((zelena_linija[3] - zelena_linija[1]) / (zelena_linija[2] - zelena_linija[0])) * (x - zelena_linija[0])) :
                            za_oduzimanje.append(pomereni_broj.vrednost)
                            pomereni_broj.presao_zelenu = True 
        #     cv2.rectangle(frame, (x + sirina , y), (x,y+visina) , (0,0,255),1)
        # cv2.line(frame,(plava_linija[0],plava_linija[1]),(plava_linija[2],plava_linija[3]),(0,0,255),3)
        # cv2.line(frame,(zelena_linija[0],zelena_linija[1]),(zelena_linija[2],zelena_linija[3]),(0,0,255),3)
        # cv2.imshow('frame', frame)
        # for broj in brojevi:
        #     if True:
        #         print('[' + str(broj.vrednost) , end='] ')
        # print(f'Count: {len(brojevi)}')
        # print(f'Result: {sum(za_sabiranje) - sum(za_oduzimanje)}')

        if cv2.waitKey(0) & 0xFF == ord('q'):
            pass
        za_obrisati = []
        for broj in brojevi:
            nije_se_pomerio = False
            for nepomeren_broj in ne_pomereni_brojevi:
                if nepomeren_broj.identifikator == broj.identifikator:
                    nije_se_pomerio = True
            if nije_se_pomerio:
                broj.broj_bez_promena +=1
            if broj.broj_bez_promena == 22:
                za_obrisati.append(broj)
        for broj in za_obrisati:
            brojevi.remove(broj)
    return za_sabiranje , za_oduzimanje

# Izvrsavanje gore navedenih funkcija za svaki snimak
 
# neuronska_mreza = treniraj_neuronsku_mrezu()
neuronska_mreza = model_neuronske_mreze()
neuronska_mreza.load_weights('model.h5') 

file = open('video/out.txt','w')
file.write('RA 177/2015 Andjela Todorovic\nfile\tsum\n')
for i in range(10):
    snimak = cv2.VideoCapture('video/video-' + str(i) + '.avi')  
    print('video-' + str(i) +'.avi')
    za_sabiranje, za_oduzimanje = izracunaj(snimak)
    suma = sum(za_sabiranje) - sum(za_oduzimanje)
    snimak.release()
    file.write('video-' + str(i) +'.avi\t'+ str(suma) +'\n')
file.close()

