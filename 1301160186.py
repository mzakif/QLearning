import numpy as np
import random

data = np.loadtxt('DataTugas3ML2019.txt')

#random untuk mencari nilai titik
def rand():
    x = random.randint(0, 14)
    return x
#random untuk mencari arah mau kemaana
def randG():
#    0 = North(atas), 1 = South(Bawah), 2 = West(Kiri), 3 = East(Kanan)
    gerak = random.randint(0, 3)
    return gerak

#untuk menampung nilai dari N,S,W,E || Qarah merupakan Qtabel penyimpan nilai gerak
Qarah = np.zeros((15 * 15,4))

#membuat tabel untuk menyampakan tatak letak 0-224
Q1 = np.zeros((15,15))
for i in range(15):
    for j in range(15):
        Q1[i][j] = ((i*15)+(j+1))-1
Q1 = Q1.T

#inisialisasi beberapa attribute
episode = 10000
alpha = 1
gamma = 0.9
move = 30

#learning mencari nilai Qarah
for i in range(episode):
    xc = rand() 
    yc = rand() 
    j = 1
    while (j <= move):
        xa = xc
        ya = yc
        x = 1
        while (x == 1):
            gerak = randG()
            if (gerak == 0):
                if (yc - 1 < 0):
                    x = 1 
                else:
                    yc = yc - 1
                    x = 0
            elif(gerak == 1):
                if (yc + 1 > 14):
                    x = 1
                else:
                    yc = yc + 1
                    x = 0
            elif(gerak == 2):
                if (xc - 1 < 0):
                    x = 1
                else:
                    xc = xc - 1
                    x = 0
            elif(gerak == 3):
                if (xc + 1 > 14):
                    x = 1
                else:
                    xc = xc + 1
                    x = 0
        f = Q1[ya,xa]
        f1 = Q1[yc,xc]

        Qarah[int(f),gerak] = Qarah[int(f),gerak] + alpha * ( data[yc,xc] + gamma * max(Qarah[int(f1)]) - Qarah[int(f),gerak] )
        
        j = j + 1
        
#mencari nilai reward untuk posisi start (14,0) dengan goal(0,14)
yawal = 14
xawal = 0
ygoal = 0
xgoal = 14
reward = 0
arah = []
while (True):
    d = Q1[yawal][xawal]
    f = np.argmax(Qarah[int(d)])
    i = 0
    if (f == 0):
        yawal = yawal - 1
        arah.append('North')
    elif (f == 1):
        yawal = yawal + 1
        arah.append('South')
    elif (f == 2):
        xawal = xawal - 1
        arah.append('West')
    elif (f == 3):
        xawal = xawal + 1
        arah.append('East')
    c= data[yawal][xawal]
    reward = reward + c
    i = i + 1
    if ((yawal == ygoal) and (xawal == xgoal)):
        break

print('Arah :', arah)

#visualisasi arah
import matplotlib.pyplot as plt
yk= 14
xk = 0
plt.figure(figsize=(20,5))
plt.axis('off')
table = plt.table(cellText=data,loc='center', fontsize=1)
mo = 1
table.get_celld()[yk,xk].set_color('red')
for mo in range(len(arah)):
    if (arah[mo] == 'North'):      
        table.get_celld()[yk-1,xk].set_color('red')
        yk = yk - 1
    elif (arah[i] == 'East'):
        table.get_celld()[yk,xk+1].set_color('red')
        xk = xk + 1
    elif (arah[i] == 'West'):
        table.get_celld()[yk,xk-1].set_color('red')
        xk = xk - 1
    elif (arah[i] == 'South'):
        table.get_celld()[yk+1,xk].set_color('red')
        yk = yk + 1
    
plt.show()
print('Reward: ',reward)
