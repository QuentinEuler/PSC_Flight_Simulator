import numpy as np
from matplotlib import pyplot as plt

T=np.zeros((1200,91,3))
alpha = 0.01
gamma = 0.9

def etat_suivant(s,a) :
    s_ = np.copy(s)
    if a==1 :
        s_[1]+=1
    if a==-1 :
        s_[1]-=1
    if s_[1]>45 :
        s_[1]=45
    if s_[1]<-45 :
        s_[1]=-45

    s_[0]+=60*np.sin(s_[1]/180*np.pi)

    return s_

def argmax(f, action_set) :
    a_max = 0
    v_max = -np.inf

    np.random.shuffle(action_set)

    for a in action_set :
        v = f(a)
        if v>v_max :
            v_max = v
            a_max = a
    return a_max

def gain(s) :
    if s[0] < 0 or s[0] > 1000 :
        return -1
    if s[0] == 500 :
        return 5
    if s[0] > 300 and s[0] < 700 :
        return 4-np.abs(s[0]-500)/50
    return 0

def aff_gain() :
    absc = [i-100 for i in range(1200)]
    ords = [gain((i-100,0)) for i in range(1200)]
    plt.plot(absc,ords)
    plt.show()

def Q(s,a) :
    if s[0]<-100 :
        return T[0][s[1]+45][a+1]
    if s[0]>=1100 :
        return T[1199][s[1]+45][a+1]
    return T[s[0]+100][s[1]+45][a+1]

def set_Q(s,a,v) :
    if s[0]<-100 :
        T[0][s[1]+45][a+1] = v
    elif s[0]>=1100 :
        T[1199][s[1]+45][a+1] = v
    else :
        T[s[0]+100][s[1]+45][a+1] = v

def aff(S) :
    absc = [i for i in range(len(S))]
    ords = [s[0] for s in S]
    plt.plot(absc,ords)
    plt.show()

def save_T() :
    for i in range(1200) :
        s=str(i-100) + " :\n"
        for j in range(91) :
            s+=" | " + str(j-45) + " : "
            for k in range(3) :
                s+=str(T[i][j][k]) + " "
        s+="\n"
        print(s)

def episod(eps, alpha, gamma, nbr=20) :
    S=[np.array([300,0])]
    A=[]
    R=[]
    for i in range(nbr) :
        a=0
        if np.random.rand() > eps :
            a = np.random.randint(-1,2)
        else :
            a = argmax(lambda x : Q(S[-1],x), [-1,0,1])

        S.append(etat_suivant(S[-1],a))
        A.append(a)
        R.append(gain(S[-1]))

        set_Q(S[-2],A[-1], Q(S[-2],A[-1]) + alpha*(R[-1] + gamma*(Q(S[-1],argmax(lambda x : Q(S[-1],x), [-1,0,1])) - Q(S[-2],A[-1]))))

    return A,R,S

def eps_delay(j, eps) :
    eps+=1/300000
    return eps

def apprentissage(alpha, gamma) :
    eps = 0
    for j in range(300000) :
        eps=eps_delay(j,eps)
        A,R,S = episod(eps, alpha, gamma)

        #print(R)
        #print(A)
        #print(S)

        if j%10000 == 0 :
            print(S)
    aff(S)
    save_T()

apprentissage(alpha, gamma)
