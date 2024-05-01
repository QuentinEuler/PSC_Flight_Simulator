import numpy as np
import torch
from matplotlib import pyplot as plt

eps = 1
alpha = 0.001
gamma = 0.9

class NeuralNetwork(torch.nn.Module) :
    def __init__(self) :
        super(NeuralNetwork, self).__init__()

        self.linear1 = torch.nn.Linear(3,16)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(16,8)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(8,1)
        self.activation3 = torch.nn.Sigmoid()

        self.optimizer = torch.optim.SGD(self.parameters(), lr = alpha)

    def forward(self, x) :
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.activation3(x)
        return x

T = NeuralNetwork()

def etat_suivant(s,a) :
    s_ = np.copy(s)
    if a==1 :
        s_[1]+=5
    if a==-1 :
        s_[1]-=5
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
        return 2
    if s[0] > 400 and s[0] < 600 :
        return 1
    return 0

def Q(s,a) :
    e = torch.tensor([s[0].item()/1200+1/12,s[1].item()/90 + 0.5,(a+1)/2])
    if s[0]<-100 :
        e[0] = 0
    if s[0]>=100 :
        e[0] = 1
    a = T(e)
    return a

def learn_Q(s,a,loss) :
    loss.backward()

    T.optimizer.step()
    T.optimizer.zero_grad(set_to_none = False)

def aff(S) :
    absc = [i for i in range(len(S))]
    ords = [i[0] for i in S]
    plt.plot(absc,ords)
    plt.show()

def save_T() :


def episod(eps, alpha, gamma, nbr=100) :
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

        learn_Q(S[-2],A[-1], (gain(S[-2]) + gamma*(Q(S[-1],argmax(lambda x : Q(S[-1],x), [0,1,-1])).item())-Q(S[-2],A[-1]))**2)

    return A,R,S

def eps_delay(j, eps) :
    eps+=1/300000
    return eps

def apprentissage(eps, alpha, gamma) :
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

apprentissage(eps, alpha, gamma)


