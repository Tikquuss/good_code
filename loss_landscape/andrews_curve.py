import torch
import numpy as np

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def element_wise_dot(u, v) :
    return u * v[:, None] # (|v|, |u|)

def get_u(alpha, d) :
    k = (d-1)//2
    r = (d-1)%2
    try :
        L = len(alpha)
        u = np.zeros((L, d-1))
        u[:, 0::2] = np.sin(element_wise_dot(np.arange(1, k + r + 1), alpha))
        u[:, 1::2] = np.cos(element_wise_dot(np.arange(1, k + 1), alpha))
        u = np.concatenate((np.array([1 / np.sqrt(2)]*L)[:, None], u), axis=1) # (L,d)
    except TypeError: #object of type 'int' has no len()
        u = np.zeros((d-1,))
        u[0::2] = np.sin(np.arange(1, k + r + 1) * alpha)
        u[1::2] = np.cos(np.arange(1, k + 1) * alpha)
        u = np.concatenate(([1 / np.sqrt(2)], u), axis=0) # (d,)
    return u

def g_ft(u, thetas) :
    """u ~ (L,d), thetas ~ [(d,.)]"""
    if type(thetas) == list :
        return np.stack([np.dot(u, theta_t) for theta_t in thetas]) # [(L,d) x (d,1) = (L,1)] x .
        #return np.dot(u, np.stack(thetas, axis=0).T).T # (L,d) x (d,.) = (L,.) 
    else :
        return np.dot(u, thetas.T).T # (L,d) x (d,.) = (L,.)


def linspace(v : list, N : int):
    M = len(v)
    #print(v)
    step = M if N == 1 else (M-1)//(N-1)
    tmp = v[0::step]
    tmp = v[:N] if len(tmp) == len(v) else tmp[:N]
    return tmp


def get_new_cmap(name, N) :
    #name='viridis'
    #name="plasma"
    #name='inferno'
    #name='magma'
    cmap = plt.get_cmap(name)
    colors = linspace(v = cmap.colors, N = N)
    new_cmap = ListedColormap(colors)
    return colors, new_cmap

if __name__ == "__main__":

    #######
    d = 3
    alpha = np.array([0, 1, 2]) + 0.0

    for a in alpha : print(get_u(a, d))

    print(get_u(alpha, d))

    #######
    d=4
    N=200
    thetas = [np.random.rand(d) for _ in range(N)]

    L=100
    alpha = np.linspace(start = -np.pi, stop = np.pi, num=L)

    u = get_u(alpha, d)
    f_t = g_ft(u, thetas)


    print(u.shape, f_t.shape)