import numpy as np
import matplotlib.pyplot as plt

def rozklad_choleskiego(A):
    n = len(A)
    L = np.zeros((n, n))
    for r in range(n):
        for c in range(r + 1):
            val = A[r, c] - np.sum(L[r, :c] * L[c, :c])
            if r == c:
                if val <= 1e-10:
                    raise ValueError("Macierz nie jest dodatnio określona")
                L[r, c] = np.sqrt(val)
            else:
                L[r, c] = val / L[c, c]
    return L

def obliczenie_dolnej_trojkatnej(L, b):
    n = len(b)
    z = np.zeros(n)
    for r in range(n):
        v = b[r] - np.sum(L[r, :r] * z[:r])
        if abs(L[r, r]) <= 1e-10:
            raise ZeroDivisionError("Dzielenie przez liczbę bliską zeru")
        z[r] = v / L[r, r]
    return z

def obliczenie_gornej_trojkatnej(U, z):
    n = len(z)
    a = np.zeros(n)
    for r in range(n - 1, -1, -1):
        s = np.dot(U[r, r + 1:], a[r + 1:])
        if abs(U[r, r]) <= 1e-10:
            raise ZeroDivisionError("Dzielenie przez liczbę bliską zeru")
        a[r] = (z[r] - s) / U[r, r]
    return a

def rysowanie_wykresu(x, y, wsp, tytul):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, color='green', label='punkty')
    xp = np.linspace(np.min(x), np.max(x), 300)
    yp = np.polyval(wsp[::-1], xp)
    ax.plot(xp, yp, label=tytul)
    ax.set(xlabel='x', ylabel='y', title=tytul)
    ax.grid(True)
    ax.legend()
    plt.show()

def Natan_Misztal_MNK(x, y, n, plot=True):
    X = np.vander(x, n + 1, increasing=True)
    ATA = X.T @ X
    ATb = X.T @ y
    L = rozklad_choleskiego(ATA)
    z = obliczenie_dolnej_trojkatnej(L, ATb)
    wsp = obliczenie_gornej_trojkatnej(L.T, z)
    if plot:
        rysowanie_wykresu(x, y, wsp, tytul=f'Aproksymacja MNK stopnia {n}')
    return wsp

x = [0, 1, 2, -2, 4, -2]
y = [0, 1, 4, 4, -16, 4]
stopien = 5
print(f"Dla danych x={x}, y={y}, stopien={stopien}")
wsp = Natan_Misztal_MNK(x, y, stopien, plot=True)
print("Obliczone współczynniki:", np.round(wsp, 5))
