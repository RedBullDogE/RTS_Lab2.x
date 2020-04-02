import matplotlib.pyplot as plt
import numpy as np
import random
from math import pi, cos, sin


def generate_signal(n, amp, ph, freq, p, a, b):
    w_step = freq / n
    rand_val = [
        (random.randint(0, amp), random.uniform(0, ph), w_step * (i + 1))
        for i in range(n)
    ]
    for r in rand_val:
        print(f'A = {r[0]};  phi = {r[1]}; W = {r[2]}')

    time_axis = np.linspace(a, b, p)
    sign_axis = sum([np.array([val[0] * sin(2 * pi * (val[2] * t + val[1])) for t in time_axis])
                     for val in rand_val])

    return (time_axis, sign_axis)


def dft(x):
    def coeff(p, k, N):
        res = cos((2 * pi * p * k) / N) - 1j * sin((2 * pi * p * k) / N)
        return complex(round(res.real, 2), round(res.imag, 2))

    p = np.arange(len(x))

    return np.array([sum([x[k] * coeff(p_i, k, len(x))
                          for k in range(len(x))]) for p_i in p])


def fft(x):
    n = x.size

    if n <= 1:
        return x

    even = fft(x[::2])
    odd = fft(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(n) / n)

    return np.concatenate([even + factor[:int(n/2)] * odd,
                           even + factor[int(n/2):] * odd])


if __name__ == "__main__":
    ampl = 10           # Max. amplitude - A
    phi = 2 * pi        # Max. phase - phi
    n = 8               # Number of harmonics - n
    w = 1200            # Limit frequency - Wgr
    points = 256        # Number of points - N
    a, b = 0, 0.1       # Generation range

    t, x = generate_signal(n, ampl, phi, w, points, a, b)

    ans1 = fft(x)
    ans2 = dft(x)

    fig1, (ax1, ax2, ax3) = plt.subplots(3)

    ax1.plot(t, x)
    ax1.set_title('Signal')

    ax2.plot(2/points * np.abs(ans1))
    ax2.set_title('DFT')

    ax3.plot(2/points * np.abs(ans2))
    ax3.set_title('FFT')

    plt.show()
