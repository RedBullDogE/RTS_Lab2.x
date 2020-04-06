import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import time
import random
from math import pi, cos, sin



def generate_signal(n, amp, ph, freq, p, a, b):
    w_step = freq / n
    rand_val = [
        (random.randint(0, amp), random.uniform(0, ph), w_step * (i + 1))
        for i in range(n)
    ]

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


def pfft(x):
    n = x.size

    if n <= 1:
        return x

    with concurrent.futures.ProcessPoolExecutor() as executor:
        even_thread = executor.submit(fft, (x[::2]))
        odd_thread = executor.submit(fft, (x[1::2]))
        even = even_thread.result()
        odd = odd_thread.result()

    factor = np.exp(-2j * np.pi * np.arange(n) / n)

    return np.concatenate([even + factor[:int(n/2)] * odd,
                           even + factor[int(n/2):] * odd])


def additional_task(x):
    start1 = time.time()
    fft(x)
    total1 = time.time() - start1

    start2 = time.time()
    pfft(x)
    total2 = time.time() - start2

    return total1, total2

if __name__ == "__main__":
    ampl = 10           # Max. amplitude - A
    phi = 2 * pi        # Max. phase - phi
    n = 8               # Number of harmonics - n
    w = 1200            # Limit frequency - Wgr
    a, b = 0, 0.1       # Generation range

    points = [2**i for i in range(8, 18)]        # Array of points

    time_arr_fft, time_arr_pfft = [], []

    for p in points:
        t, x = generate_signal(n, ampl, phi, w, p, a, b)
        t1, t2 = additional_task(x)
        time_arr_fft.append(t1)
        time_arr_pfft.append(t2)
    
    plt.plot(points, time_arr_fft, label='fft')
    plt.plot(points, time_arr_pfft, label='pfft')

    plt.xlabel('N')
    plt.ylabel('time/s')
    plt.legend()

    plt.show()


