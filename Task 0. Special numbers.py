import numpy as np
import sys


def isNaN(num):
    if float('-inf') < float(num) < float('inf'):
        return False
    else:
        return True


# Ищем машинный нуль

# 32bit
MACHINE_ZERO_32 = np.float32(1.0)
while MACHINE_ZERO_32 != np.float32(0.0):
    NUMBER_32 = MACHINE_ZERO_32
    MACHINE_ZERO_32 /= np.float32(2.0)
MACHINE_ZERO_32 = np.float32(NUMBER_32)
print('Machine zero for float32 is:', MACHINE_ZERO_32)

# 64bit
MACHINE_ZERO_64 = 1.0
while MACHINE_ZERO_64 != 0.0:
    NUMBER_64 = MACHINE_ZERO_64
    MACHINE_ZERO_64 /= 2.0
MACHINE_ZERO_64 = NUMBER_64
print('Machine zero for float64 is:', MACHINE_ZERO_64)

# Ищем машинный эпсилон

# 32bit
MACHINE_EPS_32 = np.float32(1.0)
while np.float32(1.0) != np.float32(1.0) + MACHINE_EPS_32:
    NUMBER_32 = MACHINE_EPS_32
    MACHINE_EPS_32 /= np.float32(2.0)
MACHINE_EPS_32 = NUMBER_32
print('Machine epsilon for float32 is:', MACHINE_EPS_32)

# 64 bit
MACHINE_EPS_64 = 1.0
while 1.0 != 1.0 + MACHINE_EPS_64:
    NUMBER_64 = MACHINE_EPS_64
    MACHINE_EPS_64 /= 2.0
MACHINE_EPS_64 = NUMBER_64
print('Machine epsilon for float64 is:', MACHINE_EPS_64)

# Ищем машинную бесконечность
# 64 bit
MACHINE_INF = 1.0
while not isNaN(MACHINE_INF):
    NUMBER = MACHINE_INF
    MACHINE_INF *= 2
MACHINE_INF = NUMBER
print('Machine infinity for float64 is:', MACHINE_INF)
