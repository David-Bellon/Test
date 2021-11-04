from enum import Enum

class Weapons(Enum):
    codazo = 64
    puñetazo = 20
    patada = 34
    cabezazo = 10

print(Weapons.codazo.value)