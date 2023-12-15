from pyboy import PyBoy

pyboy = PyBoy('ROMs/PokemonBlue.gb')
while not pyboy.tick():
    pass
pyboy.stop()