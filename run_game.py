from pyboy import PyBoy

pyboy = PyBoy('/home/joe/Projects/PlayPokemonRed/ROMs/PokemonRed.gb')
with open('/home/joe/Downloads/init.state', "rb") as f:
    pyboy.load_state(f)
while not pyboy.tick():
    pass
pyboy.stop()

