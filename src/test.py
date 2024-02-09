import logging
logging.basicConfig(
    filename="/home/joe/Projects/PlayPokemonRed/src/logs/Train_Ray.log",
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info("Starting Ray")