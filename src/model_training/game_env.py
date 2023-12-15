
from datetime import date, datetime
import logging
import time
from dis import dis
from enum import Enum
from multiprocessing.dummy import current_process
from re import T
import re
import sys
import os
from typing import Dict
import uuid
from PIL import Image as im
import numpy as np
from pyboy import PyBoy
import hnswlib

from pyboy.utils import WindowEvent
from enum import Enum

# State Enum
class BATTLE_TYPE(Enum):
    NON_COMBAT=0
    WILD_POKEMON=1
    TRAINER=2
    FAINT=255
    
class ACTION(Enum):
    A=0
    B=1
    UP=2
    DOWN=3
    LEFT=4
    RIGHT=5

ACTION_KEY_MAP = {
    ACTION.A: {
        'PRESS': WindowEvent.PRESS_BUTTON_A,
        'RELEASE': WindowEvent.RELEASE_BUTTON_A,
    },
    ACTION.B: {
        'PRESS': WindowEvent.PRESS_BUTTON_B,
        'RELEASE': WindowEvent.RELEASE_BUTTON_B,
    },
    ACTION.UP: {
        'PRESS': WindowEvent.PRESS_ARROW_UP,
        'RELEASE': WindowEvent.RELEASE_ARROW_UP,
    },
    ACTION.DOWN: {
        'PRESS': WindowEvent.PRESS_ARROW_DOWN,
        'RELEASE': WindowEvent.RELEASE_ARROW_DOWN,
    },
    ACTION.LEFT: {
        'PRESS': WindowEvent.PRESS_ARROW_LEFT,
        'RELEASE': WindowEvent.RELEASE_ARROW_LEFT,
    },
    ACTION.RIGHT: {
        'PRESS': WindowEvent.PRESS_ARROW_RIGHT,
        'RELEASE': WindowEvent.RELEASE_ARROW_RIGHT,
    },
}

class State():
    def __init__(
        self,
        current_screen: np.ndarray,
        battle_type: BATTLE_TYPE,
        pokemon_hp_percent: float,
    ):
        self.current_screen = current_screen
        self.battle_type = battle_type
        self.pokemon_hp_percent = pokemon_hp_percent

    def get(self):
        return self.current_screen

    def get_without_element_dim(self):
        return self.current_screen.reshape(144, 160, 1)

class Reward():

    def __init__(self, explore_score, gym_badges, total_pokemon_exp, total_pokemon_level):
        self.explore_score = explore_score
        self.gym_badges = gym_badges
        self.total_pokemon_exp = total_pokemon_exp
        self.total_pokemon_level = total_pokemon_level

    def get(self):
        return self.explore_score + self.gym_badges + self.total_pokemon_exp + self.total_pokemon_level

    def __str__(self) -> str:
        return f'Explore: {self.explore_score} | Badges: {self.gym_badges} | Total XP: {self.total_pokemon_exp} | Total Level: {self.total_pokemon_level}'


class Game_Env():

    # READ GAME DATA FROM MEMORY 
    def _read_byte(self, addr) -> int:
        return self.pyboy.get_memory_value(addr)

    def _read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self._read_byte(addr))[-bit-1] == '1'

    def _read_multi_byte(self, addr, byte_count) -> int:
        val = 0
        for i in range(byte_count):
            val += pow(256, (byte_count-1-i)) * self._read_byte(addr+i)
        return val

    def _bit_count(self, bits):
        return bin(bits).count('1')

    def __init__(self, config=None, num_env=0):
        # self.session_uuid = uuid.uuid4()
        current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        self.session_uuid = str(current_datetime) + f'_{num_env}'
        self.save_screenshots = config['save_screenshots']
        self.act_freq = config['act_freq']
        self.config = config

        self.pyboy = PyBoy(
                self.config['gb_path'],
                debugging=False,
                disable_input=self.config['disable_input'],
                window_type=self.config['window_type'],
                sound_emulated=False,
                sound=False,
            )
        self.pyboy.set_emulation_speed(5)
        with open(config['init_state'], "rb") as f:
            self.pyboy.load_state(f)

        self.screen_index = hnswlib.Index(space='l2', dim=144*160) # possible options are l2, cosine or ip
        self.screen_index.init_index(max_elements=100000, ef_construction=100, M=16)

        # self.screen_index = faiss.IndexFlatL2(144*160, self.screen_index_path)
        self.screen = self.pyboy.botsupport_manager().screen()




        self.session_dir_path = os.path.join(os.getcwd(), '..', 'sessions', str(self.session_uuid))
        self.screenshot_path = os.path.join(self.session_dir_path, 'img')
        self.screen_index_path = os.path.join(self.session_dir_path, 'screen_index.bin')
        self.game_saves_path = os.path.join(self.session_dir_path, 'game_saves')
        os.makedirs(self.screenshot_path)
        os.makedirs(self.game_saves_path)
        
    def save_game(self, file_name: str):
        with open(os.path.join(self.game_saves_path, file_name), "wb") as f:
            self.pyboy.save_state(f)
        self.screen_index.save_index(self.screen_index_path)


    def reset(self):
        # Reset the emulator
        self.pyboy.stop()
        self.pyboy = PyBoy(
                self.config['gb_path'],
                debugging=False,
                disable_input=self.config['disable_input'],
                window_type=self.config['window_type'],
                sound_emulated=False,
                sound=False,
            )
        self.pyboy.set_emulation_speed(5)
        with open(self.config['init_state'], "rb") as f:
            self.pyboy.load_state(f)

        # Reset Frame Index
        self.screen_index = hnswlib.Index(space='l2', dim=144*160) # possible options are l2, cosine or ip
        self.screen_index.init_index(max_elements=100000, ef_construction=100, M=16)
        time.sleep(4)
    
    # REWARD FUNCTIONS
    def get_explore_reward(self, current_screen: np.ndarray) -> float:
        flat_current_screen = current_screen.flatten()
        empty = self.screen_index.get_current_count() == 0

        if not empty:
            _, dist = self.screen_index.knn_query(flat_current_screen, k=1)
            print(f'Distance to nearest screen: {dist[0][0]}')

        if empty or dist[0][0] > 2_000:
            print('Added new screen to index: ', self.screen_index.get_current_count())
            self.screen_index.add_items(
                flat_current_screen, np.array([self.screen_index.get_current_count()])
            )
            current_screen=current_screen*255
            self.save_screenshot(current_screen.reshape(144, 160))
        return self.screen_index.get_current_count() * .005

    def get_levels_sum(self) -> int:
        poke_levels = [max(self._read_byte(a) - 2, 0) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        return max(sum(poke_levels) - 4, 0) # subtract starting pokemon level

    def get_xps_sum(self) -> int:
        poke_xps = [self._read_byte(a) for a in [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]]
        return sum(poke_xps)


    def get_screen(self) -> np.ndarray:
        current_screen = self.screen.screen_ndarray()

        # if self.save_screenshots:
        #     self.save_screenshot(current_screen)

        # This removes the other two channels, so we only have the grayscale. The screen ndarray has same same value for all 3 channels so we are droping 2/3 of the data
        current_screen = current_screen[:, :, 0:1]
        # Reshape the array to 1 x 160 x 143 x 1 (elements, width, height, channels) this is the format that the model expects
        current_screen = current_screen.reshape(1, 144, 160, 1)
        # Scale the screen to 0-1
        if self.save_screenshots:
            self.save_screenshot(current_screen.reshape(144, 160))
        current_screen = current_screen/255

        return current_screen

    def get_current_state(self) -> State:
        # x_pos = self.pyboy.get_memory_value(0xD362)
        # y_pos = self.pyboy.get_memory_value(0xD361)
        # map_n = self.pyboy.get_memory_value(0xD35E)
        
        current_screen = self.get_screen()
        
        battle_type = BATTLE_TYPE(self._read_byte(0xD057))
        cur_pokemon_hp = [self._read_multi_byte(a, 2) for a in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]]
        max_pokemon_hp = [self._read_multi_byte(a, 2) for a in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]]
        pokemon_hp_percent = sum(cur_pokemon_hp) / sum(max_pokemon_hp)

        current_state = State(
            current_screen=current_screen,
            battle_type=battle_type,
            pokemon_hp_percent=pokemon_hp_percent,
        )
        return current_state

    def get_current_reward(self, current_screen: np.ndarray) -> Reward:
        badge_count = self._read_byte(0xD356)
        explore_score = self.get_explore_reward(current_screen)
        levels_sum = self.get_levels_sum()
        xps_sum = self.get_xps_sum()

        reward = Reward(
            explore_score=explore_score,
            gym_badges=badge_count,
            total_pokemon_exp=xps_sum,
            total_pokemon_level=levels_sum,
        )

        return reward

    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(ACTION_KEY_MAP[action]['PRESS'])
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                self.pyboy.send_input(ACTION_KEY_MAP[action]['RELEASE'])
            self.pyboy.tick()

    def step(self, action: ACTION):
        self.run_action_on_emulator(action)

        state = self.get_current_state()
        reward = self.get_current_reward(state.current_screen)

        return state, reward, False, {}

    def save_screenshot(self, current_screen):
        # file name is timestamp
        os.makedirs(self.screenshot_path, exist_ok=True)
        file_name = datetime.now().strftime("%Y%m%d-%H%M%S") + ".png"
        im.fromarray(current_screen).convert('L').save(os.path.join(self.screenshot_path, file_name))