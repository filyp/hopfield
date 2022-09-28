# source: https://www.jhanley.com/blog/pyscript-graphics/

# resources:
# https://pyodide.org/en/stable/usage/faq.html#how-can-i-use-a-python-function-as-an-event-handler
# https://ipycanvas.readthedocs.io/en/latest/animations.html
# https://gist.github.com/jtpio/ac9fa41239fd0098ede03ec042aec574
# comparison of other libraries: https://github.com/flexxui/flexx/wiki/PyScript-vs-X

from audioop import add
import numpy as np
from time import time

from js import document
from pyodide import create_proxy
import asyncio

n_pixels = 50
canvas = document.getElementById("canvas")
ctx = canvas.getContext("2d")

class Network:
    def __init__(self, nonlinearity=5):
        self.memories = []
        self.nonlinearity = nonlinearity
    
    def add_memory(self, memory):
        self.memories.append(memory)
    
    def calculate_weights(self):
        weights = []
        for memory in self.memories:
            _weights = np.outer(memory, memory)
            # _weights /= np.mean(memory @ _weights) / np.mean(memory)
            weights.append(_weights)
        self.weights = np.mean(weights, axis=0)

    def single_value_hopfield_update(self, state, index):
        # single value hopfied update
        value = self.weights[index] @ state
        state[index] = np.clip(self.nonlinearity * value, -1, 1)
        return state


def draw(state, canvas, n_pixels=50):
    state = (state + 1) / 2
    # canvas.clear()
    state = state.reshape(8, 8)
    for i, row in enumerate(state):
        for j, value in enumerate(row):
            value = int(value * 255)
            color = f'rgb({value}, {value}, {value})'
            canvas.fillStyle = color
            canvas.fillRect(i * n_pixels, j * n_pixels, n_pixels, n_pixels)
    

def on_click(event):
    global state
    x = event.offsetX
    y = event.offsetY
    x = int(x / n_pixels)
    y = int(y / n_pixels)
    index = x * 8 + y
    if state[index] != 1.0:
        state[index] = 1.0
        draw(state, ctx)


async def run(event):
    global state, network
    # initialize random state
    state = np.random.choice([-1.0, 1.0], size=64)
    draw(state, ctx)

    # randomply shuffled indexes
    indexes = np.arange(0, 64)
    np.random.shuffle(indexes)
    # duplicate indexes
    indexes = np.concatenate([indexes, indexes])

    start_time = time()
    for i, index in enumerate(indexes):
        _to_sleep = i * 0.01 - (time() - start_time)
        await asyncio.sleep(max(0, _to_sleep))

        draw(state, ctx)
        state = network.single_value_hopfield_update(state, index)


def clear(event):
    global state
    state = -np.ones(64)
    draw(state, ctx)


def add_memory(event):
    global state, network
    network.add_memory(state)
    network.calculate_weights()


async def main():
    global state, network
    network = Network()
    clear(None)

    canvas.addEventListener("click", create_proxy(on_click))
    document.getElementById("run-button").addEventListener("click", create_proxy(run))
    document.getElementById("clear-button").addEventListener("click", create_proxy(clear))
    document.getElementById("add-memory-button").addEventListener("click", create_proxy(add_memory))


main()
