import threading
import time
from contextlib import asynccontextmanager
from queue import Queue

import mido
import numpy as np
from fastapi import FastAPI
from mido import Message, open_output
from pydantic import BaseModel, Field

##################################
# Global State & Setup
##################################

# For demonstration, we simulate generating a single .npy file "generated_output.npy".
# Make sure you have that file in the same directory or adjust `generate_fake_bar`.


class GenerationParams(BaseModel):
    bpm: float = Field(60.0, gt=0, description="Beats per minute")
    class1: float = Field(0.5, description="GAN parameter/class dimension 1")
    class2: float = Field(0.5, description="GAN parameter/class dimension 2")
    class3: float = Field(0.5, description="GAN parameter/class dimension 3")


CURRENT_PARAMS = GenerationParams()
bar_queue = Queue()
MIDI_OUTPUT_PORT = "IAC Driver Bus 1"
RUN_GENERATOR = True
RUN_PLAYER = True

TOTAL_NUM_FAMILIES = 10
TOTAL_TIME_LOCATIONS = 192
family_to_midi = {1: 36, 2: 38, 3: 48, 4: 45, 5: 43, 6: 46, 7: 42, 8: 49, 9: 57, 10: 51}

##################################
# Helper Functions
##################################


def get_division_grid() -> list:
    sixteenth_divs = np.linspace(0, 4, 128, endpoint=False)
    eighth_triplet_divs = np.linspace(0, 4, 96, endpoint=False)
    grid = sorted(set(sixteenth_divs).union(set(eighth_triplet_divs)))
    grid.append(4.0)
    return grid


def ticks_to_seconds(ticks, bpm=120.0, tpb=480):
    return ticks * (60.0 / (bpm * tpb))


##################################
# Real-time MIDI Player
##################################


def play_bar_realtime(
    bar_array: np.ndarray,
    outport: mido.ports.BaseOutput,
    bpm: float,
    tpb: int = 480,
    note_length_ticks: int = 30,
):
    grid = get_division_grid()
    events = []
    for col in range(TOTAL_TIME_LOCATIONS):
        onset_tick = int(round(grid[col] * tpb))
        for row in range(TOTAL_NUM_FAMILIES):
            cell_value = bar_array[row, col]
            if cell_value > -1:
                velocity = int(cell_value + 1)
                midi_note = family_to_midi.get(row + 1)
                if midi_note is None:
                    continue
                # Note-on
                events.append(
                    (onset_tick, Message("note_on", note=midi_note, velocity=velocity))
                )
                # Note-off
                off_tick = onset_tick + note_length_ticks
                events.append(
                    (off_tick, Message("note_off", note=midi_note, velocity=0))
                )

    events.sort(key=lambda x: x[0])

    start_time = time.monotonic()
    for abs_tick, msg in events:
        event_time_offset = ticks_to_seconds(abs_tick, bpm=bpm, tpb=tpb)
        scheduled_time = start_time + event_time_offset
        while True:
            now = time.monotonic()
            if now >= scheduled_time:
                break
            time.sleep(0.0005)
        outport.send(msg)


def bar_player_loop():
    """
    Continuously consumes bars from bar_queue and plays them in real time.
    """
    print("[Player] Starting bar_player_loop...")
    try:
        with open_output(MIDI_OUTPUT_PORT) as outport:
            while RUN_PLAYER:
                bar = bar_queue.get()
                if bar is None:
                    print("[Player] Received None. Stopping bar_player_loop.")
                    return
                local_bpm = CURRENT_PARAMS.bpm
                play_bar_realtime(bar, outport, bpm=local_bpm)
    except Exception as e:
        print(f"[Player] Error in bar_player_loop: {e}")


##################################
# Bar Generator
##################################


def generate_bar(params: GenerationParams) -> np.ndarray:
    """
    Replace this with your actual GAN generation code.
    Currently we load from "generated_output.npy" for demonstration.
    """
    return generate_fake_bar()


def generate_fake_bar(npy_file="generated_output.npy") -> np.ndarray:
    with open(npy_file, "rb") as f:
        bar_array = np.load(f)
    return bar_array


def generator_loop():
    """
    Periodically generates a new bar and puts it into the queue.
    We'll define "half bar" = 2 beats.
    If BPM=60 => half_bar_duration=2sec, so new bar is generated every 2sec.
    """
    print("[Generator] Starting generator_loop...")
    while RUN_GENERATOR:
        local_params = CURRENT_PARAMS
        bar = generate_bar(local_params)
        bar_queue.put(bar)
        half_bar_duration = (60.0 / local_params.bpm) * 2
        print(f"[Generator] Generated bar with params: {local_params.model_dump()}")
        time.sleep(half_bar_duration)
    print("[Generator] Exiting generator_loop...")


##################################
# Lifespan Manager
##################################

generator_thread = None
player_thread = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global generator_thread, player_thread
    global RUN_GENERATOR, RUN_PLAYER

    print("[FastAPI] Lifespan startup. Launching threads...")

    RUN_GENERATOR = True
    RUN_PLAYER = True

    # Start the player thread
    player_thread = threading.Thread(target=bar_player_loop, daemon=True)
    player_thread.start()

    # Start the generator thread
    generator_thread = threading.Thread(target=generator_loop, daemon=True)
    generator_thread.start()

    # Hand over control to FastAPI
    yield

    print("[FastAPI] Lifespan shutdown. Stopping threads...")

    # Signal both threads to stop
    RUN_GENERATOR = False
    RUN_PLAYER = False

    # Put None in the queue to unblock the player thread
    bar_queue.put(None)

    # Wait for threads to fully stop
    if generator_thread is not None:
        generator_thread.join()
    if player_thread is not None:
        player_thread.join()

    print("[FastAPI] Threads have been stopped.")


##################################
# Create the app
##################################

app = FastAPI(lifespan=lifespan)

##################################
# Routes
##################################


@app.post("/set_parameters")
def set_parameters(params: GenerationParams):
    """
    Update the global CURRENT_PARAMS for generation.
    Example JSON:
    {
      "bpm": 80,
      "class1": 0.7,
      "class2": 0.4,
      "class3": 0.2
    }
    """
    global CURRENT_PARAMS
    CURRENT_PARAMS = params
    return {"status": "updated", "params": CURRENT_PARAMS.model_dump()}


@app.get("/status")
def status():
    """
    Returns current BPM and queue size.
    """
    return {"bpm": CURRENT_PARAMS.bpm, "queue_size": bar_queue.qsize()}
