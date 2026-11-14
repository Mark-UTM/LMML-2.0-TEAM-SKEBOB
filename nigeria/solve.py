import wave
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

with wave.open(str(HERE / "challenge.wav")) as w:
    rate = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float)

n, hop = 1024, 256
frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * np.hanning(n)
spec = 20 * np.log10(np.abs(np.fft.rfft(frames, axis=1)).T + 1e-9)

fig, ax = plt.subplots(figsize=(18, 5))
ax.imshow(spec, origin="lower", aspect="auto", cmap="magma", extent=[0, len(x) / rate, 0, rate / 2])
ax.set_xlabel("s")
ax.set_ylabel("Hz")
fig.tight_layout()
fig.savefig(HERE / "spectrogram.png", dpi=100)
print("spectrogram.png: the word in the 3-3.5 kHz band reads TALKINGDRUM -> SIGMOID_TALKINGDRUM")
