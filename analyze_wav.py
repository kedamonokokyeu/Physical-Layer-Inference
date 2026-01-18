import numpy as np
from scipy.io import wavfile

from audio_detector import detect_t_mech_t_acoustic

def analyze_wav(path):
    sr, audio = wavfile.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono

    audio = audio.astype(np.float32)
    audio /= np.max(np.abs(audio)) + 1e-9

    print(f"Loaded {path}")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(audio)/sr:.2f}s")

    WINDOW_S = 0.08
    HOP_S = 0.01

    win = int(WINDOW_S * sr)
    hop = int(HOP_S * sr)

    for start in range(0, len(audio) - win, hop):
        segment = audio[start:start + win]

        t_mech, t_ac, dt, ordering = detect_t_mech_t_acoustic(
            segment,
            sr,
            window_start_s=0.0,
            window_len_s=WINDOW_S
        )

        if t_mech is not None and t_ac is not None:
            print(
                f"@ {start/sr:.3f}s | "
                f"t_mech={t_mech:.4f}s "
                f"t_ac={t_ac:.4f}s "
                f"Δt={dt*1000:.1f} ms "
                f"{ordering}"
            )

if __name__ == "__main__":
    analyze_wav("audio_test1.wav")
