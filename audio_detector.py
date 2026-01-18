import time
import argparse
import threading
from collections import deque

import numpy as np
from scipy import signal
from scipy.io import wavfile

try:
    import pyaudio
except Exception:
    pyaudio = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except Exception:
    plt = None
    FuncAnimation = None
    
SR = 48000 # sample rate
CHUNK = 1024 # frames per callback (-21 ms at 48kHz)
HP_CUTOFF = 1200 # high-pass cutoff
HP_ORDER = 4 # for butterworth order

ENV_SMOOTH_MS = 3.0
DERIV_WINDOW_MS = 2.0
THRESH_MULT = 6.0
REFRACTORY_MS = 120.0 # lockout for 15 ms prevent wood vibration detection
LOOKBACK_MS = 8.0

ANALYSIS_WINDOW_MS = 80.0

latest = {"x": None, "y": None, "env": None}
latest_lock = threading.Lock()

def robust_threshold(x, mult = 6.0):
    median = np.median(x)
    mad = np.median(np.abs(x - median)) + 1e-12
    return median + mult * (1.4826 * mad)

b, a = signal.butter(HP_ORDER, HP_CUTOFF, btype='highpass', fs=SR)
zi = signal.lfilter_zi(b, a)

analysis_window_samples = int(SR * ANALYSIS_WINDOW_MS / 1000.0)
ring = deque(maxlen = analysis_window_samples * 2) # ring buffer to hold audio history for t_mech/t_acoustic analysis

# ENVELOPE SMOOTHING
alpha = np.exp(-1.0 / (SR * ENV_SMOOTH_MS / 1000.0))
env_prev = 0.0

deriv_win = max(1, int(SR * DERIV_WINDOW_MS / 1000.0))

# REFRACTORY TRACKING
last_trigger_t = 0.0

# ROOM CONTEXT
hist_seconds = 2.0
hist_len = int((SR / CHUNK) * hist_seconds) # how many chunks are in 2 seconds
deriv_hist = deque(maxlen = hist_len)

# BIMODAL ONSET DETECTOR
def butter_bandpass(lo, hi, fs, order = 4):
    return signal.butter(order, [lo, hi], btype='bandpass', fs=fs) # doing cutoffs, so bandpass filter 

def ema_envelope(input_audio, fs, smooth_ms = 3.0):
    alpha = np.exp(-1.0 / (fs * smooth_ms / 1000.0))
    env = np.empty_like(input_audio, dtype = float)
    e = 0.0
    for i, v in enumerate(np.abs(input_audio)):
        e = alpha * e + (1 - alpha) * v
        env[i] = e
    return env

def find_onset(
    envelope: np.ndarray,
    sample_rate_hz: int,
    derivative_window_ms: float = 2.0,
    threshold_multiplier: float = 6.0,
    search_time_range_s: tuple = (0.0, None),
):

    derivative_window_samples = max(
        1, int(sample_rate_hz * derivative_window_ms / 1000.0) # 2 ms window
    )

    envelope_slope = (
        envelope[derivative_window_samples:]
        - envelope[:-derivative_window_samples]
    )
    positive_slope = np.maximum(envelope_slope, 0.0)

    slope_threshold = robust_threshold(positive_slope, mult=threshold_multiplier)
    trigger_level = 0.5 * slope_threshold  # earlier trigger point since might miss onset if start at slope_threshold

    search_start_sample = int(search_time_range_s[0] * sample_rate_hz)
    if search_time_range_s[1] is None:
        search_end_sample = len(envelope)  # up to end
    else:
        search_end_sample = int(search_time_range_s[1] * sample_rate_hz)

    # adjust because positive_slope is shorter by derivative_window_samples
    slope_search_start = max(0, search_start_sample - derivative_window_samples)
    slope_search_end = min(
        len(positive_slope),
        search_end_sample - derivative_window_samples
    )

    if slope_search_end <= slope_search_start:
        return None

    # after shifting, just look where first index exceeds trigger level
    local_candidates = np.where(
        positive_slope[slope_search_start:slope_search_end] > trigger_level
    )[0]

    if local_candidates.size == 0: # if none exceed trigger level (or no transient found)
        return None

    onset_slope_index = slope_search_start + int(local_candidates[0])

    return onset_slope_index


def detect_t_mech_t_acoustic(
    audio_samples: np.ndarray,
    sample_rate_hz: int,
    window_start_s: float = 0.0,
    window_len_s: float = 0.08,
):

    window_start_sample = int(window_start_s * sample_rate_hz)
    window_end_sample = int((window_start_s + window_len_s) * sample_rate_hz)
    window_audio = audio_samples[window_start_sample:window_end_sample]

    # band-pass filters
    mech_b, mech_a = butter_bandpass(20, 150, sample_rate_hz, order = 4)
    ac_b,   ac_a   = butter_bandpass(3000, 10000, sample_rate_hz, order = 4)

    mech_band_audio = signal.lfilter(mech_b, mech_a, window_audio)
    ac_band_audio   = signal.lfilter(ac_b, ac_a, window_audio)

    # envelopes
    mech_envelope = ema_envelope(mech_band_audio, sample_rate_hz, smooth_ms=0.5)
    ac_envelope   = ema_envelope(ac_band_audio,   sample_rate_hz, smooth_ms=0.5)

    # Onsets inside this window
    mech_onset_sample = find_onset(
        mech_envelope,
        sample_rate_hz,
        derivative_window_ms = 1.0,
        search_time_range_s = (0.0, window_len_s)
    )
    ac_onset_sample = find_onset(
        ac_envelope,
        sample_rate_hz,
        derivative_window_ms = 1.0,
        search_time_range_s = (0.0, window_len_s)
    )

    # converts the mech time or acoustic time to time in seconds
    t_mech_s = None if mech_onset_sample is None else (window_start_s + mech_onset_sample / sample_rate_hz) 
    t_acoustic_s = None if ac_onset_sample is None else (window_start_s + ac_onset_sample / sample_rate_hz)

    delta_t_s = None
    ordering = None

    # find time difference
    if (t_mech_s is not None) and (t_acoustic_s is not None):
        delta_t_s = t_acoustic_s - t_mech_s

        if delta_t_s > 0.00001:
            ordering = "mech_first"
        elif delta_t_s < -0.00001:
            ordering = "acoustic_first"
        else:
            ordering = "simultaneous"

    return t_mech_s, t_acoustic_s, delta_t_s, ordering

class TransientDetector:
    def __init__(self, sr=SR, chunk=CHUNK):
        self.is_triggered = False
        self.sr = sr
        self.chunk = chunk

        self.b_hp, self.a_hp = signal.butter(HP_ORDER, HP_CUTOFF, btype="highpass", fs=sr)
        self.zi = signal.lfilter_zi(self.b_hp, self.a_hp)

        self.alpha = np.exp(-1.0 / (sr * ENV_SMOOTH_MS / 1000.0))
        self.env_prev = 0.0

        self.deriv_win = max(1, int(sr * DERIV_WINDOW_MS / 1000.0))

        self.last_trigger_s = -1e9
        self.refractory_s = REFRACTORY_MS / 1000.0

        self.analysis_window_samples = int(sr * ANALYSIS_WINDOW_MS / 1000.0)
        self.ring = deque(maxlen=self.analysis_window_samples * 2)

        hist_seconds = 2.0
        hist_len = int((sr / chunk) * hist_seconds)
        self.deriv_hist = deque(maxlen=hist_len)
        self.hist_ready_min = max(10, hist_len // 4)
        self.sample_cursor = 0  # total samples processed so far

    def process_chunk(self, x: np.ndarray, print_events=True):

        chunk_start_t = self.sample_cursor / self.sr
        self.sample_cursor += len(x)
        chunk_end_t = self.sample_cursor / self.sr
        now_s = chunk_end_t

        thr = robust_threshold(np.array(self.deriv_hist), mult=THRESH_MULT)
        min_slope_floor = 0.002
        thr = max(thr, min_slope_floor)

        y, self.zi = signal.lfilter(self.b_hp, self.a_hp, x, zi=self.zi)

        abs_y = np.abs(y)
        env = np.empty_like(abs_y)
        e = self.env_prev
        for i, v in enumerate(abs_y):
            e = self.alpha * e + (1 - self.alpha) * v
            env[i] = e
        self.env_prev = float(e)

        with latest_lock:
            latest["x"] = x.copy()
            latest["y"] = y.copy()
            latest["env"] = env.copy()

        self.ring.extend(y.tolist())

        if len(env) > self.deriv_win:
            d = env[self.deriv_win:] - env[:-self.deriv_win]
            dpos = np.maximum(d, 0.0)
            d_peak = float(np.max(dpos)) if dpos.size else 0.0
        else:
            dpos = np.array([], dtype=np.float32)
            d_peak = 0.0

        refractory_ok = (now_s - self.last_trigger_s) >= self.refractory_s

        if refractory_ok:
            self.deriv_hist.append(d_peak)

        if len(self.deriv_hist) >= self.hist_ready_min:
            thr = robust_threshold(np.array(self.deriv_hist), mult=THRESH_MULT)
        else:
            thr = 1e9

        is_hit = refractory_ok and (d_peak > thr) and (not self.is_triggered)

        if is_hit:
            self.is_triggered = True
            self.last_trigger_s = now_s

            onset_sample_in_chunk = 0
            if dpos.size > 0:
                idxs = np.where(dpos > (0.5 * thr))[0]
                if idxs.size > 0:
                    onset_sample_in_chunk = int(idxs[0]) + self.deriv_win

            t_mech = t_ac = delta = ordering = None
            if len(self.ring) >= self.analysis_window_samples:
                recent = np.array(list(self.ring)[-self.analysis_window_samples:], dtype=np.float32)
                t_mech, t_ac, delta, ordering = detect_t_mech_t_acoustic(
                    recent, self.sr, window_len_s=ANALYSIS_WINDOW_MS / 1000.0
                )

            if print_events:
                t_hit_abs = chunk_start_t + (onset_sample_in_chunk / self.sr)

                window_len_s = ANALYSIS_WINDOW_MS / 1000.0
                window_start_t = chunk_end_t - window_len_s

                t_mech_abs = None if t_mech is None else (window_start_t + t_mech)
                t_ac_abs   = None if t_ac   is None else (window_start_t + t_ac)

                delta_ms = None
                if t_mech_abs is not None and t_ac_abs is not None:
                    delta_ms = (t_ac_abs - t_mech_abs) * 1000.0

                print(
                    f"@ {t_hit_abs:.3f}s | "
                    f"t_mech={0.0 if t_mech_abs is None else t_mech_abs:.4f}s "
                    f"t_ac={0.0 if t_ac_abs is None else t_ac_abs:.4f}s "
                    f"Δt={0.0 if delta_ms is None else delta_ms:.1f} ms "
                    f"{ordering}"
                )

            return {"hit": True, "d_peak": d_peak, "thr": thr, "now_s": now_s}
        
        elif self.is_triggered:
            if d_peak < (0.5 * thr):
                self.is_triggered = False
            return {"hit": False, "d_peak": d_peak, "thr": thr, "now_s": now_s}


# PLOTTING FOR LIVE RECORDING
def start_live_plot(chunk=CHUNK, sr=SR):
    if plt is None or FuncAnimation is None:
        print("Matplotlib not available; skipping plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.suptitle("Live Audio Visualization (raw + HP envelope)")

    t_ms = (np.arange(chunk) / sr) * 1000.0
    line_raw, = ax1.plot(t_ms, np.zeros(chunk))
    line_env, = ax2.plot(t_ms, np.zeros(chunk))

    ax1.set_ylabel("Raw x(t)")
    ax2.set_ylabel("Env(|HP(x)|)")
    ax2.set_xlabel("Time (ms)")

    ax1.set_ylim(-1.0, 1.0)
    ax2.set_ylim(0.0, 0.2)

    def update(_):
        with latest_lock:
            x = latest["x"]
            env = latest["env"]

        if x is None or env is None or len(x) != chunk or len(env) != chunk:
            return line_raw, line_env

        line_raw.set_ydata(x)
        line_env.set_ydata(env)

        m = float(np.max(env))
        if m > 1e-6:
            ax2.set_ylim(0.0, max(0.05, min(1.0, 1.2 * m)))

        return line_raw, line_env

    FuncAnimation(fig, update, interval=33, blit=True)
    plt.show()


# LIVE MIC mode and WAV (prerecorded version) mode
def run_live(show_plot=False):
    if pyaudio is None:
        raise RuntimeError("PyAudio not installed, cannot run live mode.")

    det = TransientDetector(sr=SR, chunk=CHUNK)

    if show_plot:
        threading.Thread(target=start_live_plot, daemon=True).start()

    pa = pyaudio.PyAudio()
    try:
        def callback(in_data, frame_count, time_info, status):
            x = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            det.process_chunk(x, print_events=True)
            return (None, pyaudio.paContinue)

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SR,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=callback,
        )

        stream.start_stream()
        print("Listening (live mic)... Ctrl+C to stop.")
        while stream.is_active():
            time.sleep(0.1)

    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        pa.terminate()


def run_wav(path, realtime=False):
    sr_in, data = wavfile.read(path)

    # convert to float32 mono [-1, 1]
    if data.ndim == 2:
        data = data.mean(axis=1)  # stereo -> mono

    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float32) / 2147483648.0
    else:
        # float wav etc.
        x = data.astype(np.float32)
        # normalize if it looks like large ints in float container
        if np.max(np.abs(x)) > 2.0:
            x = x / np.max(np.abs(x))

    # resample if needed
    if sr_in != SR:
        # resample_poly is high quality and fast
        g = np.gcd(sr_in, SR)
        up = SR // g
        down = sr_in // g
        x = signal.resample_poly(x, up, down).astype(np.float32)
        sr_in = SR

    det = TransientDetector(sr=SR, chunk=CHUNK)

    print(f"Running on WAV: {path} (sr={SR}, samples={len(x)})")
    i = 0
    start_wall = time.time()

    while i < len(x):
        chunk = x[i:i+CHUNK]
        if len(chunk) < CHUNK:
            chunk = np.pad(chunk, (0, CHUNK - len(chunk)))

        now_s = (i / SR)
        det.process_chunk(chunk, print_events=True)

        i += CHUNK

        if realtime:
            # play back in real-time speed
            target = start_wall + now_s
            sleep = target - time.time()
            if sleep > 0:
                time.sleep(sleep)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["wav", "live"], required=True)
    ap.add_argument("--wav", type=str, default=None, help="Path to .wav for wav mode")
    ap.add_argument("--realtime", action="store_true", help="In wav mode, sleep to simulate real time")
    ap.add_argument("--plot", action="store_true", help="In live mode, show matplotlib plot")
    args = ap.parse_args()

    if args.mode == "wav":
        if not args.wav:
            raise SystemExit("Provide --wav path/to/file.wav")
        run_wav(args.wav, realtime=args.realtime)
    else:
        run_live(show_plot=args.plot)

if __name__ == "__main__":
    main()

