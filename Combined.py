"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AviSuite — Unified Aviation Speech Intelligence Platform                   ║
║  Module 1 · Fatigue Analyzer      (DSP fatigue/stress/cognitive)           ║
║  Module 2 · Pilot Whisperer       (Live ATC radio discipline coach)         ║
║  Module 3 · ScenarioSynth         (Generative ATC training engine)          ║
║  ⚠️  Research / Training Use Only · NOT a certified flight-safety device     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.signal import lfilter, find_peaks
from scipy.ndimage import uniform_filter1d
import soundfile as sf
import tempfile, os, time, warnings, json, re, random, subprocess, shutil
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import requests

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CLOUD COMPATIBILITY — detect system capabilities at startup
#  Streamlit Cloud may not have ffmpeg or sounddevice available.
#  These flags gate features gracefully instead of crashing.
# ══════════════════════════════════════════════════════════════════════════════
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None

try:
    import sounddevice as _sd_probe
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AviSuite — Aviation Speech Intelligence",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  PROFESSIONAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@300;400;500&family=Syne:wght@600;700;800&display=swap');

:root {
    --bg-base:       #0c0e14;
    --bg-surface:    #111420;
    --bg-elevated:   #161a28;
    --bg-card:       #1a1f30;
    --border:        #242840;
    --border-active: #3a4060;
    --text-primary:  #e8eaf0;
    --text-secondary:#8890b0;
    --text-muted:    #4a5070;
    --accent-blue:   #4d7cfe;
    --accent-cyan:   #00c8e8;
    --accent-green:  #2dcc8f;
    --accent-amber:  #f0a030;
    --accent-red:    #e84040;
    --accent-violet: #8b5cf6;
    --font-body:     'DM Sans', sans-serif;
    --font-mono:     'DM Mono', monospace;
    --font-display:  'Syne', sans-serif;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 14px;
}

[data-testid="stAppViewContainer"] { background: var(--bg-base) !important; }

[data-testid="stMain"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(77,124,254,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(77,124,254,0.02) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--font-body) !important; color: var(--text-secondary) !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: var(--font-display) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-muted) !important;
}

.suite-header {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 24px 32px 20px;
    position: relative;
    overflow: hidden;
    margin-bottom: 4px;
}
.suite-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent 0%, var(--accent-blue) 40%, var(--accent-cyan) 60%, transparent 100%);
}
.suite-title { font-family: var(--font-display); font-size: 2rem; font-weight: 800; color: var(--text-primary); letter-spacing: 0.06em; line-height: 1; }
.suite-subtitle { font-family: var(--font-mono); font-size: 0.62rem; letter-spacing: 0.22em; text-transform: uppercase; color: var(--text-muted); margin-top: 8px; }
.suite-badge { display: inline-flex; align-items: center; gap: 6px; font-family: var(--font-mono); font-size: 0.58rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--accent-amber); background: rgba(240,160,48,0.08); border: 1px solid rgba(240,160,48,0.2); border-radius: 2px; padding: 3px 10px; margin-top: 10px; }

.stTabs [data-baseweb="tab-list"] { background: var(--bg-surface) !important; border: 1px solid var(--border) !important; border-top: none !important; border-radius: 0 0 6px 6px !important; gap: 0 !important; padding: 0 !important; }
.stTabs [data-baseweb="tab"] { font-family: var(--font-body) !important; font-size: 0.75rem !important; font-weight: 500 !important; letter-spacing: 0.05em !important; text-transform: uppercase !important; color: var(--text-muted) !important; padding: 14px 22px !important; border-right: 1px solid var(--border) !important; border-radius: 0 !important; min-width: 160px; transition: color 0.2s; }
.stTabs [aria-selected="true"] { color: var(--text-primary) !important; background: rgba(77,124,254,0.06) !important; border-bottom: 2px solid var(--accent-blue) !important; }
.stTabs [data-baseweb="tab-panel"] { background: transparent !important; padding: 24px 0 0 !important; }

.section-label { font-family: var(--font-mono); font-size: 0.6rem; font-weight: 500; letter-spacing: 0.25em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 10px; margin-top: 20px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }

.metric-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 18px 16px 14px; position: relative; overflow: hidden; }
.metric-card::after { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: var(--accent, var(--accent-blue)); opacity: 0.7; }
.metric-label { font-family: var(--font-mono); font-size: 0.58rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 6px; }
.metric-value { font-family: var(--font-display); font-size: 2.4rem; font-weight: 700; line-height: 1; color: var(--accent, var(--accent-blue)); }
.metric-sub { font-family: var(--font-mono); font-size: 0.6rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent, var(--accent-blue)); margin-top: 5px; opacity: 0.75; }
.metric-bar-track { background: var(--bg-base); border-radius: 2px; height: 2px; margin-top: 12px; }
.metric-bar-fill { height: 100%; border-radius: 2px; background: var(--accent, var(--accent-blue)); }

.data-panel { background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 16px 18px; margin: 8px 0; }
.data-panel-header { font-family: var(--font-mono); font-size: 0.6rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }

.status-nominal { background: rgba(45,204,143,0.1); border: 1px solid rgba(45,204,143,0.25); border-left: 3px solid var(--accent-green); border-radius: 4px; padding: 10px 14px; font-family: var(--font-mono); font-size: 0.72rem; color: #6eddb8; }
.status-monitor { background: rgba(240,160,48,0.08); border: 1px solid rgba(240,160,48,0.2); border-left: 3px solid var(--accent-amber); border-radius: 4px; padding: 10px 14px; font-family: var(--font-mono); font-size: 0.72rem; color: #f8c060; }
.status-caution { background: rgba(232,80,0,0.08); border: 1px solid rgba(232,80,0,0.2); border-left: 3px solid #e85020; border-radius: 4px; padding: 10px 14px; font-family: var(--font-mono); font-size: 0.72rem; color: #f89060; }
.status-alert   { background: rgba(232,64,64,0.08); border: 1px solid rgba(232,64,64,0.2); border-left: 3px solid var(--accent-red); border-radius: 4px; padding: 10px 14px; font-family: var(--font-mono); font-size: 0.72rem; color: #f08080; }

.atc-box { background: var(--bg-elevated); border: 1px solid var(--border); border-left: 3px solid var(--accent-blue); border-radius: 0 6px 6px 0; padding: 16px 20px; margin: 8px 0; font-family: var(--font-mono); font-size: 0.85rem; line-height: 1.8; color: #b8cce8; }
.pilot-box { background: var(--bg-elevated); border: 1px solid var(--border); border-left: 3px solid var(--accent-green); border-radius: 0 6px 6px 0; padding: 16px 20px; margin: 8px 0; font-family: var(--font-mono); font-size: 0.85rem; line-height: 1.8; color: #a8e8c0; }
.emergency-box { background: rgba(232,64,64,0.04); border: 1px solid rgba(232,64,64,0.2); border-left: 3px solid var(--accent-red); border-radius: 0 6px 6px 0; padding: 16px 20px; margin: 8px 0; font-family: var(--font-mono); font-size: 0.85rem; line-height: 1.8; color: #f0a8a8; }
.feedback-box { background: rgba(45,204,143,0.04); border: 1px solid rgba(45,204,143,0.15); border-left: 3px solid var(--accent-green); border-radius: 0 6px 6px 0; padding: 16px 20px; margin: 8px 0; font-size: 0.85rem; line-height: 1.8; color: #a0d8b8; }
.rl-box { background: rgba(139,92,246,0.04); border: 1px solid rgba(139,92,246,0.15); border-left: 3px solid var(--accent-violet); border-radius: 0 6px 6px 0; padding: 12px 16px; margin: 6px 0; font-size: 0.82rem; color: #c4a8f0; line-height: 1.7; }
.chain-entry { background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 4px; padding: 10px 16px; margin: 4px 0; font-family: var(--font-mono); font-size: 0.8rem; line-height: 1.6; }
.whisper-panel { background: var(--bg-elevated); border: 1px solid var(--border); border-top: 2px solid var(--accent-green); border-radius: 0 0 6px 6px; padding: 16px 18px; margin: 4px 0 8px; }
.info-strip { display: flex; flex-wrap: wrap; gap: 20px; font-family: var(--font-mono); font-size: 0.62rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-muted); padding: 10px 14px; background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 4px; margin: 8px 0 16px; }
.info-strip .val { color: var(--accent-cyan); font-weight: 500; }

.risk-pill { display: inline-flex; align-items: center; gap: 6px; font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.15em; text-transform: uppercase; border-radius: 3px; padding: 4px 12px; font-weight: 600; }
.risk-NOMINAL { background: rgba(45,204,143,0.1); color: var(--accent-green); border: 1px solid rgba(45,204,143,0.3); }
.risk-MONITOR { background: rgba(240,160,48,0.1); color: var(--accent-amber); border: 1px solid rgba(240,160,48,0.3); }
.risk-CAUTION { background: rgba(232,80,0,0.1); color: #f08040; border: 1px solid rgba(232,80,0,0.3); }
.risk-ALERT   { background: rgba(232,64,64,0.1); color: var(--accent-red); border: 1px solid rgba(232,64,64,0.3); }

.score-display { background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 16px 10px; text-align: center; }
.score-num { font-family: var(--font-display); font-size: 2rem; font-weight: 700; color: var(--accent-cyan); line-height: 1; }
.score-lbl { font-family: var(--font-mono); font-size: 0.55rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--text-muted); margin-top: 5px; }

.diff-tag { display: inline-block; font-family: var(--font-mono); font-size: 0.65rem; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; padding: 3px 10px; border-radius: 3px; }
.diff-student  { background: rgba(45,204,143,0.1); color: var(--accent-green); border: 1px solid rgba(45,204,143,0.25); }
.diff-private  { background: rgba(77,124,254,0.1); color: var(--accent-blue);  border: 1px solid rgba(77,124,254,0.25); }
.diff-commercial { background: rgba(240,160,48,0.1); color: var(--accent-amber); border: 1px solid rgba(240,160,48,0.25); }
.diff-expert   { background: rgba(232,64,64,0.1); color: var(--accent-red);   border: 1px solid rgba(232,64,64,0.25); }

div[data-testid="stButton"] > button { background: var(--bg-elevated); border: 1px solid var(--border-active); color: var(--text-secondary); font-family: var(--font-body); font-size: 0.78rem; font-weight: 500; letter-spacing: 0.05em; border-radius: 4px; padding: 9px 18px; transition: all 0.18s; width: 100%; }
div[data-testid="stButton"] > button:hover { background: rgba(77,124,254,0.1); border-color: var(--accent-blue); color: var(--text-primary); }

.stTextArea textarea, .stTextInput input { background: var(--bg-elevated) !important; border: 1px solid var(--border) !important; color: var(--text-primary) !important; font-family: var(--font-mono) !important; font-size: 0.82rem !important; border-radius: 4px !important; }
.stTextArea textarea:focus, .stTextInput input:focus { border-color: var(--accent-blue) !important; box-shadow: 0 0 0 2px rgba(77,124,254,0.12) !important; }
div[data-testid="stSlider"] label, div[data-testid="stSelectbox"] label { font-family: var(--font-mono); font-size: 0.62rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--text-muted) !important; }
div[data-testid="stSelectbox"] > div > div { background: var(--bg-elevated) !important; border-color: var(--border) !important; color: var(--text-primary) !important; font-family: var(--font-body) !important; }

.rec-active { display: inline-flex; align-items: center; gap: 8px; font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--accent-red); background: rgba(232,64,64,0.08); border: 1px solid rgba(232,64,64,0.2); border-radius: 3px; padding: 6px 12px; }
.rec-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent-red); animation: blink 1s infinite; }
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.2;} }

.history-row { background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 4px; padding: 9px 14px; margin: 3px 0; font-size: 0.78rem; font-family: var(--font-mono); color: var(--text-muted); display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }

.live-badge { display: inline-flex; align-items: center; gap: 5px; font-family: var(--font-mono); font-size: 0.6rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--accent-red); background: rgba(232,64,64,0.08); border: 1px solid rgba(232,64,64,0.2); border-radius: 2px; padding: 2px 8px; }
.live-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--accent-red); animation: blink 1.2s infinite; }

.guide-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 18px 20px; margin: 6px 0; font-family: var(--font-mono); font-size: 0.8rem; line-height: 1.9; color: var(--text-secondary); }
.guide-card strong { color: var(--text-primary); display: block; margin-bottom: 10px; font-family: var(--font-display); font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; }

.cloud-warning { background: rgba(240,160,48,0.06); border: 1px solid rgba(240,160,48,0.2); border-left: 3px solid var(--accent-amber); border-radius: 4px; padding: 12px 16px; margin: 8px 0; font-family: var(--font-mono); font-size: 0.72rem; color: #f8c060; line-height: 1.6; }

#MainMenu, footer { visibility: hidden; }
hr { border: none; border-top: 1px solid var(--border); margin: 16px 0; }
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 4px !important; }
.stProgress > div > div { background: var(--accent-blue) !important; }
.miscomm-row { background: rgba(240,160,48,0.05); border-left: 3px solid var(--accent-amber); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
SR       = 22050
FRAME_MS = 25
HOP_MS   = 10
FRAME_N  = int(SR * FRAME_MS / 1000)
HOP_N    = int(SR * HOP_MS   / 1000)
N_MFCC   = 13
N_LPC    = 16
PRE_EMP  = 0.97

MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
MISTRAL_API_URL = st.secrets["MISTRAL_API_URL"]
MISTRAL_MODEL   = "mistral-large-latest"
LIVEATC_URL     = "https://d.liveatc.net/kaus3_app_dep"

RISK_COLORS = {"NOMINAL":"#2dcc8f","MONITOR":"#f0a030","CAUTION":"#e85020","ALERT":"#e84040"}

SCENARIO_TYPES = {
    "Normal Traffic":{"icon":"📻","weight":50},
    "Heavy Traffic" :{"icon":"⚡","weight":30},
    "Emergency"     :{"icon":"🚨","weight":20},
}
DIFFICULTY_LEVELS = {
    "Student Pilot" :{"badge":"diff-student",   "speed":"slow",     "complexity":1,"noise":0.0 },
    "Private Pilot" :{"badge":"diff-private",   "speed":"moderate", "complexity":2,"noise":0.15},
    "Commercial"    :{"badge":"diff-commercial","speed":"fast",      "complexity":3,"noise":0.35},
    "Expert / ATPL" :{"badge":"diff-expert",    "speed":"very fast","complexity":4,"noise":0.6 },
}
AIRPORTS = [
    "KJFK - John F. Kennedy","EGLL - London Heathrow","OMDB - Dubai International",
    "YSSY - Sydney Kingsford Smith","RJTT - Tokyo Haneda","KLAX - Los Angeles",
    "LFPG - Paris CDG","VABB - Mumbai CSIA","FAOR - Johannesburg OR Tambo","SBGR - São Paulo Guarulhos",
]
AIRCRAFT_TYPES = [
    "Boeing 737-800","Airbus A320","Boeing 777-300ER","Cessna 172",
    "Piper PA-28","Airbus A380","Embraer E175","Boeing 787-9",
]


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "m1_results": None, "m1_features": None, "m1_recording": False, "m1_history": [],
        "m2_cycle": 0, "m2_monitoring": False,
        "current_scenario": None, "session_score": 0, "total_attempts": 0,
        "correct_responses": 0, "difficulty": "Student Pilot", "history": [],
        "feedback": None, "session_start": datetime.now().strftime("%H:%M"),
        "streaks": 0, "auto_adapt": True, "show_transcript": False,
        "custom_situation_text": "",
        "rl_q_table": {}, "rl_episode": 0, "rl_weak_areas": {}, "rl_item_errors": {},
        "rl_last_state": None, "rl_last_action": None, "rl_recommendations": [],
        "conv_chain": [], "conv_active": False, "conv_step": 0, "conv_scenario_id": None,
        "conv_pending_eval": False, "conv_render_id": 0, "conv_pilot_first": False,
        "whisper_t1": "", "whisper_chain": "",
        "whisper_t1_last_id": -1, "whisper_chain_last_id": -1,
        "whisper_t1_submit_now": False, "whisper_chain_submit_now": False,
        "chain_type_active": "Normal Traffic",
        "pi_scenario": None, "pi_pilot_call_submitted": False, "pi_pilot_call_eval": None,
        "pi_pilot_call_text": "", "pi_atc_response": None, "pi_readback_eval": None,
        "pi_readback_text": "", "pi_conv_history": [], "pi_step": "init",
        "whisper_pi": "", "whisper_pi_last_id": -1, "whisper_pi_submit_now": False,
        "whisper_pi_rb": "", "whisper_pi_rb_last_id": -1, "whisper_pi_rb_submit_now": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 1 — DSP CORE
# ══════════════════════════════════════════════════════════════════════════════
def pre_emphasis(audio, coef=PRE_EMP):
    return np.append(audio[0], audio[1:] - coef * audio[:-1])

def normalize_audio(audio):
    peak = np.max(np.abs(audio))
    return audio / (peak + 1e-9)

def frame_signal(audio, frame_len, hop):
    n_frames = 1 + (len(audio) - frame_len) // hop
    if n_frames <= 0: return np.zeros((1, frame_len))
    idx = np.arange(frame_len)[None,:] + hop * np.arange(n_frames)[:,None]
    return audio[idx]

def apply_window(frames):
    return frames * np.hamming(frames.shape[1])[None,:]

def power_spectrum(frames, n_fft=512):
    return np.abs(rfft(frames, n=n_fft, axis=1)) ** 2

def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700)
def mel_to_hz(mel): return 700 * (10 ** (mel / 2595) - 1)

def mel_filterbank(sr=SR, n_fft=512, n_mels=40, fmin=80, fmax=8000):
    n_bins = n_fft // 2 + 1
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts  = mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft+1)*hz_pts/sr).astype(int)
    fbank   = np.zeros((n_mels, n_bins))
    for m in range(1, n_mels+1):
        for k in range(bin_pts[m-1], bin_pts[m]):
            fbank[m-1, k] = (k-bin_pts[m-1])/(bin_pts[m]-bin_pts[m-1]+1e-10)
        for k in range(bin_pts[m], min(bin_pts[m+1], n_bins)):
            fbank[m-1, k] = (bin_pts[m+1]-k)/(bin_pts[m+1]-bin_pts[m]+1e-10)
    return fbank

_FBANK_CACHE = {}
def get_fbank(sr=SR, n_fft=512, n_mels=40):
    key = (sr, n_fft, n_mels)
    if key not in _FBANK_CACHE: _FBANK_CACHE[key] = mel_filterbank(sr, n_fft, n_mels)
    return _FBANK_CACHE[key]

def compute_mfcc(audio, sr=SR, n_mfcc=N_MFCC, n_fft=512):
    audio  = pre_emphasis(audio)
    frames = frame_signal(audio, FRAME_N, HOP_N)
    frames = apply_window(frames)
    power  = power_spectrum(frames, n_fft)
    fbank  = get_fbank(sr, n_fft)
    mel_e  = np.dot(power, fbank.T)
    mel_e  = np.where(mel_e < 1e-10, 1e-10, mel_e)
    log_me = np.log(mel_e)
    n_mels = log_me.shape[1]
    dct_mat = np.array([np.cos(np.pi*i/n_mels*(np.arange(n_mels)+0.5)) for i in range(n_mfcc)])
    return (log_me @ dct_mat.T).T

def delta_features(feat, width=2):
    pad   = np.pad(feat, ((0,0),(width,width)), mode='edge')
    denom = 2 * sum(i**2 for i in range(1, width+1))
    delta = np.zeros_like(feat)
    for t in range(feat.shape[1]):
        delta[:,t] = sum(n*(pad[:,t+width+n]-pad[:,t+width-n]) for n in range(1,width+1)) / denom
    return delta

def compute_lpc(frame, order=N_LPC):
    frame = frame * np.hamming(len(frame))
    r = np.correlate(frame, frame, mode='full')
    r = r[len(frame)-1:len(frame)+order]
    if r[0] < 1e-12: return np.zeros(order)
    a, e = np.zeros(order), r[0]
    for i in range(order):
        lam = (-sum(a[j]*r[i-j] for j in range(i)) - r[i+1]) / (e + 1e-12)
        a_new = a.copy(); a_new[i] = lam
        for j in range(i): a_new[j] = a[j] + lam * a[i-1-j]
        a = a_new; e = max(e*(1-lam**2), 1e-12)
    return a

def compute_lpc_features(audio):
    frames  = frame_signal(audio, FRAME_N, HOP_N)
    lpc_all, errors = [], []
    for frame in frames:
        a = compute_lpc(frame); lpc_all.append(a)
        res = lfilter(np.concatenate([[1],a]), [1], frame)
        errors.append(np.mean(res**2))
    lpc_all = np.array(lpc_all); errors = np.array(errors)
    flux = float(np.mean(np.sqrt(np.sum(np.diff(lpc_all,axis=0)**2,axis=1)))) if len(lpc_all)>1 else 0
    return {'lpc_error_mean':float(np.mean(errors)),'lpc_error_std':float(np.std(errors)),'lpc_flux':flux}

def estimate_f0_frame(frame, sr, f0_min=60, f0_max=500):
    frame = frame * np.hamming(len(frame))
    thresh = 0.3 * np.max(np.abs(frame))
    fc = np.where(np.abs(frame) > thresh, frame, 0)
    corr = np.correlate(fc, fc, mode='full')[len(fc)-1:]
    lag_min, lag_max = int(sr/f0_max), min(int(sr/f0_min), len(corr)-1)
    if lag_min >= lag_max or corr[0] < 1e-12: return 0.0
    peaks, _ = find_peaks(corr[lag_min:lag_max], height=0)
    if not len(peaks): return 0.0
    best = peaks[np.argmax(corr[lag_min:lag_max][peaks])] + lag_min
    return float(sr/best) if corr[best]/(corr[0]+1e-12) >= 0.3 else 0.0

def compute_f0_track(audio, sr=SR):
    return np.array([estimate_f0_frame(f, sr) for f in frame_signal(audio, FRAME_N, HOP_N)])

def compute_hnr(audio, sr=SR):
    frames = frame_signal(audio, FRAME_N, HOP_N); hnrs = []
    for frame in frames:
        frame = frame * np.hamming(len(frame))
        corr  = np.correlate(frame, frame, mode='full')[len(frame)-1:]
        if corr[0] < 1e-12: continue
        lmin, lmax = int(sr/500), min(int(sr/60), len(corr)-1)
        if lmin >= lmax: continue
        r = np.clip(np.max(corr[lmin:lmax])/(corr[0]+1e-12), 0, 0.9999)
        hnrs.append(10*np.log10(r/(1-r+1e-12)))
    return float(np.mean(hnrs)) if hnrs else 0.0

def compute_jitter_shimmer(audio, sr=SR):
    f0 = compute_f0_track(audio, sr); voiced = f0[f0>0]
    if len(voiced) < 4:
        return {'jitter_local':0,'jitter_rap':0,'shimmer_local':0,'shimmer_db':0}
    periods = 1.0/voiced
    jitter_local = float(np.mean(np.abs(np.diff(periods)))/(np.mean(periods)+1e-12))
    rap = [abs(periods[i]-np.mean(periods[i-1:i+2])) for i in range(1,len(periods)-1)]
    jitter_rap = float(np.mean(rap)/np.mean(periods)) if rap else 0.0
    amps  = np.sqrt(np.mean(frame_signal(audio,FRAME_N,HOP_N)**2,axis=1))+1e-12
    vamps = amps[:len(f0)][f0>0]
    if len(vamps)<2:
        return {'jitter_local':jitter_local,'jitter_rap':jitter_rap,'shimmer_local':0,'shimmer_db':0}
    sh_local = float(np.mean(np.abs(np.diff(vamps)))/np.mean(vamps))
    return {'jitter_local':jitter_local,'jitter_rap':jitter_rap,
            'shimmer_local':sh_local,'shimmer_db':float(20*np.log10(1+sh_local+1e-12))}

def vad_energy_zcr(audio, sr=SR):
    frames = frame_signal(audio, FRAME_N, HOP_N)
    energy = np.mean(frames**2, axis=1)
    zcr    = np.mean(np.abs(np.diff(np.sign(frames),axis=1)),axis=1)/2
    nf     = max(1, int(0.1*sr/HOP_N))
    e_thr  = np.mean(energy[:nf])*5+1e-12
    z_thr  = np.mean(zcr[:nf])*2+0.05
    return ((energy>e_thr)&(zcr<z_thr)).astype(float)

def compute_spectral_features(audio, sr=SR, n_fft=512):
    frames = frame_signal(audio,FRAME_N,HOP_N)
    fw     = apply_window(frames)
    power  = power_spectrum(fw, n_fft)
    freqs  = rfftfreq(n_fft, 1/sr)
    psum   = power.sum(axis=1)+1e-12
    centroid = (power@freqs)/psum
    spread   = np.sqrt(np.sum(power*(freqs[None,:]-centroid[:,None])**2,axis=1)/psum)
    flux     = np.concatenate([[0],np.sqrt(np.sum(np.diff(power,axis=0)**2,axis=1))])
    cumsum   = np.cumsum(power,axis=1)
    rolloff  = freqs[np.argmax(cumsum>=0.85*cumsum[:,-1:],axis=1)]
    flatness = np.exp(np.mean(np.log(power+1e-12),axis=1))/(np.mean(power,axis=1)+1e-12)
    zcr      = np.mean(np.abs(np.diff(np.sign(frames),axis=1)),axis=1)/2
    ste      = np.mean(frames**2,axis=1)
    return {
        'centroid_mean':float(np.mean(centroid)),'centroid_std':float(np.std(centroid)),
        'spread_mean':float(np.mean(spread)),'flux_mean':float(np.mean(flux)),
        'flux_std':float(np.std(flux)),'rolloff_mean':float(np.mean(rolloff)),
        'flatness_mean':float(np.mean(flatness)),'zcr_mean':float(np.mean(zcr)),
        'zcr_std':float(np.std(zcr)),'ste_mean':float(np.mean(ste)),
        'ste_std':float(np.std(ste)),'ste_dynamic':float(np.max(ste)/(np.mean(ste)+1e-12)),
    }

def compute_prosody(audio, sr=SR):
    voiced = vad_energy_zcr(audio, sr); total = len(voiced)
    speech_ratio = float(np.sum(voiced)/(total+1e-6))
    uv = 1 - voiced; transitions = np.diff(uv.astype(int))
    ps = np.where(transitions==1)[0]; pe = np.where(transitions==-1)[0]
    if len(ps)>0 and len(pe)>0:
        if pe[0]<ps[0]: pe=pe[1:]
        n = min(len(ps),len(pe)); pause_dur = (pe[:n]-ps[:n])*HOP_N/sr
    else: pause_dur = np.array([])
    fds = HOP_N/sr; total_dur = total*fds
    runs, in_run, rs = [], False, 0
    for i,v in enumerate(voiced):
        if v and not in_run: in_run,rs=True,i
        elif not v and in_run: runs.append(i-rs); in_run=False
    if in_run: runs.append(len(voiced)-rs)
    return {
        'speech_ratio':speech_ratio,'pause_count':int(len(pause_dur)),
        'pause_rate':float(len(pause_dur)/(total_dur+1e-6)),
        'pause_mean_dur':float(np.mean(pause_dur)) if len(pause_dur) else 0.0,
        'pause_max_dur':float(np.max(pause_dur)) if len(pause_dur) else 0.0,
        'pause_total_dur':float(np.sum(pause_dur)),
        'voiced_run_mean':float(np.mean(runs)*fds) if runs else 0.0,
        'voiced_run_std':float(np.std(runs)*fds) if runs else 0.0,
        'total_dur':total_dur,
    }

@st.cache_data(show_spinner=False)
def extract_all_features(audio_bytes: bytes, sr: int = SR) -> dict:
    audio  = np.frombuffer(audio_bytes, dtype=np.float32).copy()
    audio  = normalize_audio(audio)
    mfcc   = compute_mfcc(audio, sr)
    mfcc_d = delta_features(mfcc); mfcc_dd = delta_features(mfcc_d)
    f0t    = compute_f0_track(audio, sr); vf0 = f0t[f0t>0]
    js  = compute_jitter_shimmer(audio, sr)
    lpc = compute_lpc_features(audio)
    spec= compute_spectral_features(audio, sr)
    pros= compute_prosody(audio, sr)
    hnr = compute_hnr(audio, sr)
    f0_stats = {
        'f0_mean':float(np.mean(vf0)) if len(vf0)>0 else 0,
        'f0_std':float(np.std(vf0)) if len(vf0)>0 else 0,
        'f0_range':float(np.ptp(vf0)) if len(vf0)>1 else 0,
        'f0_slope':float(np.polyfit(np.arange(len(vf0)),vf0,1)[0]) if len(vf0)>2 else 0,
        'voiced_ratio':float(np.sum(f0t>0)/(len(f0t)+1e-6)),
    }
    mfcc_feats = {}
    for i in range(N_MFCC):
        mfcc_feats[f'mfcc{i+1}_mean']=float(np.mean(mfcc[i]))
        mfcc_feats[f'mfcc{i+1}_std'] =float(np.std(mfcc[i]))
        mfcc_feats[f'mfccD{i+1}_mean']=float(np.mean(mfcc_d[i]))
        mfcc_feats[f'mfccDD{i+1}_mean']=float(np.mean(mfcc_dd[i]))
    return {**f0_stats,**js,**lpc,**spec,**pros,**mfcc_feats,'hnr':hnr,
            '_f0_track':f0t.tolist(),'_mfcc':mfcc[:5].tolist(),'_sr':sr}

def sigmoid(x, center, scale):
    return float(1/(1+np.exp(-(x-center)/scale)))

def compute_indicators(f: dict, role: str="Pilot") -> dict:
    f0_center = 140 if "Male" in role else 200 if "Female" in role else 165
    fat={}
    fat['vocal_energy']    = (1-np.clip(f['ste_mean']*500,0,1))*20
    fat['hnr_breathiness'] = (1-sigmoid(f['hnr'],12,4))*18
    fat['shimmer']         = np.clip(f['shimmer_db']/2.5,0,1)*16
    fat['pitch_flatness']  = (1-np.clip(f['f0_range']/150,0,1))*15
    fat['pause_length']    = np.clip(f['pause_mean_dur']/1.2,0,1)*15
    fat['speech_rate']     = (1-np.clip(f['speech_ratio']/0.65,0,1))*8
    fat['voiced_ratio']    = (1-np.clip(f['voiced_ratio']/0.55,0,1))*8
    fatigue = float(np.clip(sum(fat.values()),0,100))
    stre={}
    stre['f0_elevation']  = sigmoid(f['f0_mean'],f0_center+30,25)*22
    stre['f0_variability']= np.clip(f['f0_std']/45,0,1)*15
    stre['jitter']        = np.clip(f['jitter_local']/0.015,0,1)*16
    stre['spectral_flux'] = np.clip(f['flux_mean']*4000,0,1)*13
    stre['energy_var']    = np.clip(f['ste_std']*600,0,1)*12
    stre['centroid_hi']   = sigmoid(f['centroid_mean'],2200,500)*12
    stre['lpc_flux']      = np.clip(f['lpc_flux']*60,0,1)*10
    stress = float(np.clip(sum(stre.values()),0,100))
    cog={}
    cog['hesitation_rate'] = np.clip(f['pause_rate']*2.5,0,1)*22
    cog['rhythm_irregular']= np.clip(f['voiced_run_std']/0.45,0,1)*18
    cog['short_voiced_runs']=(1-np.clip(f['voiced_run_mean']/0.7,0,1))*15
    cog['lpc_error']       = np.clip(f['lpc_error_std']*1200,0,1)*14
    cog['spectral_noise']  = np.clip(f['flatness_mean']*6,0,1)*14
    cog['jitter_rap']      = np.clip(f['jitter_rap']/0.012,0,1)*10
    cog['mfcc_delta']      = np.clip(abs(f.get('mfccD1_mean',0))/4,0,1)*7
    cognitive = float(np.clip(sum(cog.values()),0,100))
    clarity_raw = (sigmoid(f['hnr'],10,5)*35 + (1-np.clip(f['jitter_local']/0.02,0,1))*25
                   + np.clip(f['speech_ratio']/0.6,0,1)*20 + (1-np.clip(f['flatness_mean']*5,0,1))*20)
    rt_clarity = float(np.clip(clarity_raw,0,100))
    conf = float(np.clip(f['voiced_ratio']*f.get('total_dur',5)/6, 0.35, 1.0))
    composite = 0.35*fatigue + 0.30*stress + 0.35*cognitive
    if composite<30: risk_level="NOMINAL"
    elif composite<50: risk_level="MONITOR"
    elif composite<70: risk_level="CAUTION"
    else: risk_level="ALERT"
    return {
        'fatigue':fatigue,'stress':stress,'cognitive':cognitive,'rt_clarity':rt_clarity,
        'composite':composite,'risk_level':risk_level,'confidence':conf*100,
        'fat_subs':{k:min(v*5,100) for k,v in fat.items()},
        'stre_subs':{k:min(v*5,100) for k,v in stre.items()},
        'cog_subs':{k:min(v*5,100) for k,v in cog.items()},
        'f0_mean':f['f0_mean'],'f0_std':f['f0_std'],'f0_slope':f['f0_slope'],
        'hnr':f['hnr'],'jitter':f['jitter_local']*100,'shimmer_db':f['shimmer_db'],
        'speech_ratio':f['speech_ratio']*100,'pause_rate':f['pause_rate'],
        'voiced_ratio':f['voiced_ratio']*100,'centroid':f['centroid_mean'],
        'lpc_flux':f['lpc_flux'],'pause_mean':f['pause_mean_dur'],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 1 — PLOTS
# ══════════════════════════════════════════════════════════════════════════════
PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,20,32,0.8)',
    font=dict(family='DM Mono, monospace', color='#4a5070', size=9),
    margin=dict(l=10,r=10,t=35,b=10),
)

def sev_color(score):
    if score<30: return '#2dcc8f'
    if score<50: return '#f0a030'
    if score<70: return '#e85020'
    return '#e84040'

def severity_label(score):
    if score<30: return 'NOMINAL'
    if score<50: return 'MONITOR'
    if score<70: return 'CAUTION'
    return 'ALERT'

def fig_radar(ind):
    cats=['Fatigue','Stress','Cog. Load','RT Clarity']
    vals=[ind['fatigue'],ind['stress'],ind['cognitive'],ind['rt_clarity']]
    fig = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]],
        fill='toself', fillcolor='rgba(77,124,254,0.06)',
        line=dict(color='#4d7cfe',width=1.5),
        marker=dict(color=[sev_color(v) for v in vals]+[sev_color(vals[0])],size=6),
    ))
    fig.update_layout(
        polar=dict(bgcolor='rgba(17,20,32,0.8)',
            radialaxis=dict(visible=True,range=[0,100],gridcolor='#1e2236',color='#3a4060',tickfont=dict(size=8)),
            angularaxis=dict(gridcolor='#1e2236',color='#4a5a8a',tickfont=dict(size=9)),
        ), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Mono',color='#4a6080',size=9),
        height=260, showlegend=False, margin=dict(l=50,r=50,t=20,b=20))
    return fig

def fig_f0_track(f0_track, sr=SR):
    f0 = np.array(f0_track); t = np.arange(len(f0))*HOP_N/sr; vm = f0>0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t,y=np.where(vm,f0,np.nan),mode='lines',
        line=dict(color='#4d7cfe',width=1.5),name='F0 (Hz)',fill='tozeroy',fillcolor='rgba(77,124,254,0.05)'))
    if np.sum(vm)>10:
        fi=np.interp(t,t[vm],f0[vm]); sm=uniform_filter1d(fi,size=35)
        fig.add_trace(go.Scatter(x=t,y=sm,mode='lines',line=dict(color='#00c8e8',width=1.5,dash='dot'),name='Trend'))
    fig.update_layout(**PLOTLY_THEME,height=185,
        xaxis=dict(title='Time (s)',gridcolor='#1e2236',zerolinecolor='#1e2236'),
        yaxis=dict(title='F0 (Hz)',gridcolor='#1e2236',zerolinecolor='#1e2236'),
        title=dict(text='Fundamental Frequency Track',font=dict(size=10,color='#4d7cfe')),
        legend=dict(orientation='h',y=1.12,font=dict(size=8)))
    return fig

def fig_mfcc_heatmap(mfcc):
    fig = go.Figure(go.Heatmap(z=np.array(mfcc),
        colorscale=[[0,'#0c0e14'],[0.35,'#111420'],[0.7,'#1a2a5a'],[1,'#4d7cfe']],
        showscale=True,
        colorbar=dict(thickness=6,outlinewidth=0,tickfont=dict(size=8,color='#3a4060'))))
    fig.update_layout(**PLOTLY_THEME,height=140,
        xaxis=dict(title='Frame',gridcolor='#1e2236',zerolinecolor='#1e2236'),
        yaxis=dict(title='MFCC',gridcolor='#1e2236',zerolinecolor='#1e2236'),
        title=dict(text='MFCC Coefficients 1–5',font=dict(size=10,color='#4d7cfe')))
    return fig

def fig_sub_breakdown(subs, title, color):
    labels = [k.replace('_',' ').title() for k in subs]
    values = list(subs.values())
    fig = go.Figure(go.Bar(x=values,y=labels,orientation='h',
        marker=dict(color=values,colorscale=[[0,'#111420'],[0.5,'rgba(232,64,64,0.4)'],[1,color]],line=dict(width=0)),
        text=[f"{v:.0f}" for v in values],textposition='outside',textfont=dict(size=8,color='#4a5070')))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(17,20,32,0.8)',
        font=dict(family='DM Mono',color='#4a5070',size=8),height=215,
        xaxis=dict(range=[0,115],gridcolor='#1e2236',zerolinecolor='#1e2236'),
        yaxis=dict(gridcolor='#1e2236',tickfont=dict(size=7)),
        title=dict(text=title,font=dict(size=10,color=color)),margin=dict(l=10,r=40,t=35,b=10))
    return fig

def fig_session_trend(history):
    if len(history)<2: return None
    df = pd.DataFrame(history); fig = go.Figure()
    cfg=[('fatigue','#e84040','Fatigue'),('stress','#f0a030','Stress'),
         ('cognitive','#8b5cf6','Cog.Load'),('rt_clarity','#2dcc8f','RT Clarity')]
    for col,color,name in cfg:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=list(range(len(df))),y=df[col],mode='lines+markers',
                line=dict(color=color,width=1.5),marker=dict(size=4,color=color),name=name))
    fig.update_layout(**PLOTLY_THEME,height=200,
        xaxis=dict(title='Session #',gridcolor='#1e2236',zerolinecolor='#1e2236'),
        yaxis=dict(title='Score',range=[0,100],gridcolor='#1e2236',zerolinecolor='#1e2236'),
        title=dict(text='Session Trend',font=dict(size=10,color='#4d7cfe')),
        legend=dict(orientation='h',y=1.15,font=dict(size=8)))
    return fig

def fig_risk_gauge(composite):
    color = sev_color(composite)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=composite, domain={'x':[0,1],'y':[0,1]},
        number={'font':{'family':'Syne','color':color,'size':26}},
        gauge={'axis':{'range':[0,100],'tickcolor':'#2a3050','tickfont':{'size':8}},
               'bar':{'color':color,'thickness':0.22},'bgcolor':'#111420',
               'borderwidth':1,'bordercolor':'#1e2236',
               'steps':[{'range':[0,30],'color':'rgba(45,204,143,0.08)'},
                        {'range':[30,50],'color':'rgba(240,160,48,0.08)'},
                        {'range':[50,70],'color':'rgba(232,80,32,0.08)'},
                        {'range':[70,100],'color':'rgba(232,64,64,0.1)'}],
               'threshold':{'line':{'color':color,'width':2},'thickness':0.8,'value':composite}}))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Mono',color='#4a5070'),height=185,margin=dict(l=20,r=20,t=20,b=10))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 1 — RENDER HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def render_metric_card(label, value_str, sub, desc, bar_pct, score):
    accent = sev_color(score)
    pct = min(bar_pct, 100)
    st.markdown(f"""
    <div class="metric-card" style="--accent:{accent}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value_str}</div>
        <div class="metric-sub">{sub}</div>
        <div class="metric-bar-track"><div class="metric-bar-fill" style="width:{pct:.0f}%;background:{accent}"></div></div>
        <div style="font-family:var(--font-mono);font-size:0.6rem;color:var(--text-muted);margin-top:7px;line-height:1.5">{desc}</div>
    </div>""", unsafe_allow_html=True)

def generate_aviation_report(ind, feats, role):
    f,s,c = ind['fatigue'],ind['stress'],ind['cognitive']
    rl = ind['risk_level']
    rc = RISK_COLORS[rl]
    risk_body = {"NOMINAL":f"Composite score {ind['composite']:.0f}/100 — within normal parameters. Voice acoustic profile consistent with alert, rested flight crew.",
                 "MONITOR":f"Composite {ind['composite']:.0f}/100 trending above baseline. No immediate action required; supervisor situational awareness recommended.",
                 "CAUTION":f"Composite {ind['composite']:.0f}/100 — meaningful degradation detected across multiple speech dimensions. Recommend crew welfare check.",
                 "ALERT":f"Composite {ind['composite']:.0f}/100 — significant deviation across acoustic channels. ICAO fatigue risk management protocols recommended."}[rl]
    st.markdown(f"""
    <div class="data-panel" style="border-top:2px solid {rc};">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
            <span class="risk-pill risk-{rl}">{rl}</span>
            <span style="font-family:var(--font-mono);font-size:0.6rem;color:var(--text-muted)">Composite {ind['composite']:.0f}/100</span>
        </div>
        <div style="font-family:var(--font-mono);font-size:0.75rem;color:var(--text-secondary);line-height:1.7">{risk_body}</div>
    </div>""", unsafe_allow_html=True)
    items = [
        ("RT Clarity", ind['rt_clarity'], f"HNR {ind['hnr']:.1f} dB · Jitter {ind['jitter']:.2f}% · " +
          ("Readback likely intelligible." if ind['rt_clarity']>=70 else "Some breathiness. ATC may request repetition." if ind['rt_clarity']>=45 else "Compromised intelligibility. Risk of readback errors.")),
        ("Fatigue Index", f, f"Avg pause {ind['pause_mean']:.2f}s · Shimmer {ind['shimmer_db']:.2f} dB · HNR {ind['hnr']:.1f} dB"),
        ("Stress / Arousal", s, f"F0 {ind['f0_mean']:.0f} Hz (±{ind['f0_std']:.0f}) · Jitter {ind['jitter']:.2f}% · Flux {ind['lpc_flux']:.3f}"),
        ("Cognitive Load", c, f"Pause rate {ind['pause_rate']:.2f}/s · Rhythm σ {feats.get('voiced_run_std',0):.2f}s · LPC err {feats.get('lpc_error_std',0):.4f}"),
    ]
    col_a, col_b = st.columns(2)
    for i,(label,score,desc) in enumerate(items):
        sev = severity_label(score); color = sev_color(score)
        card = f"""<div style="background:var(--bg-card);border:1px solid var(--border);border-left:2px solid {color};
        border-radius:4px;padding:14px 16px;margin:5px 0">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <span style="font-family:var(--font-mono);font-size:0.62rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--text-muted)">{label}</span>
        <span style="font-family:var(--font-display);font-size:1.1rem;font-weight:700;color:{color}">{score:.0f}</span>
        </div>
        <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);line-height:1.6">{desc}</div>
        </div>"""
        with (col_a if i%2==0 else col_b):
            st.markdown(card, unsafe_allow_html=True)
    st.markdown("""<div style="margin-top:14px;font-family:var(--font-mono);font-size:0.6rem;color:var(--text-muted);
    border:1px solid var(--border);border-radius:4px;padding:10px 14px;background:var(--bg-elevated)">
    ⚠ DISCLAIMER — Experimental acoustic indicators for research and situational awareness only.
    NOT a certified airworthiness or fitness-for-duty determination. All flight safety decisions must follow
    FAR/EASA regulations, operator SOPs and qualified authority assessment per ICAO Doc 9966.
    </div>""", unsafe_allow_html=True)

def load_audio_file(path, sr=SR):
    audio, fsr = sf.read(path, dtype='float32', always_2d=False)
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if fsr != sr: audio = signal.resample(audio, int(len(audio)*sr/fsr))
    return audio


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 2 — PILOT WHISPERER HELPERS
#  FIX: ffmpeg calls wrapped with FFMPEG_AVAILABLE guard + timeout + returncode check
# ══════════════════════════════════════════════════════════════════════════════
def capture_audio_stream(url, seconds):
    """Capture live ATC stream. Raises RuntimeError if ffmpeg unavailable."""
    if not FFMPEG_AVAILABLE:
        raise RuntimeError(
            "ffmpeg is not installed on this server.\n"
            "Add 'ffmpeg' to your packages.txt file and redeploy."
        )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        path = f.name
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", url, "-t", str(seconds), "-vn",
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=seconds + 30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {result.returncode}. "
                "Check LIVEATC_URL in your Streamlit secrets."
            )
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg capture timed out. Check your LIVEATC_URL.")
    return path

def mistral_chat(prompt: str, max_tokens: int=1200, stream: bool=False):
    headers={"Authorization":f"Bearer {MISTRAL_API_KEY}","Content-Type":"application/json"}
    payload={"model":MISTRAL_MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"stream":stream}
    r=requests.post(MISTRAL_API_URL,headers=headers,json=payload,stream=stream,timeout=60)
    r.raise_for_status(); return r

def mistral_cleanup(text):
    prompt=f"""You are analyzing live ATC radio communication.
Phraseology rules: ATC gives clearances/headings/altitudes. Pilot reads back.
Tasks:
1. Split into [ATC] and [PILOT – Callsign]
2. Correct obvious recognition errors only
3. Compare ATC instructions vs pilot readbacks
4. Flag mismatches with ⚠ POSSIBLE MISCOMM
Output: each transmission on new line. End with (Confidence: XX%)
Transcript:\n{text}"""
    try:
        r=requests.post(MISTRAL_API_URL,
            headers={"Authorization":f"Bearer {MISTRAL_API_KEY}","Content-Type":"application/json"},
            json={"model":MISTRAL_MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":800},timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Mistral error: {e}]\n\n{text}"

def mistral_discipline(cleaned_text):
    prompt=f"""You are an ATC radio discipline coach. Analyze this labeled ATC transcript.
Respond with JSON only (no markdown):
{{"score":<0-100>,"issues":[{{"type":"<MISSING ELEMENT|WRONG READBACK|MISCOMM FLAG|PHRASEOLOGY ERROR>","severity":"<HIGH|MEDIUM|LOW>","quote":"<exact pilot transmission>","description":"<what was wrong>","tip":"<how to fix>"}}],"miscomm_flags":<int>,"confidence":"<XX%>","clearance":"<CLEAR|ISSUES DETECTED>","coach_summary":"<2-3 sentence coaching note>"}}
If no issues, return empty issues array and score 100.
Transcript:\n{cleaned_text}"""
    try:
        r=requests.post(MISTRAL_API_URL,
            headers={"Authorization":f"Bearer {MISTRAL_API_KEY}","Content-Type":"application/json"},
            json={"model":MISTRAL_MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":600},timeout=30)
        r.raise_for_status()
        raw=r.json()["choices"][0]["message"]["content"].strip()
        raw=re.sub(r"^```json|^```|```$","",raw,flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception as e:
        return {"score":0,"issues":[],"miscomm_flags":0,"confidence":"—","clearance":"—",
                "coach_summary":f"Analysis error: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 3 — WHISPER / STT HELPERS
#  FIX: All ffmpeg calls guarded; graceful fallback when unavailable
# ══════════════════════════════════════════════════════════════════════════════
def _ffmpeg_to_16k_mono(audio_path: str) -> str:
    """
    Convert audio to 16 kHz mono WAV using ffmpeg.
    Returns original path unchanged if ffmpeg is unavailable or conversion fails.
    Whisper can handle the raw browser WebM/WAV directly.
    """
    if not FFMPEG_AVAILABLE:
        return audio_path  # Whisper handles many formats natively
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        out_path = f.name
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=60,
        )
        if result.returncode != 0:
            # Conversion failed; use original
            try: os.unlink(out_path)
            except: pass
            return audio_path
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        try: os.unlink(out_path)
        except: pass
        return audio_path
    return out_path

def _segment_by_silence(whisper_result: dict, gap_threshold: float = 0.6) -> list:
    segments = whisper_result.get("segments", [])
    if not segments:
        return [whisper_result.get("text", "").strip()]
    words = []
    for seg in segments:
        for w in seg.get("words", []):
            word = w.get("word", "").strip()
            if word:
                words.append({"word": word, "start": w.get("start", 0), "end": w.get("end", 0)})
    if not words:
        turns, current = [], segments[0]["text"].strip()
        for i in range(1, len(segments)):
            gap = segments[i]["start"] - segments[i - 1]["end"]
            if gap >= gap_threshold:
                if current: turns.append(current)
                current = segments[i]["text"].strip()
            else:
                current += " " + segments[i]["text"].strip()
        if current: turns.append(current)
        return turns or [whisper_result.get("text", "").strip()]
    turns, current = [], words[0]["word"]
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap >= gap_threshold:
            turns.append(current.strip())
            current = words[i]["word"]
        else:
            current += " " + words[i]["word"]
    if current.strip():
        turns.append(current.strip())
    return turns or [whisper_result.get("text", "").strip()]

def transcribe_audio(audio_path: str) -> str:
    import whisper as whisper_lib
    # Attempt ffmpeg conversion; fall back gracefully
    try:
        converted = _ffmpeg_to_16k_mono(audio_path)
    except Exception:
        converted = audio_path

    # Verify the converted file exists and is non-trivial
    try:
        if not os.path.exists(converted) or os.path.getsize(converted) < 500:
            converted = audio_path
    except Exception:
        converted = audio_path

    try:
        model = whisper_lib.load_model("large")  # Use 'base' on cloud (faster/lighter)
        result = model.transcribe(
            converted, fp16=False, language="en",
            condition_on_previous_text=False, word_timestamps=True,
        )
    except Exception as e:
        raise RuntimeError(f"Whisper transcription failed: {e}")
    finally:
        # Clean up temp conversion file
        try:
            if converted != audio_path and os.path.exists(converted):
                os.unlink(converted)
        except Exception:
            pass

    turns = _segment_by_silence(result)
    st.session_state["_last_whisper_turns"] = turns
    return "\n".join(turns).strip() if len(turns) > 1 else result["text"].strip()

def mistral_aviation_correct(raw_transcript: str) -> str:
    turns = st.session_state.get("_last_whisper_turns", None)
    if turns and len(turns) > 1:
        turn_block = "\n".join(f"[TURN {i+1}]\n{t}\n" for i, t in enumerate(turns))
        input_context = (f"The recording was split into {len(turns)} transmissions.\n{turn_block}")
    else:
        input_context = f"Single transmission:\n{raw_transcript}"
    prompt = f"""Post-process ASR transcript of a PILOT READBACK in aviation radio.
RULES: Preserve wording. Only fix obvious ASR errors.
Fix: 'India 123' → 'Air India 123', numbers, frequencies, phraseology.
Return ONLY the corrected readback text — no labels, no explanations.
{input_context}"""
    try:
        r = requests.post(MISTRAL_API_URL,
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
            json={"model": MISTRAL_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 200},
            timeout=20)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return raw_transcript

def whisper_readback_widget(state_key: str, widget_key: str, label: str = "Voice Readback"):
    """
    Voice readback widget using streamlit-mic-recorder.
    Fully cloud-compatible: ffmpeg used when available, skipped otherwise.
    Whisper model is 'base' for speed on Streamlit Cloud.
    """
    id_key     = f"{state_key}_last_id"
    submit_key = f"{state_key}_submit_now"

    try:
        from streamlit_mic_recorder import mic_recorder
    except ImportError:
        st.markdown("""<div class="cloud-warning">
        ⚠ streamlit-mic-recorder not installed.<br>
        Add <code>streamlit-mic-recorder</code> to requirements.txt
        </div>""", unsafe_allow_html=True)
        return st.session_state.get(state_key, "")

    # Check whisper availability
    whisper_available = False
    try:
        import whisper as _wtest
        whisper_available = True
    except ImportError:
        pass

    st.markdown(f'<div class="whisper-panel">', unsafe_allow_html=True)
    st.markdown(f"""<div style="font-family:var(--font-mono);font-size:0.62rem;letter-spacing:0.18em;
    text-transform:uppercase;color:var(--accent-green);margin-bottom:6px">{label}</div>""",
    unsafe_allow_html=True)

    if not whisper_available:
        st.markdown("""<div class="cloud-warning">
        ⚠ openai-whisper not installed. Add <code>openai-whisper</code> to requirements.txt.<br>
        You can still type your readback manually below.
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return st.session_state.get(state_key, "")

    ffmpeg_note = "" if FFMPEG_AVAILABLE else " · ffmpeg unavailable, using direct decode"
    st.caption(f"Start → speak → Stop  ·  Whisper STT + Mistral aviation correction{ffmpeg_note}")

    audio = mic_recorder(
        start_prompt="● Start Recording",
        stop_prompt="■ Stop & Transcribe",
        key=widget_key
    )

    if audio:
        current_id = audio.get("id", 0)
        if current_id != st.session_state.get(id_key, -1):
            st.session_state[id_key] = current_id
            st.session_state[submit_key] = False
            with st.spinner("Transcribing with Whisper..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                        f.write(audio["bytes"])
                        audio_path = f.name
                    whisper_raw = transcribe_audio(audio_path)
                    try: os.unlink(audio_path)
                    except: pass
                except Exception as e:
                    st.error(f"Transcription error: {e}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    return st.session_state.get(state_key, "")

            with st.spinner("Applying aviation correction..."):
                try:
                    corrected = mistral_aviation_correct(whisper_raw)
                except Exception:
                    corrected = whisper_raw
                st.session_state[state_key] = corrected

            st.markdown(f"""<div style="background:rgba(45,204,143,0.06);border:1px solid rgba(45,204,143,0.2);
            border-radius:4px;padding:10px 14px;margin:6px 0;font-family:var(--font-mono);font-size:0.8rem;color:#a0d8b8">
            ✓ <strong>Corrected:</strong> {corrected}</div>""", unsafe_allow_html=True)
            if whisper_raw.lower().strip() != corrected.lower().strip():
                st.caption(f'Whisper heard: "{whisper_raw}"')
        else:
            stored = st.session_state.get(state_key, "")
            if stored:
                st.markdown(f"""<div style="background:rgba(45,204,143,0.06);border:1px solid rgba(45,204,143,0.2);
                border-radius:4px;padding:10px 14px;margin:6px 0;font-family:var(--font-mono);font-size:0.8rem;color:#a0d8b8">
                ✓ <strong>Corrected:</strong> {stored}</div>""", unsafe_allow_html=True)

    stored_transcript = st.session_state.get(state_key, "")
    if stored_transcript:
        if st.button("Submit Voice Readback", use_container_width=True,
                     key=f"{widget_key}_voice_submit"):
            st.session_state[submit_key] = True

    st.markdown('</div>', unsafe_allow_html=True)
    return stored_transcript


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 3 — RL ENGINE
# ══════════════════════════════════════════════════════════════════════════════
RL_ALPHA,RL_GAMMA,RL_EPSILON=0.3,0.7,0.2

def rl_get_state():
    diff_idx=list(DIFFICULTY_LEVELS.keys()).index(st.session_state.difficulty)
    weak=st.session_state.rl_weak_areas
    top_weak=min(weak.items(),key=lambda x:x[1]["attempts"]/max(x[1]["errors"]+0.1,1))[0] if weak else "none"
    return (diff_idx,top_weak)

def rl_choose_action(state):
    actions=["Normal Traffic","Heavy Traffic","Emergency"]; qt=st.session_state.rl_q_table
    if random.random()<RL_EPSILON or state not in qt: return random.choice(actions)
    return max(actions,key=lambda a:qt.get(state,{}).get(a,0.0))

def rl_update(state,action,reward,next_state):
    actions=["Normal Traffic","Heavy Traffic","Emergency"]; qt=st.session_state.rl_q_table
    for s in [state,next_state]:
        if s not in qt: qt[s]={a:0.0 for a in actions}
    old=qt[state].get(action,0.0)
    qt[state][action]=old+RL_ALPHA*(reward+RL_GAMMA*max(qt[next_state].values())-old)
    st.session_state.rl_q_table=qt

def rl_update_weak_areas(scenario_type,items_missed,items_incorrect):
    weak=st.session_state.rl_weak_areas
    if scenario_type not in weak: weak[scenario_type]={"attempts":0,"errors":0}
    weak[scenario_type]["attempts"]+=1; weak[scenario_type]["errors"]+=len(items_missed)+len(items_incorrect)
    st.session_state.rl_weak_areas=weak
    ie=st.session_state.rl_item_errors
    for item in items_missed+items_incorrect: ie[item]=ie.get(item,0)+1
    st.session_state.rl_item_errors=ie

def rl_generate_recommendations():
    recs=[]
    for stype,stats in st.session_state.rl_weak_areas.items():
        if stats["attempts"]>=2:
            er=stats["errors"]/max(stats["attempts"],1)
            if er>1.5: recs.append(f"Focus area: **{stype}** — {er:.1f} errors/attempt")
    se=sorted(st.session_state.rl_item_errors.items(),key=lambda x:x[1],reverse=True)[:3]
    for item,cnt in se: recs.append(f"Often missed: **{item}** ({cnt}×)")
    st.session_state.rl_recommendations=recs; return recs

def weighted_scenario_type():
    types=list(SCENARIO_TYPES.keys())
    return random.choices(types,weights=[SCENARIO_TYPES[t]["weight"] for t in types])[0]

def adaptive_difficulty():
    if not st.session_state.auto_adapt or st.session_state.total_attempts<3: return
    acc=st.session_state.correct_responses/max(st.session_state.total_attempts,1)
    levels=list(DIFFICULTY_LEVELS.keys()); idx=levels.index(st.session_state.difficulty)
    if acc>0.85 and idx<len(levels)-1: st.session_state.difficulty=levels[idx+1]
    elif acc<0.50 and idx>0: st.session_state.difficulty=levels[idx-1]

AVIATE_RULES="""
━━━━ MANDATORY AVIATION RADIO PHRASEOLOGY ━━━━
FREQUENCIES: "121.3" or "119.05"
HEADINGS: "270" or "090"
ALTITUDES: "4,000 feet" or "FL350"
RUNWAYS: "Runway 27L" or "Runway 09R"
SQUAWK: "squawk 7700"
FLIGHT NUMBERS: "Air India 123"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

def aviate_numbers(text):
    def dw(d): return {"0":"zero","1":"one","2":"two","3":"tree","4":"four","5":"fife","6":"six","7":"seven","8":"eight","9":"niner"}.get(d,d)
    text=re.sub(r'\bFL\s*(\d{2,3})\b',lambda m:"flight level "+" ".join(dw(d) for d in m.group(1)),text,flags=re.IGNORECASE)
    text=re.sub(r'\b(\d{3})\.(\d{1,2})\b',lambda m:" ".join(dw(d) for d in m.group(1))+" decimal "+" ".join(dw(d) for d in m.group(2).rstrip("0") or m.group(2)),text)
    text=re.sub(r'\brunway\s+(\d{1,2})([LRC]?)\b',lambda m:"runway "+" ".join(dw(d) for d in m.group(1))+{"L":" left","R":" right","C":" centre"}.get(m.group(2).upper(),""),text,flags=re.IGNORECASE)
    return text

def generate_scenario(airport,aircraft,scenario_type,difficulty,custom_situation=None):
    complexity=DIFFICULTY_LEVELS[difficulty]["complexity"]
    pilot_first=scenario_type=="Emergency" and not custom_situation
    situation_context=f"\nSPECIFIC SITUATION: {custom_situation}\n" if custom_situation else ""
    prompt=f"""Senior ATC instructor. Create realistic radio training scenario.
Airport: {airport} | Aircraft: {aircraft} | Type: {scenario_type} | Difficulty: {difficulty} (complexity {complexity}/4)
{situation_context}{AVIATE_RULES}
Return ONLY JSON:
{{"scenario_id":"SC{random.randint(1000,9999)}","scenario_type":"{scenario_type}","pilot_speaks_first":{str(pilot_first).lower()},"pilot_first_call":"","situation_briefing":"2-3 sentences.","atc_transmission":"Full ATC message.","key_readback_items":["item1","item2","item3"],"correct_response":"Ideal pilot readback.","coaching_notes":"What to focus on.","phonetic_tips":["tip1","tip2"],"difficulty_tags":["tag1"],"next_atc_hint":"Brief hint"}}"""
    response=mistral_chat(prompt,max_tokens=1100,stream=True)
    full=""
    for line in response.iter_lines():
        if line:
            line=line.decode("utf-8") if isinstance(line,bytes) else line
            if line.startswith("data: "):
                ds=line[6:]
                if ds.strip()=="[DONE]": break
                try:
                    delta=json.loads(ds).get("choices",[{}])[0].get("delta",{}).get("content","")
                    if delta: full+=delta; yield delta
                except: pass

def generate_pilot_init_scenario(airport,aircraft,scenario_type,difficulty,custom_situation=None):
    complexity=DIFFICULTY_LEVELS[difficulty]["complexity"]
    situation_context=f"\nSPECIFIC SITUATION: {custom_situation}\n" if custom_situation else ""
    prompt=f"""Senior ATC instructor. Create a realistic pilot-initiates training scenario.
The PILOT will initiate radio contact with ATC. Provide the situation and a model pilot initial call.
Airport: {airport} | Aircraft: {aircraft} | Type: {scenario_type} | Difficulty: {difficulty} (complexity {complexity}/4)
{situation_context}{AVIATE_RULES}
Return ONLY JSON:
{{"scenario_id":"SC{random.randint(1000,9999)}","scenario_type":"{scenario_type}","pilot_speaks_first":true,
"situation_briefing":"2-3 sentences describing what the pilot needs to do and why they are calling ATC.",
"pilot_initial_context":"One sentence describing what the pilot should say/request.",
"model_pilot_call":"The ideal pilot initial transmission verbatim. Include callsign, position, altitude, and request.",
"key_pilot_call_items":["item1","item2","item3"],
"coaching_notes":"What to focus on for the initial call.",
"phonetic_tips":["tip1","tip2"],
"difficulty_tags":["tag1"]}}"""
    response=mistral_chat(prompt,max_tokens=900,stream=True)
    full=""
    for line in response.iter_lines():
        if line:
            line=line.decode("utf-8") if isinstance(line,bytes) else line
            if line.startswith("data: "):
                ds=line[6:]
                if ds.strip()=="[DONE]": break
                try:
                    delta=json.loads(ds).get("choices",[{}])[0].get("delta",{}).get("content","")
                    if delta: full+=delta; yield delta
                except: pass

def generate_atc_response_to_pilot(airport,aircraft,difficulty,pilot_call,conv_history=None,scenario_type="Normal Traffic"):
    complexity=DIFFICULTY_LEVELS[difficulty]["complexity"]
    history_text=""
    if conv_history:
        history_text="Previous exchange:\n"+"".join(f"{'ATC' if ex['role']=='atc' else 'PILOT'}: {ex['text']}\n" for ex in conv_history[-6:])
    prompt=f"""You are a realistic ATC controller responding to a pilot transmission.
Airport: {airport} | Aircraft: {aircraft} | Difficulty: {difficulty} | Type: {scenario_type}
{history_text}Pilot just said: {pilot_call}
{AVIATE_RULES}
Generate a realistic, concise ATC response.
Return ONLY JSON:
{{"atc_transmission":"Realistic ATC reply to pilot.","key_readback_items":["item1","item2","item3"],"correct_response":"Ideal pilot readback to THIS ATC response.","coaching_notes":"What the pilot should focus on when reading back.","flight_phase":"taxi|departure|climb|cruise|descent|approach|landing","session_complete":false}}"""
    r=mistral_chat(prompt,max_tokens=600)
    text=r.json()["choices"][0]["message"]["content"].strip()
    try:
        if text.startswith("```"): text=text.split("```")[1].lstrip("json").strip()
        return json.loads(text)
    except:
        return {"atc_transmission":"Roger, stand by.","key_readback_items":[],"correct_response":"Stand by.",
                "coaching_notes":"","flight_phase":"unknown","session_complete":False}

def evaluate_pilot_initial_call(scenario,pilot_call):
    model_call=scenario.get("model_pilot_call","")
    items=scenario.get("key_pilot_call_items",[])
    context=scenario.get("pilot_initial_context","")
    prompt=f"""Expert ATC instructor evaluating a pilot's initial radio call.
Situation: {context}
Model call: {model_call}
Key items: {json.dumps(items)}
Pilot said: {pilot_call}
{AVIATE_RULES}
Return ONLY JSON:
{{"score":<0-100>,"pass_fail":"PASS|FAIL","items_correct":["..."],"items_missed":["..."],"items_incorrect":["..."],"overall_feedback":"2-3 sentences","specific_corrections":"Corrected phrasing","coaching_tip":"One tip","grade":"Excellent|Good|Needs Work|Unsatisfactory"}}"""
    r=mistral_chat(prompt,max_tokens=500)
    text=r.json()["choices"][0]["message"]["content"].strip()
    try:
        if text.startswith("```"): text=text.split("```")[1].lstrip("json").strip()
        return json.loads(text)
    except:
        return {"score":0,"pass_fail":"ERROR","overall_feedback":"Parse error.","grade":"N/A",
                "items_correct":[],"items_missed":[],"items_incorrect":[],"specific_corrections":"","coaching_tip":""}

def generate_next_exchange(airport,aircraft,difficulty,conv_history,scenario_type):
    complexity=DIFFICULTY_LEVELS[difficulty]["complexity"]
    history_text="\n".join(f"{'ATC' if ex['role']=='atc' else 'PILOT'}: {ex['text']}" for ex in conv_history[-6:])
    prompt=f"""ATC instructor running continuous training.
Airport: {airport} | Aircraft: {aircraft} | Difficulty: {difficulty} | Type: {scenario_type}
Conversation:\n{history_text}\n{AVIATE_RULES}
Generate NEXT logical ATC instruction. If aircraft landed/vacated, set session_complete: true.
Return ONLY JSON:
{{"atc_transmission":"Next ATC call.","key_readback_items":["item1","item2"],"correct_response":"Ideal readback.","coaching_notes":"Focus.","flight_phase":"taxi|departure|climb|cruise|descent|approach|landing|vacate","session_complete":false}}"""
    r=mistral_chat(prompt,max_tokens=600)
    text=r.json()["choices"][0]["message"]["content"].strip()
    try:
        if text.startswith("```"): text=text.split("```")[1].lstrip("json").strip()
        return json.loads(text)
    except:
        return {"atc_transmission":"Stand by.","key_readback_items":[],"correct_response":"",
                "coaching_notes":"","flight_phase":"unknown","session_complete":False}

def evaluate_response(scenario_or_exchange,pilot_response,is_chain=False):
    atc=scenario_or_exchange.get("atc_transmission","")
    correct=scenario_or_exchange.get("correct_response","")
    items=scenario_or_exchange.get("key_readback_items",[])
    is_brief=len(correct.split())<=10
    brevity="NOTE: High-urgency — brevity correct. Don't penalize short responses if key items present." if is_brief else ""
    prompt=f"""Expert ATC instructor evaluating pilot readback.
ATC said: {atc}
Model answer: {correct}
Key items: {json.dumps(items)}
Pilot said: {pilot_response}
{brevity}
Return ONLY JSON:
{{"score":<0-100>,"pass_fail":"PASS|FAIL","items_correct":["..."],"items_missed":["..."],"items_incorrect":["..."],"overall_feedback":"2-3 sentences","specific_corrections":"Corrected phrasing","coaching_tip":"One tip","grade":"Excellent|Good|Needs Work|Unsatisfactory"}}"""
    r=mistral_chat(prompt,max_tokens=500)
    text=r.json()["choices"][0]["message"]["content"].strip()
    try:
        if text.startswith("```"): text=text.split("```")[1].lstrip("json").strip()
        return json.loads(text)
    except:
        return {"score":0,"pass_fail":"ERROR","overall_feedback":"Parse error.","grade":"N/A",
                "items_correct":[],"items_missed":[],"items_incorrect":[],"specific_corrections":"","coaching_tip":""}

def build_tts_player(text, difficulty, emergency=False, label_prefix="ATC"):
    speed_map={"Student Pilot":0.82,"Private Pilot":0.92,"Commercial":1.05,"Expert / ATPL":1.2}
    rate=speed_map.get(difficulty,0.9); pitch=0.85 if emergency else 1.0
    safe=aviate_numbers(text).replace('"','\\"').replace('\n',' ').replace("'","\\'")
    pid=f"atc_{random.randint(10000,99999)}"
    accent="#e84040" if emergency else "#4d7cfe"
    return f"""
<div style="background:var(--bg-elevated,#161a28);border:1px solid var(--border,#242840);
border-left:3px solid {accent};border-radius:4px;padding:14px 18px;margin:8px 0;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
    <div style="font-family:'DM Mono',monospace;color:{accent};font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase">
      {label_prefix} · {difficulty}
    </div>
    <div id="{pid}_status" style="font-family:'DM Mono',monospace;color:#4a5070;font-size:0.62rem">READY</div>
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:0.8rem;color:#b8cce8;line-height:1.6;
  padding:10px 12px;background:rgba(0,0,0,0.2);border-radius:3px;margin-bottom:10px">{text}</div>
  <div style="display:flex;gap:8px">
    <button onclick="atcPlay_{pid}()" style="background:rgba(77,124,254,0.1);color:{accent};
    border:1px solid {accent}44;border-radius:3px;padding:6px 16px;font-family:'DM Sans',sans-serif;
    font-size:0.75rem;font-weight:500;cursor:pointer;letter-spacing:0.05em">▶ Play</button>
    <button onclick="atcStop_{pid}()" style="background:transparent;color:#4a5070;
    border:1px solid #242840;border-radius:3px;padding:6px 12px;font-size:0.75rem;cursor:pointer">■ Stop</button>
    <button onclick="atcReplay_{pid}()" style="background:transparent;color:#4a5070;
    border:1px solid #242840;border-radius:3px;padding:6px 12px;font-size:0.75rem;cursor:pointer">↺</button>
  </div>
</div>
<script>(function(){{var t="{safe}",rate={rate},pitch={pitch},playing=false,pid="{pid}",utt=null;
window['atcPlay_'+pid]=function(){{if(playing)return;playing=true;
var s=document.getElementById(pid+'_status');if(s)s.textContent='TRANSMITTING';
utt=new SpeechSynthesisUtterance(t);utt.rate=rate;utt.pitch=pitch;
var vs=speechSynthesis.getVoices();var mv=vs.find(v=>/(male|david|alex|daniel)/i.test(v.name));
if(mv)utt.voice=mv;utt.onend=function(){{playing=false;if(s)s.textContent='COMPLETE';}};
speechSynthesis.speak(utt);}};
window['atcStop_'+pid]=function(){{speechSynthesis.cancel();playing=false;
var s=document.getElementById(pid+'_status');if(s)s.textContent='STOPPED';}};
window['atcReplay_'+pid]=function(){{window['atcStop_'+pid]();setTimeout(function(){{window['atcPlay_'+pid]();}},200);}};
if(speechSynthesis.onvoiceschanged!==undefined)speechSynthesis.onvoiceschanged=function(){{speechSynthesis.getVoices();}};
}})();</script>"""


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="suite-header">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:16px">
    <div>
      <div style="font-family:var(--font-mono);font-size:0.6rem;letter-spacing:0.3em;text-transform:uppercase;
      color:var(--text-muted);margin-bottom:10px">Aviation Speech Intelligence Suite</div>
      <div class="suite-title">AviSuite</div>
      <div class="suite-subtitle">Fatigue Analyzer  ·  Pilot Whisperer  ·  ScenarioSynth</div>
      <div class="suite-badge">⚠ Research & Training Use Only · Not a Certified Safety Device</div>
    </div>
    <div style="text-align:right;padding-top:4px">
      <div style="font-family:var(--font-mono);font-size:0.56rem;letter-spacing:0.18em;text-transform:uppercase;color:var(--text-muted)">
        ICAO · FAR/EASA · NASA Ames
      </div>
      <div style="margin-top:6px;font-family:var(--font-mono);font-size:0.52rem;color:var(--text-muted)">
        DSP · HNR · MFCC · F0 · LPC · Whisper STT · Mistral NLP · RL
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="padding:12px 0 6px;font-family:var(--font-display);font-size:0.65rem;letter-spacing:0.2em;text-transform:uppercase;color:var(--text-muted);border-bottom:1px solid var(--border)">Suite Configuration</div>', unsafe_allow_html=True)

    # Cloud capability banner
    cap_items = []
    if FFMPEG_AVAILABLE: cap_items.append("ffmpeg ✓")
    else: cap_items.append("ffmpeg ✗ (upload only)")
    if SOUNDDEVICE_AVAILABLE: cap_items.append("sounddevice ✓")
    else: cap_items.append("mic (browser ✓)")
    st.markdown(f'<div style="font-family:var(--font-mono);font-size:0.55rem;color:var(--text-muted);padding:4px 0;line-height:1.8">{" · ".join(cap_items)}</div>', unsafe_allow_html=True)

    st.markdown("### Common")
    m1_role = st.selectbox("Speaker Role", ["Pilot (Male)","Pilot (Female)","ATC Controller","Cabin Crew"], key="m1_role_sel")
    m1_dur  = st.slider("Recording Duration (s)", 5, 60, 15, 5)

    st.markdown("### Module 3 — Training")
    m3_airport  = st.selectbox("Airport", AIRPORTS, index=7)
    m3_aircraft = st.selectbox("Aircraft", AIRCRAFT_TYPES, index=1)
    diff_options= list(DIFFICULTY_LEVELS.keys())
    sel_diff    = st.select_slider("Difficulty", options=diff_options, value=st.session_state.difficulty)
    st.session_state.difficulty = sel_diff
    badge = DIFFICULTY_LEVELS[sel_diff]["badge"]
    st.markdown(f'<span class="diff-tag {badge}">{sel_diff}</span>', unsafe_allow_html=True)
    st.session_state.auto_adapt = st.toggle("Auto-Adapt Difficulty", value=st.session_state.auto_adapt)

    st.markdown("---")
    st.markdown("### Session Statistics")
    acc_pct = (st.session_state.correct_responses/max(st.session_state.total_attempts,1))*100
    c1,c2=st.columns(2)
    with c1:
        st.markdown(f'<div class="score-display"><div class="score-num">{st.session_state.session_score}</div><div class="score-lbl">Score</div></div>',unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="score-display"><div class="score-num">{acc_pct:.0f}%</div><div class="score-lbl">Accuracy</div></div>',unsafe_allow_html=True)
    if st.session_state.total_attempts>0: st.progress(min(acc_pct/100,1.0))

    st.markdown("---")
    st.markdown(f'<div style="font-family:var(--font-mono);font-size:0.62rem;color:var(--accent-violet);letter-spacing:0.1em">RL Q-Learning · Episode {st.session_state.rl_episode}</div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Reset Session"):
        for k in ["session_score","total_attempts","correct_responses","history","current_scenario",
                  "feedback","streaks","rl_q_table","rl_episode","rl_weak_areas","rl_item_errors",
                  "rl_recommendations","conv_chain","conv_active","conv_step","m1_history",
                  "m1_results","m1_features","whisper_t1","whisper_chain"]:
            if k in st.session_state:
                v=st.session_state[k]
                if isinstance(v,int): st.session_state[k]=0
                elif isinstance(v,list): st.session_state[k]=[]
                elif isinstance(v,dict): st.session_state[k]={}
                elif isinstance(v,bool): st.session_state[k]=False
                elif isinstance(v,str): st.session_state[k]=""
                else: st.session_state[k]=None
        st.rerun()

    st.markdown("---")
    st.markdown('<div style="font-size:0.6rem;color:var(--text-muted);line-height:1.9;font-family:var(--font-mono)">ICAO Doc 9966<br>FAA AC 120-100<br>NASA Ames Fatigue Research<br>EUROCONTROL HF Guidelines</div>',unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "Fatigue Analyzer",
    "Pilot Whisperer",
    "Scenario Synth",
    "Performance",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — MODULE 1: FATIGUE ANALYZER
#  FIX: sounddevice block replaced with upload-only on Streamlit Cloud
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_main1, col_side1 = st.columns([3,1])
    with col_main1:
        st.markdown('<div class="section-label">Acoustic Fatigue · Stress · Cognitive Load · RT Clarity Analysis</div>', unsafe_allow_html=True)

    with col_side1:
        if st.session_state.m1_results:
            ind=st.session_state.m1_results; rl=ind['risk_level']; rc=RISK_COLORS[rl]
            st.markdown(f"""<div style="text-align:center;padding:12px;background:var(--bg-card);
            border:1px solid {rc}44;border-radius:6px;border-top:2px solid {rc}">
            <div style="font-family:var(--font-mono);font-size:0.52rem;letter-spacing:0.25em;text-transform:uppercase;color:{rc}88">Risk Level</div>
            <div style="font-family:var(--font-display);font-size:1.4rem;font-weight:700;color:{rc};margin:4px 0">{rl}</div>
            <div style="font-family:var(--font-mono);font-size:0.58rem;color:{rc}88">{ind['composite']:.0f} / 100</div>
            </div>""", unsafe_allow_html=True)

    # ── Recording / Upload row ──────────────────────────────────────────────
    if SOUNDDEVICE_AVAILABLE:
        # Local environment: show record button
        col_rec1, col_up1, col_conf1 = st.columns([2,1,1])
        with col_rec1:
            if not st.session_state.m1_recording:
                if st.button("● Record Transmission", use_container_width=True):
                    st.session_state.m1_recording = True; st.rerun()
            else:
                st.markdown('<div class="rec-active"><span class="rec-dot"></span>Recording — Speak Naturally</div>',unsafe_allow_html=True)
                prog=st.progress(0)
                try:
                    import sounddevice as sd
                    for i in range(m1_dur*10): time.sleep(0.1); prog.progress((i+1)/(m1_dur*10))
                    with st.spinner("Analyzing..."):
                        rec=sd.rec(int(m1_dur*SR),samplerate=SR,channels=1,dtype='float32'); sd.wait()
                        audio=rec.flatten(); ab=audio.astype(np.float32).tobytes()
                        feats=extract_all_features(ab,SR); ind=compute_indicators(feats,m1_role)
                        st.session_state.m1_results=ind; st.session_state.m1_features=feats
                        st.session_state.m1_history.append({
                            'time':time.strftime("%H:%M:%S"),'role':m1_role,
                            **{k:ind[k] for k in ['fatigue','stress','cognitive','rt_clarity','composite','risk_level','confidence']}})
                except Exception as e:
                    st.error(f"Recording error: {e}")
                st.session_state.m1_recording=False; st.rerun()
        with col_up1:
            uploaded1=st.file_uploader("Upload Audio",type=['wav','mp3','ogg','flac'],label_visibility='collapsed',key="m1_upload")
        with col_conf1:
            if st.session_state.m1_results:
                conf=st.session_state.m1_results['confidence']
                cc=sev_color(100-abs(conf-100))
                st.markdown(f"""<div class="score-display">
                <div class="score-num" style="color:{cc};font-size:1.6rem">{conf:.0f}%</div>
                <div class="score-lbl">Signal Confidence</div>
                </div>""",unsafe_allow_html=True)
    else:
        # Cloud environment: upload-only with clear messaging
        st.markdown("""<div class="cloud-warning">
        🎙 <strong>Microphone recording</strong> requires a local installation of AviSuite (sounddevice library).
        On Streamlit Cloud, please <strong>upload an audio file</strong> below for analysis.
        WAV, MP3, OGG, and FLAC are supported. Minimum 5 seconds recommended.
        </div>""", unsafe_allow_html=True)
        col_up1, col_conf1 = st.columns([2,1])
        with col_conf1:
            if st.session_state.m1_results:
                conf=st.session_state.m1_results['confidence']
                cc=sev_color(100-abs(conf-100))
                st.markdown(f"""<div class="score-display">
                <div class="score-num" style="color:{cc};font-size:1.6rem">{conf:.0f}%</div>
                <div class="score-lbl">Signal Confidence</div>
                </div>""",unsafe_allow_html=True)

    # File upload processing (works in both modes)
    uploaded1 = st.file_uploader(
        "Upload Audio File for Fatigue Analysis",
        type=['wav','mp3','ogg','flac'],
        key="m1_upload_main",
        help="Upload a WAV, MP3, OGG or FLAC recording. Minimum 5 seconds recommended."
    )
    if uploaded1:
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded1.name)[1],delete=False) as tmp:
            tmp.write(uploaded1.read()); tmp_path=tmp.name
        with st.spinner("Processing audio..."):
            try:
                audio=load_audio_file(tmp_path); ab=audio.astype(np.float32).tobytes()
                feats=extract_all_features(ab,SR); ind=compute_indicators(feats,m1_role)
                st.session_state.m1_results=ind; st.session_state.m1_features=feats
                st.session_state.m1_history.append({
                    'time':time.strftime("%H:%M:%S"),'role':m1_role,
                    **{k:ind[k] for k in ['fatigue','stress','cognitive','rt_clarity','composite','risk_level','confidence']}})
            except Exception as e: st.error(f"Audio processing error: {e}")
            finally:
                try: os.unlink(tmp_path)
                except: pass
        st.rerun()

    if st.session_state.m1_results and st.session_state.m1_features:
        ind=st.session_state.m1_results; feat=st.session_state.m1_features
        st.markdown("---")
        st.markdown(f"""<div class="info-strip">
        F0 <span class="val">{ind['f0_mean']:.0f} Hz</span>
        F0± <span class="val">{ind['f0_std']:.0f} Hz</span>
        HNR <span class="val">{ind['hnr']:.1f} dB</span>
        Jitter <span class="val">{ind['jitter']:.2f}%</span>
        Shimmer <span class="val">{ind['shimmer_db']:.2f} dB</span>
        Speech <span class="val">{ind['speech_ratio']:.0f}%</span>
        Pauses/s <span class="val">{ind['pause_rate']:.2f}</span>
        Centroid <span class="val">{ind['centroid']:.0f} Hz</span>
        </div>""",unsafe_allow_html=True)

        c1,c2,c3,c4=st.columns(4)
        with c1: render_metric_card("Fatigue",f"{ind['fatigue']:.0f}",severity_label(ind['fatigue']),"Energy · HNR · Shimmer · Pauses",ind['fatigue'],ind['fatigue'])
        with c2: render_metric_card("Stress",f"{ind['stress']:.0f}",severity_label(ind['stress']),"F0 Elevation · Jitter · Flux",ind['stress'],ind['stress'])
        with c3: render_metric_card("Cognitive Load",f"{ind['cognitive']:.0f}",severity_label(ind['cognitive']),"Hesitations · Rhythm · LPC",ind['cognitive'],ind['cognitive'])
        with c4: render_metric_card("RT Clarity",f"{ind['rt_clarity']:.0f}","Readback Quality","HNR · Jitter · Speech Rate",ind['rt_clarity'],100-ind['rt_clarity'])

        st.markdown("<br>",unsafe_allow_html=True)
        vtab1,vtab2,vtab3,vtab4=st.tabs(["Visualizations","Sub-Scores","Feature Table","Trend"])

        with vtab1:
            vg,vr,vf=st.columns([1,1,2])
            with vg:
                st.plotly_chart(fig_risk_gauge(ind['composite']),use_container_width=True,config={'displayModeBar':False})
                st.markdown('<div style="text-align:center;font-family:var(--font-mono);font-size:0.56rem;letter-spacing:0.15em;color:var(--text-muted);text-transform:uppercase">Composite Risk</div>',unsafe_allow_html=True)
            with vr:
                st.plotly_chart(fig_radar(ind),use_container_width=True,config={'displayModeBar':False})
            with vf:
                f0t=feat.get('_f0_track',[])
                if f0t: st.plotly_chart(fig_f0_track(f0t,SR),use_container_width=True,config={'displayModeBar':False})
            mfcc_d=feat.get('_mfcc',[])
            if mfcc_d: st.plotly_chart(fig_mfcc_heatmap(mfcc_d),use_container_width=True,config={'displayModeBar':False})

        with vtab2:
            cs1,cs2,cs3=st.columns(3)
            with cs1: st.plotly_chart(fig_sub_breakdown(ind['fat_subs'],'Fatigue Contributors','#e84040'),use_container_width=True,config={'displayModeBar':False})
            with cs2: st.plotly_chart(fig_sub_breakdown(ind['stre_subs'],'Stress Contributors','#f0a030'),use_container_width=True,config={'displayModeBar':False})
            with cs3: st.plotly_chart(fig_sub_breakdown(ind['cog_subs'],'Cognitive Contributors','#8b5cf6'),use_container_width=True,config={'displayModeBar':False})

        with vtab3:
            disp={k:v for k,v in feat.items() if not k.startswith('_')}
            df_f=pd.DataFrame([{'Feature':k,'Value':f"{v:.5f}" if isinstance(v,float) else str(v)} for k,v in disp.items()])
            st.dataframe(df_f,use_container_width=True,height=400)

        with vtab4:
            if len(st.session_state.m1_history)>=2:
                fig_t=fig_session_trend(st.session_state.m1_history)
                if fig_t: st.plotly_chart(fig_t,use_container_width=True,config={'displayModeBar':False})
            if st.session_state.m1_history:
                st.dataframe(pd.DataFrame(st.session_state.m1_history).round(1),use_container_width=True)
            else: st.caption("Record 2+ sessions to see trend.")

        st.markdown("---")
        st.markdown('<div class="section-label">Aviation Analysis Report</div>', unsafe_allow_html=True)
        generate_aviation_report(ind, feat, m1_role)
    else:
        st.markdown("""<div style="text-align:center;padding:60px 20px;border:1px dashed var(--border);
        border-radius:6px;margin-top:16px">
        <div style="font-size:2.5rem;opacity:0.12;margin-bottom:14px">🎙</div>
        <div style="font-family:var(--font-display);font-size:1rem;color:var(--text-muted);letter-spacing:0.08em">Awaiting Transmission</div>
        <div style="font-family:var(--font-mono);font-size:0.65rem;margin-top:8px;color:var(--text-muted)">Upload an audio file above · Minimum 5 seconds recommended</div>
        </div>""",unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — MODULE 2: PILOT WHISPERER
#  FIX: ffmpeg guard → shows clear error instead of crashing; monitoring loop
#       exits cleanly if ffmpeg missing
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">Live ATC Radio Discipline Coach · Whisper STT · Mistral NLP</div>', unsafe_allow_html=True)

    # Show ffmpeg status prominently
    if not FFMPEG_AVAILABLE:
        st.markdown("""<div class="cloud-warning">
        ⚠ <strong>ffmpeg not found</strong> on this server.<br>
        Live ATC capture requires ffmpeg. To fix:<br>
        1. Create a file named <code>packages.txt</code> in your repo root<br>
        2. Add a single line: <code>ffmpeg</code><br>
        3. Redeploy your app on Streamlit Cloud<br><br>
        The Scenario Synth (Tab 3) and Fatigue Analyzer (Tab 1) work without ffmpeg.
        </div>""", unsafe_allow_html=True)

    m2_col1, m2_col2 = st.columns([3,1])
    with m2_col1:
        st.markdown("""<div class="data-panel">
        <div class="data-panel-header">Live ATC Stream Monitor</div>
        <div style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-secondary)">
        ffmpeg captures audio → Whisper large model → Mistral labels speakers + discipline analysis
        </div>
        </div>""", unsafe_allow_html=True)
    with m2_col2:
        m2_seconds = st.slider("Capture Duration (s)", 10, 120, 30, key="m2_sec")

    m2_start = st.toggle(
        "Activate Live Monitoring",
        value=False,
        key="m2_start_toggle",
        disabled=not FFMPEG_AVAILABLE,
        help="Requires ffmpeg. Add 'ffmpeg' to packages.txt and redeploy." if not FFMPEG_AVAILABLE else None
    )

    if not m2_start:
        st.markdown(f"""<div style="text-align:center;padding:50px;border:1px dashed var(--border);
        border-radius:6px;color:var(--text-muted)">
        <div style="font-size:2rem;opacity:0.1;margin-bottom:12px">📡</div>
        <div style="font-family:var(--font-display);font-size:0.9rem;letter-spacing:0.1em">Monitoring Inactive</div>
        <div style="font-family:var(--font-mono);font-size:0.62rem;margin-top:6px">
        {"ffmpeg not available — add to packages.txt and redeploy" if not FFMPEG_AVAILABLE else "Toggle above to begin live monitoring · Requires ffmpeg + network access"}
        </div>
        </div>""", unsafe_allow_html=True)
    else:
        try:
            import whisper as whisper_lib
            model_whisper_loaded = True
        except ImportError:
            model_whisper_loaded = False
            st.warning("Whisper not installed. Add `openai-whisper` to requirements.txt")

        if model_whisper_loaded and FFMPEG_AVAILABLE:
            @st.cache_resource
            def load_whisper_model():
                import whisper as wlib
                return wlib.load_model("large")  # 'base' is faster on cloud

            m2_status_ph   = st.empty()
            m2_audio_ph    = st.empty()
            m2_metrics_ph  = st.empty()
            m2_results_ph  = st.empty()
            m2_cycle_ph    = st.empty()

            cycle_count = 0
            while st.session_state.get("m2_start_toggle", False):
                cycle_count += 1
                ts = time.strftime("%H:%M:%S")

                m2_status_ph.markdown(f"""<div class="live-badge">
                <span class="live-dot"></span>Capturing cycle {cycle_count} · {m2_seconds}s
                </div>""", unsafe_allow_html=True)

                try:
                    audio_path = capture_audio_stream(LIVEATC_URL, m2_seconds)

                    with open(audio_path, "rb") as f2: audio_bytes = f2.read()
                    m2_audio_ph.audio(audio_bytes, format="audio/wav")
                    m2_cycle_ph.markdown(f'<div style="font-family:var(--font-mono);font-size:0.6rem;color:var(--text-muted)">Cycle {cycle_count} · {ts}</div>', unsafe_allow_html=True)

                    m2_status_ph.markdown("""<div class="live-badge" style="color:var(--accent-blue);
                    background:rgba(77,124,254,0.08);border-color:rgba(77,124,254,0.2)">
                    <span style="width:5px;height:5px;border-radius:50%;background:var(--accent-blue);display:inline-block"></span>
                    Transcribing with Whisper...</div>""", unsafe_allow_html=True)

                    wmodel = load_whisper_model()
                    result = wmodel.transcribe(audio_path, fp16=False, language="en",
                                              condition_on_previous_text=False)
                    raw_text = result["text"].strip()

                    try: os.remove(audio_path)
                    except: pass

                    if not raw_text:
                        m2_status_ph.markdown("""<div style="font-family:var(--font-mono);font-size:0.68rem;
                        color:var(--text-muted)">No speech detected in this capture window.</div>""", unsafe_allow_html=True)
                    else:
                        m2_status_ph.markdown("""<div class="live-badge" style="color:var(--accent-violet);
                        background:rgba(139,92,246,0.08);border-color:rgba(139,92,246,0.2)">
                        <span style="width:5px;height:5px;border-radius:50%;background:var(--accent-violet);display:inline-block"></span>
                        Mistral analyzing...</div>""", unsafe_allow_html=True)

                        cleaned  = mistral_cleanup(raw_text)
                        analysis = mistral_discipline(cleaned)
                        has_miscomm = analysis.get("miscomm_flags", 0) > 0 or "POSSIBLE MISCOMM" in cleaned

                        m2_status_ph.markdown(f"""<div class="live-badge" style="color:var(--accent-green);
                        background:rgba(45,204,143,0.08);border-color:rgba(45,204,143,0.2)">
                        <span style="width:5px;height:5px;border-radius:50%;background:var(--accent-green);display:inline-block"></span>
                        Cycle {cycle_count} complete · {ts}</div>""", unsafe_allow_html=True)

                        score_v = analysis.get("score", 0)
                        n_issues = len(analysis.get("issues", []))
                        n_miscomm = analysis.get("miscomm_flags", 0)
                        clearance = analysis.get("clearance", "—")
                        score_color = "#2dcc8f" if score_v>=80 else "#f0a030" if score_v>=60 else "#e84040"
                        clear_color = "#2dcc8f" if clearance=="CLEAR" else "#e84040"

                        with m2_metrics_ph.container():
                            mc1,mc2,mc3,mc4,mc5 = st.columns(5)
                            def m2_card(label, val, color):
                                st.markdown(f"""<div class="score-display">
                                <div class="score-num" style="color:{color};font-size:1.5rem">{val}</div>
                                <div class="score-lbl">{label}</div></div>""", unsafe_allow_html=True)
                            with mc1: m2_card("Discipline", score_v, score_color)
                            with mc2: m2_card("Issues", n_issues, "#f0a030" if n_issues>0 else "#2dcc8f")
                            with mc3: m2_card("Miscomm", n_miscomm, "#e84040" if n_miscomm>0 else "#2dcc8f")
                            with mc4: m2_card("Clearance", clearance, clear_color)
                            with mc5: m2_card("Confidence", analysis.get("confidence","—"), "#4d7cfe")

                        with m2_results_ph.container():
                            left2, right2 = st.columns(2)
                            with left2:
                                st.markdown('<div class="section-label">Raw Whisper Transcript</div>', unsafe_allow_html=True)
                                st.markdown(f'<div class="data-panel" style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-secondary);line-height:1.7">{raw_text}</div>', unsafe_allow_html=True)
                                st.markdown('<div class="section-label">Mistral Labeled Output</div>', unsafe_allow_html=True)
                                cl_fmt = cleaned.replace("\n","<br>")
                                cl_fmt = cl_fmt.replace("[ATC]",'<span style="color:var(--accent-amber);font-weight:600">[ATC]</span>')
                                cl_fmt = re.sub(r'\[PILOT([^\]]*)\]',r'<span style="color:var(--accent-blue);font-weight:600">[PILOT\1]</span>',cl_fmt)
                                box_style = "border-left-color:var(--accent-amber);" if has_miscomm else ""
                                st.markdown(f'<div class="data-panel" style="font-family:var(--font-mono);font-size:0.72rem;line-height:1.8;color:var(--text-primary);{box_style}">{cl_fmt}</div>', unsafe_allow_html=True)
                            with right2:
                                st.markdown('<div class="section-label">Discipline Issues</div>', unsafe_allow_html=True)
                                issues = analysis.get("issues", [])
                                if not issues:
                                    st.markdown('<div class="status-nominal">No discipline issues detected in this capture window.</div>', unsafe_allow_html=True)
                                else:
                                    sev_colors = {"HIGH":("#e84040","rgba(232,64,64,0.08)"),"MEDIUM":("#f0a030","rgba(240,160,48,0.08)"),"LOW":("#2dcc8f","rgba(45,204,143,0.08)")}
                                    for issue in issues:
                                        sev = issue.get("severity","MEDIUM")
                                        ic, ibg = sev_colors.get(sev, ("#4d7cfe","rgba(77,124,254,0.08)"))
                                        st.markdown(f"""<div style="background:{ibg};border:1px solid {ic}33;
                                        border-left:3px solid {ic};border-radius:4px;padding:12px 14px;margin:6px 0">
                                        <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px">
                                        <span style="font-family:var(--font-mono);font-size:0.62rem;letter-spacing:0.1em;color:{ic}">{issue.get('type','')}</span>
                                        <span style="font-family:var(--font-mono);font-size:0.6rem;color:{ic};opacity:0.7">{sev}</span>
                                        </div>
                                        <div style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-secondary);padding:6px 8px;background:rgba(0,0,0,0.2);border-radius:3px;margin:4px 0">{issue.get('quote','')}</div>
                                        <div style="font-size:0.72rem;color:var(--text-secondary);margin:6px 0">{issue.get('description','')}</div>
                                        <div style="font-size:0.7rem;color:var(--accent-amber)">→ {issue.get('tip','')}</div>
                                        </div>""", unsafe_allow_html=True)
                                st.markdown('<div class="section-label">Coach Summary</div>', unsafe_allow_html=True)
                                st.markdown(f'<div class="data-panel" style="font-size:0.78rem;color:var(--text-secondary);line-height:1.7">{analysis.get("coach_summary","")}</div>', unsafe_allow_html=True)

                except Exception as e:
                    m2_status_ph.error(f"Capture error (cycle {cycle_count}): {e}")
                    time.sleep(5)

                time.sleep(1)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — MODULE 3: SCENARIO SYNTH
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    m3_inner = st.tabs(["Training","Chain Session","Coaching","RL Engine"])

    # ── M3 Tab A — Single Training ──────────────────────────────────────────
    with m3_inner[0]:
        col_main3, col_info3 = st.columns([3,1])
        with col_main3:

            st.markdown('<div class="section-label">Interaction Mode</div>', unsafe_allow_html=True)
            interaction_mode = st.radio(
                "interaction_mode_radio",
                ["ATC Speaks First  (Standard Readback)", "Pilot Initiates  (You call ATC, ATC responds — continuous loop)"],
                key="m3_interaction_mode", label_visibility="collapsed",
                horizontal=True
            )
            pilot_initiates = "Pilot Initiates" in interaction_mode

            st.markdown("---")

            sc1c, sc2c = st.columns([2,1])
            with sc1c:
                if pilot_initiates:
                    m3_presets = {
                        "Random (Weighted)": None,
                        "Check-in on New Frequency": "Aircraft checking in on a new frequency after sector handoff",
                        "Request IFR Clearance": "Aircraft on ground requesting IFR clearance before departure",
                        "Request Descent (STAR)": "Aircraft en-route requesting descent via STAR",
                        "Request Direct Routing": "Aircraft requesting direct routing to save time/fuel",
                        "Declare MAYDAY – Engine Failure": "Aircraft declaring MAYDAY due to engine failure",
                        "Declare PAN-PAN – Medical": "Aircraft declaring PAN-PAN for medical emergency on board",
                        "Request ILS Approach": "Aircraft requesting ILS approach and full clearance",
                        "Startup & Push Clearance": "Aircraft on stand requesting startup and push-back clearance",
                        "Taxi to Holding Point": "Aircraft requesting taxi after receiving ATC clearance",
                        "Go-Around – Initiate": "Pilot initiating a go-around due to unstable approach",
                        "TCAS RA – Report to ATC": "Pilot reporting TCAS Resolution Advisory to ATC",
                        "Request Special VFR": "VFR aircraft requesting Special VFR clearance through CTR",
                        "Custom...": "CUSTOM",
                    }
                else:
                    m3_presets = {
                        "Random (Weighted)": None,
                        "ILS Approach Clearance": "Aircraft on ILS approach receiving final clearance",
                        "Takeoff Clearance": "Aircraft ready for departure receiving takeoff clearance",
                        "SID Departure Clearance": "Full departure clearance with SID, squawk, initial altitude",
                        "Frequency Handoff": "Being handed off between sectors with altitude changes",
                        "Descent via STAR": "ATC issuing descent clearance via published STAR",
                        "Radar Vectors for Approach": "ATC vectoring aircraft for instrument approach",
                        "Speed Restriction": "ATC issuing speed restriction for sequencing",
                        "Hold at Fix": "ATC issuing holding instructions at navigation fix",
                        "Conditional Clearance": "ATC issuing conditional clearance (e.g. behind traffic)",
                        "Pushback & Startup": "ATC issuing pushback and startup clearance",
                        "Runway Vacate Instructions": "ATC issuing vacate and taxi-to-gate instructions after landing",
                        "TCAS RA Coordination": "ATC responding to pilot TCAS RA report",
                        "Engine Failure": "Single engine failure after takeoff, declaring emergency",
                        "Fuel Emergency": "Aircraft declaring MAYDAY for minimum fuel",
                        "Windshear Alert": "ATC issuing windshear alert and go-around instruction",
                        "Custom...": "CUSTOM",
                    }
                sit_preset = st.selectbox("Scenario Preset", list(m3_presets.keys()), key="m3_sit_preset")
            with sc2c:
                stype_override = st.selectbox("Scenario Type", ["RL-Guided","Random","Normal Traffic","Heavy Traffic","Emergency"], key="m3_stype_override")

            custom_sit3 = None
            pv3 = m3_presets.get(sit_preset)
            if pv3 == "CUSTOM":
                ctxt3 = st.text_area("Custom situation:", value=st.session_state.custom_situation_text, placeholder="Describe your scenario...", height=70, key="m3_cust")
                st.session_state.custom_situation_text = ctxt3; custom_sit3 = ctxt3
            elif pv3:
                custom_sit3 = pv3
                st.markdown(f'<div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);padding:5px 8px;background:var(--bg-elevated);border-radius:3px;margin:4px 0">{pv3}</div>', unsafe_allow_html=True)

            gc3, _ = st.columns([2,3])
            with gc3:
                gen_label = "Generate Scenario" if not pilot_initiates else "Set Up Situation"
                if st.button(gen_label, use_container_width=True, key="m3_gen_btn"):
                    adaptive_difficulty()
                    if stype_override == "RL-Guided":
                        state3 = rl_get_state(); s_type3 = rl_choose_action(state3)
                        st.session_state.rl_last_state = state3; st.session_state.rl_last_action = s_type3
                        st.session_state.rl_episode += 1
                    elif stype_override == "Random": s_type3 = weighted_scenario_type()
                    else: s_type3 = stype_override
                    st.session_state.feedback = None
                    st.session_state.whisper_t1 = ""
                    st.session_state.pi_scenario = None
                    st.session_state.pi_step = "init"
                    st.session_state.pi_pilot_call_eval = None
                    st.session_state.pi_atc_response = None
                    st.session_state.pi_readback_eval = None
                    st.session_state.pi_conv_history = []
                    st.session_state.whisper_pi = ""
                    st.session_state.whisper_pi_rb = ""
                    if pilot_initiates:
                        with st.spinner("Setting up situation..."):
                            raw3 = "".join(generate_pilot_init_scenario(m3_airport, m3_aircraft, s_type3, st.session_state.difficulty, custom_sit3))
                        try:
                            clean3 = raw3.strip().lstrip("```").lstrip("json").rstrip("```").strip()
                            st.session_state.pi_scenario = json.loads(clean3)
                            st.session_state.pi_step = "pilot_call"
                        except Exception as e3:
                            st.error(f"Parse error: {e3}")
                    else:
                        with st.spinner("Synthesizing scenario..."):
                            raw3 = "".join(generate_scenario(m3_airport, m3_aircraft, s_type3, st.session_state.difficulty, custom_sit3))
                        try:
                            clean3 = raw3.strip().lstrip("```").lstrip("json").rstrip("```").strip()
                            sc3 = json.loads(clean3); st.session_state.current_scenario = sc3
                        except Exception as e3:
                            st.error(f"Parse error: {e3}"); st.session_state.current_scenario = None
                    st.rerun()

            # ── MODE A: ATC SPEAKS FIRST ──────────────────────────────────
            if not pilot_initiates:
                if st.session_state.current_scenario:
                    sc3 = st.session_state.current_scenario; s_type3 = sc3.get("scenario_type","Normal Traffic")
                    urgent_kws3 = ["tcas","windshear","go around","mayday","pan-pan","expedite","immediately"]
                    is_urg3 = s_type3 == "Emergency" or any(k in sc3.get("atc_transmission","").lower() for k in urgent_kws3)

                    st.markdown("---")
                    st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;margin:8px 0 4px">
                    <span style="font-family:var(--font-mono);font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;
                    color:{'var(--accent-red)' if is_urg3 else 'var(--text-muted)'}">{s_type3}</span>
                    <span style="font-family:var(--font-mono);font-size:0.6rem;color:var(--text-muted)">{sc3.get('scenario_id','')}</span>
                    {'<span style="font-family:var(--font-mono);font-size:0.6rem;background:rgba(232,64,64,0.1);color:var(--accent-red);border:1px solid rgba(232,64,64,0.2);border-radius:2px;padding:2px 8px">URGENT</span>' if is_urg3 else ''}
                    </div>""", unsafe_allow_html=True)

                    st.markdown('<div class="section-label">Situation Briefing</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="atc-box">{sc3.get("situation_briefing","")}</div>', unsafe_allow_html=True)

                    st.markdown('<div class="section-label">ATC Transmission</div>', unsafe_allow_html=True)
                    atc_tts3 = build_tts_player(sc3.get("atc_transmission",""), st.session_state.difficulty, emergency=is_urg3)
                    st.components.v1.html(atc_tts3, height=155)

                    show_t3 = st.checkbox("Show transcript", key="m3_show_t")
                    if show_t3:
                        box_cls3 = "emergency-box" if is_urg3 else "atc-box"
                        st.markdown(f'<div class="{box_cls3}">{sc3.get("atc_transmission","")}</div>', unsafe_allow_html=True)

                    if sc3.get("key_readback_items"):
                        st.markdown(f'<div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);margin:4px 0">Read back: {" · ".join(sc3["key_readback_items"])}</div>', unsafe_allow_html=True)

                    st.markdown('<div class="section-label">Your Readback</div>', unsafe_allow_html=True)
                    whisper_readback_widget(state_key="whisper_t1", widget_key="mic_t1", label="Voice Readback  ·  Whisper + Mistral Aviation Correction")

                    pilot_input3 = st.text_area("Or type / edit readback:", value=st.session_state.get("whisper_t1",""),
                        placeholder="Speak above — transcript auto-fills. Or type manually.", height=70, key="m3_pilot_resp")

                    voice_submitted3 = st.session_state.get("whisper_t1_submit_now", False)

                    def do_eval_m3(text3):
                        with st.spinner("Evaluating..."):
                            fb3 = evaluate_response(sc3, text3)
                            st.session_state.feedback = fb3; st.session_state.total_attempts += 1
                            score3 = fb3.get("score",0); st.session_state.session_score += score3
                            if fb3.get("pass_fail") == "PASS": st.session_state.correct_responses += 1; st.session_state.streaks += 1
                            else: st.session_state.streaks = 0
                            rl_update_weak_areas(s_type3, fb3.get("items_missed",[]), fb3.get("items_incorrect",[]))
                            if st.session_state.rl_last_state is not None:
                                rl_update(st.session_state.rl_last_state, st.session_state.rl_last_action or s_type3, score3/100, rl_get_state())
                            rl_generate_recommendations()
                            st.session_state.history.append({"time":datetime.now().strftime("%H:%M:%S"),"type":s_type3,"score":score3,
                                "grade":fb3.get("grade","N/A"),"pass_fail":fb3.get("pass_fail",""),
                                "scenario_id":sc3.get("scenario_id",""),"items_missed":fb3.get("items_missed",[])})
                            st.session_state.whisper_t1 = ""; st.session_state.whisper_t1_submit_now = False

                    if voice_submitted3:
                        vt3 = st.session_state.get("whisper_t1","").strip()
                        if vt3: do_eval_m3(vt3); st.rerun()

                    ec3, rc3 = st.columns([2,1])
                    with ec3:
                        if st.button("Submit & Evaluate", use_container_width=True, key="m3_submit"):
                            txt3 = st.session_state.get("m3_pilot_resp","").strip() or pilot_input3.strip()
                            if txt3: do_eval_m3(txt3); st.rerun()
                            else: st.warning("Enter readback first.")
                    with rc3:
                        if st.button("Show Answer", use_container_width=True, key="m3_show_answer"):
                            st.session_state.feedback = {"score":0,"pass_fail":"REPLAY","overall_feedback":"Reviewing correct response:",
                                "specific_corrections":sc3.get("correct_response",""),"coaching_tip":sc3.get("coaching_notes",""),
                                "grade":"Review","items_correct":[],"items_missed":[]}
                            st.rerun()

                    if st.session_state.feedback:
                        fb3 = st.session_state.feedback; pf3 = fb3.get("pass_fail","")
                        st.markdown("---")
                        if pf3 not in ("REPLAY",):
                            sc3_score = fb3.get("score",0)
                            color3 = "#2dcc8f" if pf3 == "PASS" else "#e84040"
                            s1,s2,s3 = st.columns(3)
                            with s1: st.markdown(f'<div class="score-display"><div class="score-num" style="color:{color3}">{sc3_score}</div><div class="score-lbl">Points</div></div>', unsafe_allow_html=True)
                            with s2: st.markdown(f'<div class="score-display"><div class="score-num" style="color:{color3};font-size:1.4rem">{"PASS" if pf3=="PASS" else "FAIL"}</div><div class="score-lbl">Result</div></div>', unsafe_allow_html=True)
                            with s3: st.markdown(f'<div class="score-display"><div class="score-num" style="font-size:1rem;color:{color3}">{fb3.get("grade","")}</div><div class="score-lbl">Grade</div></div>', unsafe_allow_html=True)
                            if fb3.get("items_correct"): st.markdown(f'<div style="color:var(--accent-green);font-size:0.78rem;margin:4px 0">✓ {" · ".join(fb3["items_correct"])}</div>', unsafe_allow_html=True)
                            if fb3.get("items_missed"):  st.markdown(f'<div style="color:var(--accent-amber);font-size:0.78rem;margin:4px 0">⚠ Missed: {" · ".join(fb3["items_missed"])}</div>', unsafe_allow_html=True)
                            if fb3.get("items_incorrect"): st.markdown(f'<div style="color:var(--accent-red);font-size:0.78rem;margin:4px 0">✗ Wrong: {" · ".join(fb3["items_incorrect"])}</div>', unsafe_allow_html=True)
                            recs3 = st.session_state.rl_recommendations
                            if recs3: st.markdown(f'<div class="rl-box"><strong style="color:var(--accent-violet)">RL Recommendation</strong><br><br>{"<br>".join(recs3[:2])}</div>', unsafe_allow_html=True)
                        cr3 = fb3.get("specific_corrections","")
                        st.markdown(f'<div class="feedback-box"><strong>{"Correct Response" if pf3=="REPLAY" else "Instructor Feedback"}</strong><br><br>{fb3.get("overall_feedback","")}<br><br><strong>Corrected phrasing:</strong><br><em style="color:#7fe89a">{cr3}</em></div>', unsafe_allow_html=True)
                        st.markdown(f'<div style="color:var(--text-muted);font-size:0.78rem;margin-top:6px">💡 {fb3.get("coaching_tip","")}</div>', unsafe_allow_html=True)
                else:
                    st.markdown("""<div style="text-align:center;padding:50px 20px;border:1px dashed var(--border);
                    border-radius:6px;margin-top:16px">
                    <div style="font-size:2rem;opacity:0.1;margin-bottom:12px">📡</div>
                    <div style="font-family:var(--font-display);font-size:0.9rem;color:var(--text-muted);letter-spacing:0.08em">Awaiting Scenario</div>
                    <div style="font-family:var(--font-mono);font-size:0.62rem;margin-top:6px;color:var(--text-muted)">Select preset → Generate Scenario</div>
                    </div>""", unsafe_allow_html=True)

            # ── MODE B: PILOT INITIATES ───────────────────────────────────
            else:
                pi_sc = st.session_state.get("pi_scenario")
                pi_step = st.session_state.get("pi_step","init")

                if not pi_sc:
                    st.markdown("""<div style="text-align:center;padding:50px 20px;border:1px dashed var(--border);
                    border-radius:6px;margin-top:16px">
                    <div style="font-size:2rem;opacity:0.1;margin-bottom:12px">🎙️</div>
                    <div style="font-family:var(--font-display);font-size:0.9rem;color:var(--text-muted);letter-spacing:0.08em">You Speak First</div>
                    <div style="font-family:var(--font-mono);font-size:0.62rem;margin-top:6px;color:var(--text-muted)">Select preset → Set Up Situation</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    pi_s_type = pi_sc.get("scenario_type","Normal Traffic")
                    pi_is_urg = pi_s_type == "Emergency" or "mayday" in pi_sc.get("situation_briefing","").lower() or "pan-pan" in pi_sc.get("situation_briefing","").lower()

                    st.markdown("---")
                    st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;margin:8px 0 4px">
                    <span style="font-family:var(--font-mono);font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;
                    color:{'var(--accent-red)' if pi_is_urg else 'var(--accent-cyan)'}">{pi_s_type}</span>
                    <span style="font-family:var(--font-mono);font-size:0.6rem;background:rgba(0,200,232,0.08);color:var(--accent-cyan);border:1px solid rgba(0,200,232,0.2);border-radius:2px;padding:2px 8px">PILOT INITIATES</span>
                    <span style="font-family:var(--font-mono);font-size:0.6rem;color:var(--text-muted)">{pi_sc.get('scenario_id','')}</span>
                    </div>""", unsafe_allow_html=True)

                    st.markdown('<div class="section-label">Situation Briefing</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="pilot-box">{pi_sc.get("situation_briefing","")}</div>', unsafe_allow_html=True)

                    st.markdown('<div class="section-label">Your Task</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-family:var(--font-mono);font-size:0.72rem;color:var(--accent-cyan);padding:8px 12px;background:rgba(0,200,232,0.05);border-left:3px solid var(--accent-cyan);border-radius:0 3px 3px 0;margin:4px 0">{pi_sc.get("pilot_initial_context","")}</div>', unsafe_allow_html=True)

                    if pi_sc.get("key_pilot_call_items"):
                        st.markdown(f'<div style="font-family:var(--font-mono);font-size:0.62rem;color:var(--text-muted);margin:3px 0">Include: {" · ".join(pi_sc["key_pilot_call_items"])}</div>', unsafe_allow_html=True)

                    conv_hist = st.session_state.get("pi_conv_history",[])
                    if conv_hist:
                        st.markdown('<div class="section-label">Conversation Log</div>', unsafe_allow_html=True)
                        for ex in conv_hist:
                            if ex["role"] == "pilot":
                                pf_color = "#2dcc8f" if ex.get("pass_fail") == "PASS" else "#e84040" if ex.get("pass_fail") == "FAIL" else "#8890b0"
                                badge = f'<span style="font-family:var(--font-mono);font-size:0.55rem;background:rgba(45,204,143,0.1);color:{pf_color};border:1px solid {pf_color}44;border-radius:2px;padding:1px 6px;margin-left:6px">{ex.get("pass_fail","")}</span>' if ex.get("pass_fail") else ""
                                st.markdown(f'<div class="pilot-box" style="margin:3px 0"><span style="font-family:var(--font-mono);font-size:0.6rem;color:var(--accent-green);letter-spacing:0.1em">YOU</span>{badge}<br><span style="font-size:0.82rem">{ex["text"]}</span></div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="atc-box" style="margin:3px 0"><span style="font-family:var(--font-mono);font-size:0.6rem;color:var(--accent-blue);letter-spacing:0.1em">ATC</span><br><span style="font-size:0.82rem">{ex["text"]}</span></div>', unsafe_allow_html=True)

                    if pi_step == "pilot_call":
                        st.markdown("---")
                        st.markdown('<div class="section-label" style="color:var(--accent-cyan)">▶ Your Initial Call to ATC</div>', unsafe_allow_html=True)

                        whisper_readback_widget(state_key="whisper_pi", widget_key="mic_pi", label="Speak Your Call  ·  Whisper + Mistral Aviation Correction")

                        pi_call_input = st.text_area("Or type your transmission:",
                            value=st.session_state.get("whisper_pi",""),
                            placeholder="e.g. 'Mumbai Approach, Indigo 6E201, passing 8000ft, request ILS 27'",
                            height=70, key="m3_pi_call_input")

                        pi_voice_sub = st.session_state.get("whisper_pi_submit_now", False)

                        def do_pi_initial_call(call_text):
                            with st.spinner("Evaluating your call..."):
                                eval_res = evaluate_pilot_initial_call(pi_sc, call_text)
                                st.session_state.pi_pilot_call_eval = eval_res
                                st.session_state.pi_conv_history.append({"role":"pilot","text":call_text,"pass_fail":eval_res.get("pass_fail",""),"score":eval_res.get("score",0)})
                                st.session_state.total_attempts += 1
                                score_pi = eval_res.get("score",0)
                                st.session_state.session_score += score_pi
                                if eval_res.get("pass_fail") == "PASS": st.session_state.correct_responses += 1; st.session_state.streaks += 1
                                else: st.session_state.streaks = 0
                            with st.spinner("ATC is responding..."):
                                atc_resp = generate_atc_response_to_pilot(m3_airport, m3_aircraft, st.session_state.difficulty, call_text, [], pi_s_type)
                                st.session_state.pi_atc_response = atc_resp
                                st.session_state.pi_conv_history.append({"role":"atc","text":atc_resp.get("atc_transmission","")})
                            st.session_state.pi_step = "pilot_readback"
                            st.session_state.whisper_pi = ""; st.session_state.whisper_pi_submit_now = False
                            st.session_state.whisper_pi_rb = ""; st.session_state.pi_readback_eval = None

                        if pi_voice_sub:
                            vt_pi = st.session_state.get("whisper_pi","").strip()
                            if vt_pi: do_pi_initial_call(vt_pi); st.rerun()

                        pc1, pc2 = st.columns([2,1])
                        with pc1:
                            if st.button("Transmit to ATC", use_container_width=True, key="m3_pi_call_submit"):
                                txt_pi = st.session_state.get("m3_pi_call_input","").strip() or pi_call_input.strip()
                                if txt_pi: do_pi_initial_call(txt_pi); st.rerun()
                                else: st.warning("Enter your transmission first.")
                        with pc2:
                            if st.button("Show Model Call", use_container_width=True, key="m3_pi_call_hint"):
                                st.markdown(f'<div class="feedback-box" style="margin-top:6px"><strong style="color:var(--accent-cyan)">Model Call</strong><br><br><em style="color:#7fe89a">{pi_sc.get("model_pilot_call","")}</em><br><br><span style="color:var(--text-muted);font-size:0.72rem">💡 {pi_sc.get("coaching_notes","")}</span></div>', unsafe_allow_html=True)

                    elif pi_step == "pilot_readback":
                        if st.session_state.pi_pilot_call_eval:
                            ev = st.session_state.pi_pilot_call_eval; pf_c = "#2dcc8f" if ev.get("pass_fail")=="PASS" else "#e84040"
                            sc_v = ev.get("score",0)
                            sc_col1,sc_col2,sc_col3 = st.columns(3)
                            with sc_col1: st.markdown(f'<div class="score-display"><div class="score-num" style="color:{pf_c}">{sc_v}</div><div class="score-lbl">Call Score</div></div>', unsafe_allow_html=True)
                            with sc_col2: st.markdown(f'<div class="score-display"><div class="score-num" style="color:{pf_c};font-size:1.4rem">{"PASS" if ev.get("pass_fail")=="PASS" else "FAIL"}</div><div class="score-lbl">Result</div></div>', unsafe_allow_html=True)
                            with sc_col3: st.markdown(f'<div class="score-display"><div class="score-num" style="font-size:1rem;color:{pf_c}">{ev.get("grade","")}</div><div class="score-lbl">Grade</div></div>', unsafe_allow_html=True)
                            if ev.get("items_missed"): st.markdown(f'<div style="color:var(--accent-amber);font-size:0.78rem;margin:3px 0">⚠ Missed: {" · ".join(ev["items_missed"])}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="feedback-box" style="margin:6px 0"><strong>Call Feedback</strong><br><br>{ev.get("overall_feedback","")}<br><br><em style="color:#7fe89a">{ev.get("specific_corrections","")}</em></div>', unsafe_allow_html=True)

                        atc_resp = st.session_state.pi_atc_response or {}
                        atc_tx = atc_resp.get("atc_transmission","")
                        pi_is_urg_resp = "expedite" in atc_tx.lower() or "immediately" in atc_tx.lower() or "go around" in atc_tx.lower()

                        st.markdown("---")
                        st.markdown('<div class="section-label">ATC Response</div>', unsafe_allow_html=True)
                        atc_tts_pi = build_tts_player(atc_tx, st.session_state.difficulty, emergency=pi_is_urg_resp)
                        st.components.v1.html(atc_tts_pi, height=155)

                        show_atc_pi = st.checkbox("Show ATC transcript", key="m3_pi_show_atc")
                        if show_atc_pi:
                            st.markdown(f'<div class="atc-box">{atc_tx}</div>', unsafe_allow_html=True)

                        if atc_resp.get("key_readback_items"):
                            st.markdown(f'<div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);margin:4px 0">Read back: {" · ".join(atc_resp["key_readback_items"])}</div>', unsafe_allow_html=True)

                        st.markdown('<div class="section-label" style="color:var(--accent-green)">▶ Your Readback to ATC</div>', unsafe_allow_html=True)
                        whisper_readback_widget(state_key="whisper_pi_rb", widget_key="mic_pi_rb", label="Voice Readback  ·  Whisper + Mistral Aviation Correction")

                        pi_rb_input = st.text_area("Or type your readback:",
                            value=st.session_state.get("whisper_pi_rb",""),
                            placeholder="Read back the ATC clearance/instructions...",
                            height=70, key="m3_pi_rb_input")

                        pi_rb_voice_sub = st.session_state.get("whisper_pi_rb_submit_now", False)

                        def do_pi_readback(rb_text):
                            with st.spinner("Evaluating readback..."):
                                rb_eval = evaluate_response(atc_resp, rb_text)
                                st.session_state.pi_readback_eval = rb_eval
                                st.session_state.pi_conv_history.append({"role":"pilot","text":rb_text,"pass_fail":rb_eval.get("pass_fail",""),"score":rb_eval.get("score",0)})
                                st.session_state.total_attempts += 1
                                sc_rb = rb_eval.get("score",0); st.session_state.session_score += sc_rb
                                if rb_eval.get("pass_fail")=="PASS": st.session_state.correct_responses+=1; st.session_state.streaks+=1
                                else: st.session_state.streaks=0
                                rl_update_weak_areas(pi_s_type, rb_eval.get("items_missed",[]), rb_eval.get("items_incorrect",[]))
                                rl_generate_recommendations()
                                st.session_state.history.append({"time":datetime.now().strftime("%H:%M:%S"),"type":pi_s_type,"score":sc_rb,
                                    "grade":rb_eval.get("grade","N/A"),"pass_fail":rb_eval.get("pass_fail",""),"scenario_id":pi_sc.get("scenario_id",""),"items_missed":rb_eval.get("items_missed",[])})
                            st.session_state.pi_step = "continue"
                            st.session_state.whisper_pi_rb = ""; st.session_state.whisper_pi_rb_submit_now = False

                        if pi_rb_voice_sub:
                            vt_rb = st.session_state.get("whisper_pi_rb","").strip()
                            if vt_rb: do_pi_readback(vt_rb); st.rerun()

                        rb1, rb2 = st.columns([2,1])
                        with rb1:
                            if st.button("Submit Readback", use_container_width=True, key="m3_pi_rb_submit"):
                                txt_rb = st.session_state.get("m3_pi_rb_input","").strip() or pi_rb_input.strip()
                                if txt_rb: do_pi_readback(txt_rb); st.rerun()
                                else: st.warning("Enter your readback first.")
                        with rb2:
                            if st.button("Show Answer", use_container_width=True, key="m3_pi_rb_hint"):
                                st.markdown(f'<div class="feedback-box" style="margin-top:6px"><strong style="color:var(--accent-green)">Model Readback</strong><br><br><em style="color:#7fe89a">{atc_resp.get("correct_response","")}</em></div>', unsafe_allow_html=True)

                    elif pi_step == "continue":
                        rb_ev = st.session_state.pi_readback_eval or {}
                        if rb_ev:
                            pf_rb = "#2dcc8f" if rb_ev.get("pass_fail")=="PASS" else "#e84040"
                            sc_rb = rb_ev.get("score",0)
                            st.markdown("---")
                            rc1,rc2,rc3 = st.columns(3)
                            with rc1: st.markdown(f'<div class="score-display"><div class="score-num" style="color:{pf_rb}">{sc_rb}</div><div class="score-lbl">Readback</div></div>', unsafe_allow_html=True)
                            with rc2: st.markdown(f'<div class="score-display"><div class="score-num" style="color:{pf_rb};font-size:1.4rem">{"PASS" if rb_ev.get("pass_fail")=="PASS" else "FAIL"}</div><div class="score-lbl">Result</div></div>', unsafe_allow_html=True)
                            with rc3: st.markdown(f'<div class="score-display"><div class="score-num" style="font-size:1rem;color:{pf_rb}">{rb_ev.get("grade","")}</div><div class="score-lbl">Grade</div></div>', unsafe_allow_html=True)
                            if rb_ev.get("items_correct"): st.markdown(f'<div style="color:var(--accent-green);font-size:0.78rem;margin:3px 0">✓ {" · ".join(rb_ev["items_correct"])}</div>', unsafe_allow_html=True)
                            if rb_ev.get("items_missed"): st.markdown(f'<div style="color:var(--accent-amber);font-size:0.78rem;margin:3px 0">⚠ Missed: {" · ".join(rb_ev["items_missed"])}</div>', unsafe_allow_html=True)
                            if rb_ev.get("items_incorrect"): st.markdown(f'<div style="color:var(--accent-red);font-size:0.78rem;margin:3px 0">✗ Wrong: {" · ".join(rb_ev["items_incorrect"])}</div>', unsafe_allow_html=True)
                            cr_rb = rb_ev.get("specific_corrections","")
                            st.markdown(f'<div class="feedback-box"><strong>Readback Feedback</strong><br><br>{rb_ev.get("overall_feedback","")}<br><br><em style="color:#7fe89a">{cr_rb}</em></div>', unsafe_allow_html=True)
                            st.markdown(f'<div style="color:var(--text-muted);font-size:0.78rem;margin-top:4px">💡 {rb_ev.get("coaching_tip","")}</div>', unsafe_allow_html=True)
                            recs_pi = st.session_state.rl_recommendations
                            if recs_pi: st.markdown(f'<div class="rl-box"><strong style="color:var(--accent-violet)">RL Recommendation</strong><br><br>{"<br>".join(recs_pi[:2])}</div>', unsafe_allow_html=True)

                        st.markdown("---")
                        st.markdown('<div class="section-label">Continue Conversation?</div>', unsafe_allow_html=True)
                        cont1, cont2 = st.columns(2)
                        with cont1:
                            if st.button("📡  Continue — I Call ATC Again", use_container_width=True, key="m3_pi_continue"):
                                with st.spinner("Generating next situation..."):
                                    next_ex = generate_next_exchange(m3_airport, m3_aircraft, st.session_state.difficulty,
                                        [{"role":ex["role"],"text":ex["text"]} for ex in st.session_state.pi_conv_history[-6:]], pi_s_type)
                                    st.session_state.pi_atc_response = next_ex
                                    next_atc_text = next_ex.get("atc_transmission","")
                                    st.session_state.pi_conv_history.append({"role":"atc","text":next_atc_text})
                                st.session_state.pi_pilot_call_eval = None; st.session_state.pi_readback_eval = None
                                st.session_state.whisper_pi_rb = ""
                                st.session_state.pi_step = "pilot_readback"
                                st.rerun()
                        with cont2:
                            if st.button("🔁  New Scenario from Scratch", use_container_width=True, key="m3_pi_new"):
                                st.session_state.pi_scenario = None; st.session_state.pi_step = "init"
                                st.session_state.pi_conv_history = []; st.session_state.pi_pilot_call_eval = None
                                st.session_state.pi_atc_response = None; st.session_state.pi_readback_eval = None
                                st.session_state.whisper_pi = ""; st.session_state.whisper_pi_rb = ""
                                st.rerun()

        with col_info3:
            ac3 = m3_airport.split(" - ")[0]
            st.markdown(f'<div class="score-display" style="margin-bottom:8px"><div class="score-num" style="font-size:1.2rem">{ac3}</div><div class="score-lbl">Airport</div></div>', unsafe_allow_html=True)
            db3 = DIFFICULTY_LEVELS[st.session_state.difficulty]["badge"]
            st.markdown(f'<span class="diff-tag {db3}">{st.session_state.difficulty}</span>', unsafe_allow_html=True)
            np3 = int(DIFFICULTY_LEVELS[st.session_state.difficulty]["noise"]*100)
            st.markdown(f'<div style="font-family:var(--font-mono);font-size:0.6rem;color:var(--text-muted);margin-top:5px">Noise: {np3}%</div>', unsafe_allow_html=True)
            st.progress(DIFFICULTY_LEVELS[st.session_state.difficulty]["noise"])
            st.markdown("<br>", unsafe_allow_html=True)
            streak_str = "🔥"*min(st.session_state.streaks,5) if st.session_state.streaks else "—"
            st.markdown(f'<div style="text-align:center;font-size:1.2rem;margin:4px 0">{streak_str}</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-family:var(--font-mono);font-size:0.55rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--text-muted);text-align:center">Streak</div>', unsafe_allow_html=True)
            if pilot_initiates and st.session_state.pi_conv_history:
                st.markdown("<br>", unsafe_allow_html=True)
                pi_exchanges = len([e for e in st.session_state.pi_conv_history if e["role"]=="pilot"])
                st.markdown(f'<div class="score-display"><div class="score-num" style="font-size:1.2rem">{pi_exchanges}</div><div class="score-lbl">Transmissions</div></div>', unsafe_allow_html=True)

    # ── M3 Tab B — Chain Session ─────────────────────────────────────────────
    with m3_inner[1]:
        st.markdown('<div class="section-label">Full Flight Conversation Chain</div>', unsafe_allow_html=True)
        st.markdown('<div class="rl-box">Practice a complete flight from taxi to parking. RL engine evaluates each exchange and generates the next ATC instruction automatically.</div>', unsafe_allow_html=True)

        if not st.session_state.conv_active:
            chain_c1,chain_c2=st.columns(2)
            with chain_c1:
                chain_sit=st.selectbox("Starting situation",["Full flight from pushback to parking","Departure sequence","Arrival sequence","Emergency — pilot initiates"],key="chain_sit_sel")
                chain_type=st.selectbox("Traffic type",["Normal Traffic","Heavy Traffic","Emergency"],key="chain_type_sel")
            with chain_c2:
                if st.button("Start Chain Session",use_container_width=True,key="chain_start_btn"):
                    with st.spinner("Generating first exchange..."):
                        raw_ch="".join(generate_scenario(m3_airport,m3_aircraft,chain_type,st.session_state.difficulty,chain_sit))
                    try:
                        clean_ch=raw_ch.strip().lstrip("```").lstrip("json").rstrip("```").strip()
                        first_sc=json.loads(clean_ch)
                        st.session_state.conv_active=True; st.session_state.conv_step=1
                        st.session_state.conv_render_id=0; st.session_state.chain_type_active=chain_type
                        st.session_state.conv_scenario_id=first_sc.get("scenario_id","CHAIN")
                        st.session_state.conv_chain=[{"role":"atc","text":first_sc.get("atc_transmission",""),
                            "key_items":first_sc.get("key_readback_items",[]),"correct_resp":first_sc.get("correct_response",""),
                            "coaching":first_sc.get("coaching_notes",""),"scored":False,"score":0,"phase":"initial"}]
                        st.session_state.whisper_chain=""
                        st.session_state.whisper_chain_submit_now=False
                        st.session_state.feedback=None
                    except Exception as e: st.error(f"Failed: {e}")
                    st.rerun()
        else:
            chain=st.session_state.conv_chain; step=st.session_state.conv_step
            render_id=st.session_state.get("conv_render_id",0)
            active_chain_type=st.session_state.get("chain_type_active","Normal Traffic")
            phases=["taxi","departure","climb","cruise","descent","approach","landing","vacate","complete"]
            current_phase="taxi"
            for ex in reversed(chain):
                if ex.get("phase") and ex["phase"] not in ("initial",): current_phase=ex["phase"]; break
            phase_idx=phases.index(current_phase) if current_phase in phases else 0

            st.markdown(f"""<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
            <span style="font-family:var(--font-mono);font-size:0.62rem;color:var(--text-muted)">Phase:</span>
            <span style="font-family:var(--font-mono);font-size:0.65rem;color:var(--accent-cyan);text-transform:uppercase">{current_phase}</span>
            <span style="font-family:var(--font-mono);font-size:0.6rem;color:var(--text-muted)">Exchange {step}</span>
            </div>""", unsafe_allow_html=True)
            st.progress(min(phase_idx/8,1.0))

            st.markdown('<div class="section-label">Conversation Log</div>', unsafe_allow_html=True)
            for ex in chain:
                role=ex.get("role","atc"); scored=ex.get("scored",False)
                score_badge=(f' <span style="background:rgba(45,204,143,0.12);color:var(--accent-green);padding:1px 7px;border-radius:10px;font-size:0.6rem">{ex.get("score",0)} pts</span>'
                             if scored and role=="pilot" else "")
                if role=="atc":
                    st.markdown(f'<div class="chain-entry" style="border-left:2px solid var(--accent-blue)"><span style="color:var(--accent-blue);font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase">ATC</span>{score_badge}<br><span style="color:var(--text-secondary)">{ex["text"]}</span></div>',unsafe_allow_html=True)
                elif role=="pilot":
                    pf_c="var(--accent-green)" if ex.get("pass_fail")=="PASS" else "var(--accent-amber)" if ex.get("pass_fail")=="FAIL" else "var(--text-secondary)"
                    st.markdown(f'<div class="chain-entry" style="border-left:2px solid var(--accent-green)"><span style="color:var(--accent-green);font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase">Pilot</span>{score_badge}<br><span style="color:{pf_c}">{ex["text"]}</span></div>',unsafe_allow_html=True)

            pending_atc=None
            for ex in reversed(chain):
                if ex["role"]=="atc" and not ex.get("scored",False): pending_atc=ex; break

            if pending_atc:
                st.markdown(f'<div class="section-label">Exchange {step} — Read Back</div>', unsafe_allow_html=True)
                urg_ch=any(k in pending_atc["text"].lower() for k in ["tcas","windshear","go around","mayday","expedite"])
                tts_ch=build_tts_player(pending_atc["text"],st.session_state.difficulty,emergency=urg_ch)
                st.components.v1.html(tts_ch,height=155)
                if pending_atc.get("key_items"):
                    st.markdown(f'<div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);margin:4px 0">Read back: {" · ".join(pending_atc["key_items"])}</div>',unsafe_allow_html=True)

                whisper_readback_widget(
                    state_key="whisper_chain",
                    widget_key=f"mic_chain_{step}_{render_id}",
                    label="Voice Readback  ·  Whisper + Mistral Aviation Correction"
                )

                chain_input=st.text_area("Or type readback:",value=st.session_state.get("whisper_chain",""),
                    height=70,key=f"chain_inp_{step}_{render_id}",placeholder="Speak above or type here...")

                voice_chain_submitted=st.session_state.get("whisper_chain_submit_now",False)

                def do_chain_rb(submit_text):
                    with st.spinner("Evaluating + generating next ATC call..."):
                        fb=evaluate_response(pending_atc,submit_text,is_chain=True); score_ch=fb.get("score",0)
                        for ex in st.session_state.conv_chain:
                            if ex is pending_atc: ex["scored"]=True; ex["score"]=score_ch
                        st.session_state.conv_chain.append({"role":"pilot","text":submit_text,"pass_fail":fb.get("pass_fail",""),"score":score_ch,"scored":True,"phase":current_phase,"feedback":fb})
                        st.session_state.total_attempts+=1; st.session_state.session_score+=score_ch
                        if fb.get("pass_fail")=="PASS": st.session_state.correct_responses+=1; st.session_state.streaks+=1
                        else: st.session_state.streaks=0
                        rl_update_weak_areas(active_chain_type,fb.get("items_missed",[]),fb.get("items_incorrect",[]))
                        hist_ch=[{"role":ex["role"],"text":ex["text"]} for ex in st.session_state.conv_chain[-8:]]
                        next_ex=generate_next_exchange(m3_airport,m3_aircraft,st.session_state.difficulty,hist_ch,active_chain_type)
                        if next_ex.get("session_complete"):
                            st.session_state.conv_chain.append({"role":"atc","text":"SESSION COMPLETE — Aircraft at gate.","key_items":[],"correct_resp":"","coaching":"","scored":True,"score":0,"phase":"complete"})
                            st.session_state.conv_active=False
                        else:
                            st.session_state.conv_chain.append({"role":"atc","text":next_ex.get("atc_transmission",""),"key_items":next_ex.get("key_readback_items",[]),"correct_resp":next_ex.get("correct_response",""),"coaching":next_ex.get("coaching_notes",""),"scored":False,"score":0,"phase":next_ex.get("flight_phase","en-route")})
                        st.session_state.conv_step+=1; st.session_state.conv_render_id=render_id+1
                        st.session_state.conv_pending_eval=True; rl_generate_recommendations()
                        st.session_state.whisper_chain=""; st.session_state.whisper_chain_submit_now=False

                if voice_chain_submitted:
                    vt=st.session_state.get("whisper_chain","").strip()
                    if vt: do_chain_rb(vt); st.rerun()

                cc1,cc2,cc3=st.columns(3)
                with cc1:
                    if st.button("Submit & Next",use_container_width=True,key=f"ch_sub_{step}_{render_id}"):
                        txt_ch=st.session_state.get(f"chain_inp_{step}_{render_id}","").strip() or chain_input.strip()
                        if txt_ch: do_chain_rb(txt_ch); st.rerun()
                        else: st.warning("Enter readback first.")
                with cc2:
                    if st.button("Show Hint",use_container_width=True,key=f"ch_hint_{step}_{render_id}"):
                        st.markdown(f'<div class="feedback-box">✓ {pending_atc.get("correct_resp","")}<br><br>💡 {pending_atc.get("coaching","")}</div>',unsafe_allow_html=True)
                with cc3:
                    if st.button("Skip",use_container_width=True,key=f"ch_skip_{step}_{render_id}"):
                        for ex in st.session_state.conv_chain:
                            if ex is pending_atc: ex["scored"]=True; ex["score"]=0
                        st.session_state.conv_chain.append({"role":"pilot","text":"[SKIPPED]","pass_fail":"SKIP","score":0,"scored":True,"phase":current_phase,"feedback":{}})
                        hist_sk=[{"role":ex["role"],"text":ex["text"]} for ex in st.session_state.conv_chain[-6:]]
                        nx_sk=generate_next_exchange(m3_airport,m3_aircraft,st.session_state.difficulty,hist_sk,active_chain_type)
                        if not nx_sk.get("session_complete"):
                            st.session_state.conv_chain.append({"role":"atc","text":nx_sk.get("atc_transmission",""),"key_items":nx_sk.get("key_readback_items",[]),"correct_resp":nx_sk.get("correct_response",""),"coaching":nx_sk.get("coaching_notes",""),"scored":False,"score":0,"phase":nx_sk.get("flight_phase","en-route")})
                        st.session_state.conv_step+=1; st.session_state.conv_render_id=render_id+1
                        st.session_state.whisper_chain=""
                        st.rerun()

                if st.session_state.conv_pending_eval:
                    last_pilot=next((ex for ex in reversed(st.session_state.conv_chain) if ex["role"]=="pilot" and ex.get("feedback")),None)
                    if last_pilot:
                        fb_lp=last_pilot["feedback"]; pf_lp=fb_lp.get("pass_fail","")
                        c_lp="var(--accent-green)" if pf_lp=="PASS" else "var(--accent-red)"
                        st.markdown(f'<div class="feedback-box" style="margin-top:10px"><strong style="color:{c_lp}">{pf_lp} — {fb_lp.get("grade","")}</strong> · {fb_lp.get("score",0)} pts<br><br>{fb_lp.get("overall_feedback","")}<br><span style="color:var(--text-muted);font-size:0.75rem">💡 {fb_lp.get("coaching_tip","")}</span></div>',unsafe_allow_html=True)
                        st.session_state.conv_pending_eval=False
            else:
                total_ch=sum(ex.get("score",0) for ex in chain if ex["role"]=="pilot")
                total_ex=sum(1 for ex in chain if ex["role"]=="pilot")
                st.markdown(f"""<div style="text-align:center;padding:28px;background:rgba(45,204,143,0.04);
                border:1px solid rgba(45,204,143,0.15);border-radius:6px;margin:10px 0">
                <div style="font-family:var(--font-display);font-size:1.1rem;color:var(--accent-green);letter-spacing:0.06em;margin:8px 0">Session Complete</div>
                <div style="font-family:var(--font-mono);font-size:0.75rem;color:var(--text-secondary)">{total_ch} pts · {total_ex} exchanges</div>
                </div>""", unsafe_allow_html=True)
                if st.button("Start New Chain",use_container_width=True,key="chain_new"):
                    st.session_state.conv_active=False; st.session_state.conv_chain=[]
                    st.session_state.conv_step=0; st.rerun()

    # ── M3 Tab C — Coaching ──────────────────────────────────────────────────
    with m3_inner[2]:
        st.markdown('<div class="section-label">ATC Phraseology Reference</div>', unsafe_allow_html=True)
        coc1,coc2=st.columns(2)
        with coc1:
            st.markdown("""<div class="guide-card"><strong>Standard Readback Format</strong>
[Callsign] + [Readback items] + [Callsign]<br><br>
ATC: "Golf Alpha Bravo, runway 27L, cleared for takeoff, wind 270/12."<br><br>
✓ Pilot: "Runway 27 left, cleared for takeoff, Golf Alpha Bravo."<br><br>
<span style="color:var(--text-muted);font-size:0.72rem">Always read back: Runway · Clearances · Altitudes · Headings · Frequencies · Squawk</span>
</div>""", unsafe_allow_html=True)
            st.markdown("""<div class="guide-card" style="border-left:2px solid var(--accent-red)"><strong style="color:var(--accent-red)">Urgent — Brevity = Correctness</strong>
TCAS RA: <em>"Unable, TCAS RA, [CS]"</em><br>
Go-around: <em>"Going around, [CS]"</em><br>
Traffic: <em>"Traffic in sight, [CS]"</em><br>
MAYDAY: <em>"MAYDAY MAYDAY MAYDAY, [CS], [nature], [intention]"</em><br><br>
<span style="color:var(--text-muted);font-size:0.72rem">7700=Emergency · 7600=Radio Fail · 7500=Hijack</span>
</div>""", unsafe_allow_html=True)
        with coc2:
            st.markdown("""<div class="guide-card"><strong>Aviation Number Pronunciation</strong>
Frequencies: one two one decimal tree<br>
Headings: two seven zero<br>
Flight levels: flight level tree fife zero<br>
Altitudes: four thousand feet<br>
Runways: runway two seven left<br>
Squawk: seven seven zero zero<br><br>
<span style="color:var(--text-muted);font-size:0.72rem">3=tree · 5=fife · 9=niner</span>
</div>""", unsafe_allow_html=True)
            st.markdown("""<div class="guide-card"><strong>NATO Phonetic Alphabet</strong>
Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliet Kilo Lima Mike November Oscar Papa Quebec Romeo Sierra Tango Uniform Victor Whiskey X-Ray Yankee Zulu
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label">Ask the ATC Coach</div>', unsafe_allow_html=True)
        cq3=st.text_input("Question:",placeholder="e.g. What is the correct readback for a TCAS RA climb?",key="m3_coach_q")
        if st.button("Get Answer",key="m3_coach_btn") and cq3:
            with st.spinner("Consulting ATC manual..."):
                r=mistral_chat(f"Expert ATC instructor, concise ICAO-standard answer: {cq3}",max_tokens=400)
                ans=r.json()["choices"][0]["message"]["content"]
            st.markdown(f'<div class="feedback-box">{ans}</div>',unsafe_allow_html=True)

    # ── M3 Tab D — RL Engine ─────────────────────────────────────────────────
    with m3_inner[3]:
        st.markdown('<div class="section-label">Reinforcement Learning — Q-Learning Adaptive Curriculum</div>', unsafe_allow_html=True)
        st.markdown('<div class="rl-box">Q-Learning maps (difficulty, weak_area) → scenario_type with reward=score/100.<br>ε=0.2 exploration · α=0.3 learning rate · γ=0.7 discount</div>', unsafe_allow_html=True)
        rl_c1,rl_c2=st.columns(2)
        with rl_c1:
            st.markdown("**Q-Table**"); qt3=st.session_state.rl_q_table
            if qt3:
                rows3=[{"State":str(s3),"Action":a3,"Q-Value":round(v3,3)} for s3,av3 in qt3.items() for a3,v3 in av3.items()]
                if rows3: st.dataframe(pd.DataFrame(rows3),use_container_width=True)
            else: st.info("Complete RL-guided scenarios to populate.")
        with rl_c2:
            st.markdown("**Weak Areas**")
            if st.session_state.rl_weak_areas:
                rows_rl=[{"Type":st3,"Attempts":ss3["attempts"],"Errors":ss3["errors"],"E/A":round(ss3["errors"]/max(ss3["attempts"],1),2)} for st3,ss3 in st.session_state.rl_weak_areas.items()]
                st.dataframe(pd.DataFrame(rows_rl),use_container_width=True)
            else: st.info("No data yet.")
        st.markdown("**Recommendations**")
        recs_rl=rl_generate_recommendations()
        if recs_rl:
            for r_rl in recs_rl: st.markdown(f'<div class="rl-box" style="padding:8px 14px;margin:3px 0">{r_rl}</div>',unsafe_allow_html=True)
        else: st.info("Complete more scenarios to generate recommendations.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-label">Session Performance Dashboard</div>', unsafe_allow_html=True)
    p_c1,p_c2,p_c3,p_c4=st.columns(4)
    with p_c1: st.metric("Total Score",st.session_state.session_score)
    with p_c2: st.metric("Accuracy",f"{acc_pct:.1f}%")
    with p_c3: st.metric("Attempts",st.session_state.total_attempts)
    with p_c4: st.metric("Streak",st.session_state.streaks)

    if st.session_state.history:
        st.markdown('<div class="section-label">Score History</div>', unsafe_allow_html=True)
        scores_h=[h["score"] for h in st.session_state.history]
        df_h=pd.DataFrame({"Attempt":list(range(1,len(scores_h)+1)),"Score":scores_h}).set_index("Attempt")
        st.line_chart(df_h,use_container_width=True)

        st.markdown('<div class="section-label">By Scenario Type</div>', unsafe_allow_html=True)
        ts_h={}
        for h in st.session_state.history:
            t_h=h.get("type","Unknown")
            if t_h not in ts_h: ts_h[t_h]={"count":0,"passed":0,"total":0}
            ts_h[t_h]["count"]+=1; ts_h[t_h]["total"]+=h.get("score",0)
            if h.get("pass_fail")=="PASS": ts_h[t_h]["passed"]+=1
        for t_h,s_h in ts_h.items():
            pct_h=(s_h["passed"]/s_h["count"])*100; avg_h=s_h["total"]/s_h["count"]
            icon_h=SCENARIO_TYPES.get(t_h,{}).get("icon","📻")
            st.markdown(f"**{icon_h} {t_h}** — {s_h['count']} attempts · {pct_h:.0f}% pass rate · avg {avg_h:.0f} pts")
            st.progress(pct_h/100)

        st.markdown('<div class="section-label">Session Log</div>', unsafe_allow_html=True)
        for i,item_h in enumerate(reversed(st.session_state.history)):
            pf_c_h="var(--accent-green)" if item_h.get("pass_fail")=="PASS" else "var(--accent-red)"
            icon_h=SCENARIO_TYPES.get(item_h.get("type",""),{}).get("icon","📻")
            ms_h=item_h.get("items_missed",[])
            ms_str=f" · missed: {', '.join(ms_h[:2])}" if ms_h else ""
            st.markdown(f"""<div class="history-row">
            <span style="color:var(--text-muted)">#{len(st.session_state.history)-i:02d}</span>
            <span>{item_h.get('time','')}</span>
            <span>{icon_h} {item_h.get('type','')}</span>
            <span style="color:{pf_c_h};font-weight:600">{item_h.get('pass_fail','')}</span>
            <span style="color:var(--accent-cyan)">{item_h.get('score',0)} pts</span>
            <span style="color:var(--text-muted)">{item_h.get('grade','')}</span>
            <span style="color:var(--accent-red);font-size:0.7rem">{ms_str}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style="text-align:center;padding:50px;border:1px dashed var(--border);
        border-radius:6px">
        <div style="font-family:var(--font-display);font-size:0.9rem;color:var(--text-muted)">No Data Yet</div>
        <div style="font-family:var(--font-mono);font-size:0.62rem;margin-top:6px;color:var(--text-muted)">Complete scenarios to see performance analytics</div>
        </div>""",unsafe_allow_html=True)

    if st.session_state.m1_history:
        st.markdown('<div class="section-label">Fatigue Analysis Trend (Module 1)</div>', unsafe_allow_html=True)
        fig_m1t=fig_session_trend(st.session_state.m1_history)
        if fig_m1t:
            st.plotly_chart(fig_m1t,use_container_width=True,config={'displayModeBar':False},key="m1_trend_chart")


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-top:48px;padding:12px 0;border-top:1px solid var(--border);
font-family:var(--font-mono);font-size:0.52rem;color:var(--text-muted);letter-spacing:0.12em;
text-transform:uppercase;display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px">
    <span>AviSuite · Aviation Speech Intelligence · Fatigue Analyzer · Pilot Whisperer · ScenarioSynth</span>
    <span>Research Use Only · Not Certified · Not a Medical or Safety Device · ICAO · FAR/EASA</span>
</div>
""", unsafe_allow_html=True)
