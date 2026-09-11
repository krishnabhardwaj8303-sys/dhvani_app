import os
import subprocess
import threading
import imageio_ffmpeg
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

import streamlit as st
import numpy as np
import io
import tempfile
import time
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt
from streamlit_mic_recorder import mic_recorder
from streamlit_autorefresh import st_autorefresh

from dhvani_core import DhvaniPipeline, CLASS_NAMES, SR

st.set_page_config(page_title="DHVANI", page_icon=":dart:", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #0a0e14 0%, #0d1420 100%); }
    #MainMenu, footer, header {visibility: hidden;}
    .hero-tag { color: #ff5722; font-size: 13px; letter-spacing: 3px; font-weight: 600; margin-bottom: 8px; }
    .hero-title { font-size: 56px; font-weight: 900; color: #ffffff; line-height: 1; margin: 0; }
    .hero-title span { color: #ff5722; }
    .hero-sub { font-size: 18px; color: #cbd5e1; margin-top: 8px; font-weight: 500; }
    .status-badge { display: inline-block; background: rgba(0,255,136,0.1); border: 1px solid #00ff88; color: #00ff88; padding: 4px 14px; border-radius: 20px; font-size: 12px; margin-top: 14px; }
    .section-title { font-size: 26px; font-weight: 800; color: #ffffff; margin-bottom: 4px; }
    .section-title span { color: #ff5722; }
    .metric-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 18px; }
    .metric-label { color: #9ca3af; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: white; font-size: 26px; font-weight: 800; margin-top: 4px; }
    .metric-orange { color: #ff5722; }
    .metric-blue { color: #4fc3f7; }
    .status-panel { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,87,34,0.3); border-radius: 12px; padding: 20px; }
    .status-row { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.06); color: #cbd5e1; font-size: 14px; }
    .status-online { color: #00ff88; font-weight: 700; }
    .status-warn { color: #ffc107; font-weight: 700; }
    .stButton>button { background: linear-gradient(90deg, #ff5722, #ff8a50); color: white; font-weight: 700; border-radius: 8px; border: none; padding: 12px 28px; width: 100%; }
    .stTabs [aria-selected="true"] { color: #ff5722 !important; border-bottom-color: #ff5722 !important; }
    .footer-text { color: #6b7280; font-size: 12px; text-align: center; padding: 20px 0; }
    .class-legend { display:flex; gap:12px; margin:10px 0; flex-wrap:wrap; }
    .class-chip { padding:4px 12px; border-radius:20px; font-size:11px; font-weight:700; }
    .device-box { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 20px; }
    .device-title { font-size: 18px; font-weight: 800; color: #ff5722; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-tag">SIH 2026 - DRDO - AI DEFENCE TECHNOLOGY</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">DHV<span>ANI</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-Powered Adaptive Noise Cancellation for Defence Communications</div>', unsafe_allow_html=True)
st.markdown('<div class="status-badge">AI ENGINE READY</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

CLASS_COLORS = {-1: "#607d8b", 0: "#ff9800", 1: "#ffc107", 2: "#f44336", 3: "#00ff88"}
FRAME_LEN = 1600

@st.cache_resource
def load_pipeline():
    return DhvaniPipeline(
        classifier_path="dhvani_model_multiclass.pkl",
        enhancer_path="dhvani_enhancement_model.pt",
    )

pipeline = load_pipeline()
bilstm_ready = pipeline.enhancer_loaded

# ---- Shared "radio mailbox" for the Field Comms Relay demo ----
# A process-level dict shared by ALL browser sessions connected to this
# Streamlit app, simulating a single shared radio channel between two
# field devices. Not persisted to disk -- resets when the app restarts.
@st.cache_resource
def get_mailbox():
    return {"lock": threading.Lock(), "audio_bytes": None, "sr": None,
            "timestamp": None, "sender": None, "seq": 0}

mailbox = get_mailbox()


def load_mic_audio(raw_bytes, target_sr=16000):
    in_path = None
    out_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(raw_bytes)
            in_path = tmp.name
        out_path = in_path.replace(".webm", ".wav")

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        result = subprocess.run(
            [ffmpeg_exe, "-y", "-i", in_path, "-ar", str(target_sr), "-ac", "1", out_path],
            capture_output=True, text=True
        )
        if result.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")

        sr, audio = wavfile.read(out_path)
        if audio.dtype != np.int16:
            audio = (audio / (np.max(np.abs(audio)) + 1e-8) * 32767).astype(np.int16)
        return sr, audio
    finally:
        for p in (in_path, out_path):
            if p and os.path.exists(p):
                os.remove(p)


def plot_comparison(original, cleaned, class_log, frame_len=FRAME_LEN):
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.5))
    fig.patch.set_facecolor('#0a0e14')
    axes[0].plot(original, color='#ff5722', linewidth=0.6)
    axes[0].set_title("BEFORE - Noisy Input", color='white', fontsize=10)
    axes[0].set_facecolor('#0d1420')
    axes[0].tick_params(colors='#6b7280')
    for spine in axes[0].spines.values(): spine.set_color('#2d3748')

    axes[1].plot(cleaned, color='#4fc3f7', linewidth=0.6)
    for i, cls in enumerate(class_log):
        if cls in CLASS_COLORS:
            axes[1].axvspan(i*frame_len, (i+1)*frame_len, color=CLASS_COLORS[cls], alpha=0.12)
    method = "BiLSTM Complex-Mask + Classifier Gating" if bilstm_ready else "Classifier Gain-Gating (BiLSTM not loaded)"
    axes[1].set_title(f"AFTER - DHVANI Output ({method})", color='white', fontsize=10)
    axes[1].set_facecolor('#0d1420')
    axes[1].tick_params(colors='#6b7280')
    for spine in axes[1].spines.values(): spine.set_color('#2d3748')
    plt.tight_layout()
    return fig


def calculate_snr(clean, noisy):
    noise = noisy - clean
    signal_power = np.mean(clean**2)
    noise_power = np.mean(noise**2) + 1e-10
    return max(10 * np.log10(signal_power / noise_power), 0)


def show_results(sr, audio, use_lms, use_bilstm=False):
    if audio.ndim > 1:
        audio = audio[:, 0]

    result = pipeline.process(sr, audio, use_lms=use_lms, use_bilstm=use_bilstm)
    cleaned_float = result["cleaned"]
    class_log = result["class_log"]
    class_counts = result["class_counts"]

    cleaned_int16 = (np.clip(cleaned_float, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, cleaned_int16)
    buf.seek(0)

    audio_float = audio.astype(np.float32) / 32767.0
    min_len = min(len(audio_float), len(cleaned_float))
    snr = calculate_snr(cleaned_float[:min_len], audio_float[:min_len])
    avg_conf = np.mean(result["conf_log"]) if result["conf_log"] else 0
    spike_count = sum(result["spike_log"])

    st.markdown('<div class="class-legend">', unsafe_allow_html=True)
    for cid, cname in CLASS_NAMES.items():
        pct = round((class_counts.get(cid, 0) / max(result["num_frames"], 1)) * 100, 1)
        st.markdown(f'<span class="class-chip" style="background:{CLASS_COLORS.get(cid,"#888")}22; color:{CLASS_COLORS.get(cid,"#888")}; border:1px solid {CLASS_COLORS.get(cid,"#888")}55;">{cname}: {pct}%</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    fig = plot_comparison(audio_float, cleaned_float, class_log)
    st.pyplot(fig)

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-card"><div class="metric-label">SNR Improvement</div><div class="metric-value metric-blue">+{snr:.1f}dB</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><div class="metric-label">Avg Classifier Confidence</div><div class="metric-value metric-orange">{avg_conf*100:.0f}%</div></div>', unsafe_allow_html=True)
    method_label = "BiLSTM CRM (Deep Learning)" if result.get("used_bilstm", False) else "Spectral Subtraction (Stable)"
    m3.markdown(f'<div class="metric-card"><div class="metric-label">Method</div><div class="metric-value" style="font-size:13px; color:#00ff88;">{method_label}</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><div class="metric-label">Impulsive Spikes Caught</div><div class="metric-value" style="color:#f44336;">{spike_count}</div></div>', unsafe_allow_html=True)

    st.caption("Layer 1 (Random Forest) classifies each 100ms frame - Stage A (Spectral Subtraction) removes bulk noise - optional Stage B (BiLSTM Complex-Ratio-Mask) refines in the phase-preserving complex domain - confidence-gated gain smoothing applied on top.")
    st.markdown("<br>**Cleaned Output**", unsafe_allow_html=True)
    st.audio(buf, format="audio/wav")
    st.download_button("Download Cleaned Audio", buf, file_name="dhvani_cleaned.wav")
    return cleaned_int16, result


main_col, status_col = st.columns([2.2, 1])

with main_col:
    st.markdown('<div class="section-title">TEST THE <span>AI ENGINE</span></div>', unsafe_allow_html=True)
    use_lms = st.checkbox("Enable optional LMS residual cleanup stage", value=False)
    use_bilstm = st.checkbox("Enable Experimental Deep-Learning Mode (BiLSTM Complex-Mask)", value=False)
    tab1, tab2, tab3 = st.tabs(["LIVE MIC DEMO", "UPLOAD AUDIO FILE", "FIELD COMMS RELAY (2-DEVICE)"])

    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        audio_data = mic_recorder(start_prompt="START RECORDING", stop_prompt="STOP RECORDING", key="recorder")
        if audio_data is not None:
            raw_bytes = audio_data['bytes']
            st.audio(raw_bytes, format="audio/wav")
            if st.button("RUN DHVANI AI ENGINE", key="mic_btn"):
                with st.spinner("Processing through DHVANI engine..."):
                    sr, audio = load_mic_audio(raw_bytes)
                    show_results(sr, audio, use_lms, use_bilstm)

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload noisy audio (.wav)", type=["wav"])
        if uploaded_file is not None:
            sr, audio = wavfile.read(uploaded_file)
            st.audio(uploaded_file, format="audio/wav")
            if st.button("RUN DHVANI AI ENGINE", key="upload_btn"):
                with st.spinner("Processing through DHVANI engine..."):
                    show_results(sr, audio, use_lms, use_bilstm)

    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Simulates two field radios: Device A transmits the AI-cleaned signal, "
                   "Device B receives it over the shared channel -- exactly as it would happen "
                   "between two soldiers' radios. Open this tab in two browser windows to demo live.")
        colA, colB = st.columns(2)

        with colA:
            st.markdown('<div class="device-box"><div class="device-title">DEVICE A — Transmit</div>', unsafe_allow_html=True)
            device_name_a = st.text_input("Callsign", value="Alpha-1", key="device_a_name")
            tx_source = st.radio("Audio source", ["Upload file", "Record mic"], key="tx_source", horizontal=True)

            tx_audio = None
            tx_sr = None
            if tx_source == "Upload file":
                tx_file = st.file_uploader("Noisy audio to transmit (.wav)", type=["wav"], key="tx_upload")
                if tx_file is not None:
                    tx_sr, tx_audio = wavfile.read(tx_file)
                    st.audio(tx_file, format="audio/wav")
            else:
                tx_rec = mic_recorder(start_prompt="RECORD", stop_prompt="STOP", key="tx_mic")
                if tx_rec is not None:
                    tx_sr, tx_audio = load_mic_audio(tx_rec['bytes'])
                    st.audio(tx_rec['bytes'], format="audio/wav")

            if tx_audio is not None and st.button("CLEAN & TRANSMIT", key="tx_btn"):
                with st.spinner("Cleaning audio and transmitting over channel..."):
                    if tx_audio.ndim > 1:
                        tx_audio = tx_audio[:, 0]
                    result = pipeline.process(tx_sr, tx_audio, use_lms=use_lms, use_bilstm=use_bilstm)
                    cleaned_int16 = (np.clip(result["cleaned"], -1, 1) * 32767).astype(np.int16)
                    buf = io.BytesIO()
                    wavfile.write(buf, tx_sr, cleaned_int16)
                    buf.seek(0)
                    with mailbox["lock"]:
                        mailbox["audio_bytes"] = buf.getvalue()
                        mailbox["sr"] = tx_sr
                        mailbox["timestamp"] = time.strftime("%H:%M:%S")
                        mailbox["sender"] = device_name_a
                        mailbox["seq"] += 1
                    st.success(f"Transmitted cleaned audio as '{device_name_a}' -- Device B will receive it.")
            st.markdown('</div>', unsafe_allow_html=True)

        with colB:
            st.markdown('<div class="device-box"><div class="device-title">DEVICE B — Receive</div>', unsafe_allow_html=True)
            auto_poll = st.checkbox("Auto-check for incoming transmissions", value=True, key="rx_autopoll")
            if auto_poll:
                st_autorefresh(interval=2000, key="rx_refresh")

            with mailbox["lock"]:
                has_msg = mailbox["audio_bytes"] is not None
                seq = mailbox["seq"]
                sender = mailbox["sender"]
                ts = mailbox["timestamp"]
                audio_bytes = mailbox["audio_bytes"]
                sr = mailbox["sr"]

            if has_msg:
                st.markdown(f"**Incoming transmission #{seq}** from **{sender}** at {ts}")
                st.audio(audio_bytes, format="audio/wav")
                st.download_button("Download received audio", audio_bytes, file_name="received_clean.wav", key="rx_download")
            else:
                st.info("No transmission received yet. Waiting for Device A...")
            st.markdown('</div>', unsafe_allow_html=True)

with status_col:
    bilstm_status = '<span class="status-online">LOADED (CRM)</span>' if bilstm_ready else '<span class="status-warn">NOT FOUND</span>'
    st.markdown(f"""
    <div class="status-panel">
    <div class="status-row"><span>System Status</span><span class="status-online">ONLINE</span></div>
    <div class="status-row"><span>Layer 1 (Classifier)</span><span class="status-online">RANDOM FOREST</span></div>
    <div class="status-row"><span>Layer 2 (Enhancer)</span>{bilstm_status}</div>
    <div class="status-row"><span>Domain</span><span class="status-online">COMPLEX (Phase-Preserving)</span></div>
    <div class="status-row"><span>Fast-Path Spike Detector</span><span class="status-online">5ms ACTIVE</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><hr style='border-color:#1f2937;'>", unsafe_allow_html=True)
st.markdown('<div class="footer-text">SIH 2026 | Problem Statement SIH26052 | DRDO | Team DHVANI</div>', unsafe_allow_html=True)
