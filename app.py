import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from pathlib import Path
from PIL import Image

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wild Animal Detector",
    page_icon="🐘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-deep:      #0a0d0f;
    --bg-card:      #111518;
    --bg-raised:    #181d21;
    --accent:       #00e5a0;
    --accent-dim:   #00e5a033;
    --accent-glow:  #00e5a066;
    --danger:       #ff4757;
    --danger-dim:   #ff475733;
    --warn:         #ffa502;
    --warn-dim:     #ffa50233;
    --text-primary: #e8edf2;
    --text-muted:   #6b7a87;
    --border:       #1e2730;
    --border-glow:  #00e5a044;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-deep);
    color: var(--text-primary);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem; max-width: 1400px; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1a14 0%, #0a1510 40%, #0a0d0f 100%);
    border: 1px solid var(--border-glow);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, var(--accent-glow) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem;
    letter-spacing: 0.08em;
    color: var(--accent);
    line-height: 1;
    margin: 0 0 0.3rem;
    text-shadow: 0 0 40px var(--accent-glow);
}
.hero-sub {
    color: var(--text-muted);
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.hero-badge {
    display: inline-block;
    background: var(--accent-dim);
    color: var(--accent);
    border: 1px solid var(--accent-glow);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }
.sidebar-logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    color: var(--accent);
    letter-spacing: 0.1em;
    text-align: center;
    padding: 0.5rem 0 1.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
    text-shadow: 0 0 20px var(--accent-glow);
}
.sidebar-section {
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.2rem 0 0.6rem;
}

/* ── Cards ── */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: var(--accent-glow); }
.stat-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    color: var(--accent);
    line-height: 1;
    text-shadow: 0 0 20px var(--accent-glow);
}
.stat-label {
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* ── Alert boxes ── */
.alert-box {
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.9rem 1.2rem;
    border-radius: 10px;
    font-size: 0.9rem;
    font-weight: 500;
    margin: 0.5rem 0;
}
.alert-danger {
    background: var(--danger-dim);
    border: 1px solid var(--danger);
    color: var(--danger);
}
.alert-success {
    background: var(--accent-dim);
    border: 1px solid var(--accent);
    color: var(--accent);
}
.alert-warn {
    background: var(--warn-dim);
    border: 1px solid var(--warn);
    color: var(--warn);
}

/* ── Detection log ── */
.log-container {
    background: var(--bg-deep);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    height: 220px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
}
.log-entry {
    padding: 3px 0;
    border-bottom: 1px solid var(--border);
    color: var(--text-muted);
}
.log-entry.hit { color: var(--accent); }
.log-entry.alert { color: var(--danger); }

/* ── Animal pill tags ── */
.animal-tag {
    display: inline-block;
    background: var(--accent-dim);
    color: var(--accent);
    border: 1px solid var(--accent-glow);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 2px 3px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Streamlit widget overrides ── */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.4rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    background: #00ffb2 !important;
    box-shadow: 0 0 20px var(--accent-glow) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--bg-raised) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}
.stSlider [data-testid="stSlider"] > div > div > div { background: var(--accent) !important; }
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border-glow) !important;
    border-radius: 10px !important;
}
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
}
label, .stRadio label { color: var(--text-muted) !important; font-size: 0.85rem !important; }
.stProgress > div > div > div > div { background: var(--accent) !important; }
h2, h3 {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.06em;
    color: var(--text-primary) !important;
}
.divider { height: 1px; background: var(--border); margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ───────────────────────────────────────────────────────────────
for key, default in {
    "model": None,
    "model_path": "",
    "detection_log": [],
    "total_detections": 0,
    "frames_processed": 0,
    "alert_count": 0,
    "running": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
ANIMALS = ["buffalo", "elephant", "rhino", "zebra"]
ANIMAL_EMOJI = {"buffalo": "🐃", "elephant": "🐘", "rhino": "🦏", "zebra": "🦓"}

# ─── Audio helpers ───────────────────────────────────────────────────────────────
import base64

BEEP_JS = """
<script>
function playBeep() {
    var ctx = new (window.AudioContext || window.webkitAudioContext)();
    var osc = ctx.createOscillator();
    var gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = 'square';
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    gain.gain.setValueAtTime(0.3, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.4);
}
playBeep();
</script>
"""

def play_browser_beep():
    """Inject a one-shot Web Audio beep into the Streamlit page."""
    st.components.v1.html(BEEP_JS, height=0)

def play_custom_sound(audio_bytes: bytes, mime: str):
    """Inject an HTML5 audio autoplay element for a custom sound file."""
    b64 = base64.b64encode(audio_bytes).decode()
    html = f"""
    <audio autoplay style="display:none">
      <source src="data:{mime};base64,{b64}" type="{mime}">
    </audio>
    """
    st.components.v1.html(html, height=0)

BOX_COLOR  = (0, 229, 160)   # accent green
TEXT_COLOR = (0, 229, 160)

def load_model(path):
    try:
        from ultralytics import YOLO
        return YOLO(path), None
    except Exception as e:
        return None, str(e)

def draw_boxes(frame, results, model_names, conf_thresh):
    detected = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue
            cls_id = int(box.cls[0])
            label  = model_names[cls_id]
            if label.lower() in ANIMALS:
                detected.append((label, conf))
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
                tag = f"{label.upper()}  {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), BOX_COLOR, -1)
                cv2.putText(frame, tag, (x1 + 4, y1 - 4),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 1)
    return frame, detected

def add_hud(frame, fps, detections):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (10, 13, 15), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    status = "● ANIMAL DETECTED" if detections else "● MONITORING"
    color  = (0, 229, 160) if not detections else (255, 71, 87)
    cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)
    cv2.putText(frame, f"FPS {fps:.1f}", (w - 90, 24),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (107, 122, 135), 1)
    return frame

def fire_alert():
    """Trigger the appropriate audio alert based on sidebar setting."""
    if sound_mode == "Browser beep":
        play_browser_beep()
    elif sound_mode == "Custom file" and custom_audio_bytes:
        play_custom_sound(custom_audio_bytes, custom_audio_mime)
    # "Mute" → do nothing

def log_entry(msg, kind="info"):
    ts = time.strftime("%H:%M:%S")
    st.session_state.detection_log.insert(0, {"ts": ts, "msg": msg, "kind": kind})
    if len(st.session_state.detection_log) > 80:
        st.session_state.detection_log.pop()

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🐘 WILDGUARD</div>', unsafe_allow_html=True)

    # Auto-load best.pt on startup
    if st.session_state.model is None:
        with st.spinner("Loading best.pt…"):
            mdl, err = load_model("best.pt")
        if err:
            st.error(f"❌ Could not load best.pt: {err}")
        else:
            st.session_state.model = mdl
            st.session_state.model_path = "best.pt"
            log_entry("best.pt loaded", "hit")

    st.markdown('<div class="sidebar-section">Detection</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence threshold", 0.10, 0.95, 0.45, 0.05,
                                format="%.2f")

    st.markdown('<div class="sidebar-section">Target animals</div>', unsafe_allow_html=True)
    selected_animals = []
    cols = st.columns(2)
    for i, a in enumerate(ANIMALS):
        with cols[i % 2]:
            if st.checkbox(f"{ANIMAL_EMOJI.get(a,'')} {a.capitalize()}", value=True, key=f"chk_{a}"):
                selected_animals.append(a)

    st.markdown('<div class="sidebar-section">Alert Sound</div>', unsafe_allow_html=True)
    SOUND_FILE = "beeep.mp3"
    custom_audio_bytes = None
    custom_audio_mime  = "audio/mpeg"
    if os.path.exists(SOUND_FILE):
        with open(SOUND_FILE, "rb") as f:
            custom_audio_bytes = f.read()
        st.success(f"🔊 {SOUND_FILE} loaded")
    else:
        st.caption("⚠ beeep.mp3 not found — browser beep used as fallback")
    sound_mode = "Custom file" if custom_audio_bytes else "Browser beep"

    st.markdown('<div class="sidebar-section">Output</div>', unsafe_allow_html=True)
    save_output = st.checkbox("Save annotated video", value=False)
    show_conf   = st.checkbox("Show confidence scores", value=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    model_status = "🟢 Ready" if st.session_state.model else "🔴 No model"
    st.markdown(f"<div style='font-family:JetBrains Mono;font-size:0.75rem;color:var(--text-muted)'>"
                f"STATUS &nbsp; <span style='color:var(--text-primary)'>{model_status}</span></div>",
                unsafe_allow_html=True)

# ─── Hero ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-badge">YOLOv8 · Real-time Detection</div>
  <div class="hero-title">WildGuard AI</div>
  <div class="hero-sub">Wildlife Intrusion Detection &amp; Alert System</div>
</div>
""", unsafe_allow_html=True)

# ─── Stats row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{st.session_state.total_detections}</div>'
                f'<div class="stat-label">Detections</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{st.session_state.frames_processed}</div>'
                f'<div class="stat-label">Frames</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{st.session_state.alert_count}</div>'
                f'<div class="stat-label">Alerts</div></div>', unsafe_allow_html=True)
with c4:
    mdl_name = Path(st.session_state.model_path).name if st.session_state.model_path else "—"
    st.markdown(f'<div class="stat-card"><div class="stat-value" style="font-size:1.1rem;padding-top:0.5rem">'
                f'{mdl_name}</div><div class="stat-label">Active model</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Mode tabs ───────────────────────────────────────────────────────────────────
tab_img, tab_vid, tab_cam = st.tabs(["📷  Image", "🎬  Video", "📡  Webcam"])

# ══════════════════════════════════════════════════════════════════════════════════
# IMAGE TAB
# ══════════════════════════════════════════════════════════════════════════════════
with tab_img:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("### Upload Image")
        img_file = st.file_uploader("Drop an image", type=["jpg","jpeg","png","bmp","webp"],
                                     label_visibility="collapsed", key="img_up")
        if img_file:
            img_arr = np.frombuffer(img_file.read(), np.uint8)
            frame_in = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB),
                     caption="Original", use_container_width=True)

    with col_r:
        st.markdown("### Detection Result")
        if img_file and st.session_state.model:
            if st.button("▶ Run Detection", key="run_img"):
                with st.spinner("Running inference…"):
                    results = st.session_state.model(frame_in)
                    frame_out, dets = draw_boxes(
                        frame_in.copy(), results,
                        st.session_state.model.names, conf_threshold
                    )
                    frame_out = add_hud(frame_out, 0, dets)

                st.image(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB),
                         caption="Annotated", use_container_width=True)

                if dets:
                    st.session_state.total_detections += len(dets)
                    st.session_state.alert_count += 1
                    fire_alert()
                    names_str = ", ".join(f"{ANIMAL_EMOJI.get(d[0].lower(),'')} {d[0]}" for d in dets)
                    st.markdown(f'<div class="alert-box alert-danger">🚨 Animals detected: {names_str}</div>',
                                unsafe_allow_html=True)
                    for d in dets:
                        log_entry(f"Image · {d[0]} ({d[1]:.0%})", "alert")
                    # Download button
                    _, buf = cv2.imencode(".jpg", frame_out)
                    st.download_button("⬇ Download annotated image", buf.tobytes(),
                                       "detection_result.jpg", "image/jpeg")
                else:
                    st.markdown('<div class="alert-box alert-success">✅ No target animals detected</div>',
                                unsafe_allow_html=True)
                    log_entry("Image · no detections", "info")
                st.session_state.frames_processed += 1
        elif img_file and not st.session_state.model:
            st.markdown('<div class="alert-box alert-warn">⚠ Load a model first (sidebar)</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:var(--text-muted);font-size:0.9rem;padding-top:3rem;text-align:center">'
                        'Upload an image to begin</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════════
# VIDEO TAB
# ══════════════════════════════════════════════════════════════════════════════════
with tab_vid:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("### Upload Video")
        vid_file = st.file_uploader("Drop a video", type=["mp4","avi","mov","mkv"],
                                     label_visibility="collapsed", key="vid_up")
        skip_n = st.slider("Process every N frames", 1, 10, 2,
                            help="Higher = faster but may miss detections")
        max_frames = st.number_input("Max frames to process (0 = all)", 0, 5000, 300, step=50)

    with col_r:
        st.markdown("### Live Preview")
        if vid_file and st.session_state.model:
            run_btn = st.button("▶ Analyse Video", key="run_vid")
            stop_placeholder = st.empty()
            frame_display = st.empty()
            prog_bar = st.progress(0)
            info_placeholder = st.empty()

            if run_btn:
                tmp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=Path(vid_file.name).suffix)
                tmp_vid.write(vid_file.read()); tmp_vid.flush()

                cap = cv2.VideoCapture(tmp_vid.name)
                total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_vid = cap.get(cv2.CAP_PROP_FPS) or 25

                out_path = None
                out_writer = None
                if save_output:
                    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    out_writer = cv2.VideoWriter(out_path,
                                                  cv2.VideoWriter_fourcc(*"mp4v"),
                                                  fps_vid, (fw, fh))

                frame_idx = 0
                processed = 0
                alert_played_flag = False
                t0 = time.time()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_idx += 1
                    limit = max_frames if max_frames > 0 else total_vid_frames
                    if frame_idx > limit:
                        break

                    if frame_idx % skip_n != 0:
                        continue

                    results = st.session_state.model(frame)
                    frame_out, dets = draw_boxes(frame.copy(), results,
                                                  st.session_state.model.names, conf_threshold)
                    cur_fps = processed / (time.time() - t0 + 1e-6)
                    frame_out = add_hud(frame_out, cur_fps, dets)

                    if out_writer:
                        out_writer.write(frame_out)

                    frame_display.image(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB),
                                        use_container_width=True)
                    prog_bar.progress(min(frame_idx / limit, 1.0))

                    if dets:
                        st.session_state.total_detections += len(dets)
                        if not alert_played_flag:
                            st.session_state.alert_count += 1
                            alert_played_flag = True
                            fire_alert()
                        for d in dets:
                            log_entry(f"Frame {frame_idx} · {d[0]} ({d[1]:.0%})", "alert")
                    else:
                        alert_played_flag = False

                    processed += 1
                    st.session_state.frames_processed += 1

                cap.release()
                if out_writer:
                    out_writer.release()

                info_placeholder.markdown(
                    f'<div class="alert-box alert-success">✅ Done · {processed} frames · '
                    f'{st.session_state.total_detections} detections</div>',
                    unsafe_allow_html=True)
                log_entry(f"Video complete · {processed} frames processed", "hit")

                if out_path and os.path.exists(out_path):
                    with open(out_path, "rb") as f:
                        st.download_button("⬇ Download annotated video", f.read(),
                                           "annotated_video.mp4", "video/mp4")
        elif vid_file and not st.session_state.model:
            st.markdown('<div class="alert-box alert-warn">⚠ Load a model first (sidebar)</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:var(--text-muted);font-size:0.9rem;padding-top:3rem;text-align:center">'
                        'Upload a video to begin</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════════
# WEBCAM TAB
# ══════════════════════════════════════════════════════════════════════════════════
with tab_cam:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("### Webcam Settings")
        cam_idx = st.selectbox("Camera index", [0, 1, 2, 3], index=0)
        if st.button("▶ Start Webcam", key="start_cam"):
            if not st.session_state.model:
                st.error("Load a model first.")
            else:
                st.session_state.running = True
                log_entry(f"Webcam {cam_idx} started", "hit")

        if st.button("⏹ Stop", key="stop_cam"):
            st.session_state.running = False
            log_entry("Webcam stopped", "info")

    with col_r:
        st.markdown("### Live Feed")
        cam_display = st.empty()
        cam_status  = st.empty()

        if st.session_state.running and st.session_state.model:
            cap = cv2.VideoCapture(cam_idx)
            if not cap.isOpened():
                cam_status.markdown('<div class="alert-box alert-danger">🚨 Cannot open webcam</div>',
                                    unsafe_allow_html=True)
                st.session_state.running = False
            else:
                alert_flag = False
                t0 = time.time()
                frame_count = 0
                for _ in range(300):   # ~10 s snapshot loop
                    if not st.session_state.running:
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = st.session_state.model(frame)
                    frame_out, dets = draw_boxes(frame.copy(), results,
                                                  st.session_state.model.names, conf_threshold)
                    cur_fps = frame_count / (time.time() - t0 + 1e-6)
                    frame_out = add_hud(frame_out, cur_fps, dets)
                    cam_display.image(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB),
                                      use_container_width=True)
                    if dets:
                        st.session_state.total_detections += len(dets)
                        if not alert_flag:
                            st.session_state.alert_count += 1
                            alert_flag = True
                            fire_alert()
                        cam_status.markdown(
                            f'<div class="alert-box alert-danger">🚨 ALERT — '
                            f'{", ".join(d[0] for d in dets)}</div>',
                            unsafe_allow_html=True)
                        for d in dets:
                            log_entry(f"Webcam · {d[0]} ({d[1]:.0%})", "alert")
                    else:
                        alert_flag = False
                        cam_status.markdown(
                            '<div class="alert-box alert-success">✅ Monitoring…</div>',
                            unsafe_allow_html=True)
                    frame_count += 1
                    st.session_state.frames_processed += 1
                cap.release()
        else:
            cam_display.markdown(
                '<div style="color:var(--text-muted);font-size:0.9rem;padding-top:3rem;text-align:center">'
                'Press Start Webcam to begin</div>', unsafe_allow_html=True)

# ─── Detection log ───────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
lcol, rcol = st.columns([2, 1], gap="large")

with lcol:
    st.markdown("### Detection Log")
    log_html = '<div class="log-container">'
    if not st.session_state.detection_log:
        log_html += '<div class="log-entry" style="color:var(--text-muted)">— No events yet —</div>'
    for entry in st.session_state.detection_log:
        cls = "log-entry alert" if entry["kind"] == "alert" else (
              "log-entry hit"   if entry["kind"] == "hit"   else "log-entry")
        log_html += f'<div class="{cls}">[{entry["ts"]}] {entry["msg"]}</div>'
    log_html += '</div>'
    st.markdown(log_html, unsafe_allow_html=True)

    if st.button("🗑 Clear log", key="clear_log"):
        st.session_state.detection_log = []
        st.rerun()

with rcol:
    st.markdown("### Target Species")
    tags = "".join(
        f'<span class="animal-tag">{ANIMAL_EMOJI.get(a,"")} {a}</span>'
        for a in selected_animals
    ) or '<span style="color:var(--text-muted);font-size:0.85rem">None selected</span>'
    st.markdown(f'<div style="padding:0.5rem 0">{tags}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Session Summary")
    summary_rows = [
        ("Frames processed", st.session_state.frames_processed),
        ("Total detections",  st.session_state.total_detections),
        ("Alert triggers",    st.session_state.alert_count),
        ("Conf threshold",    f"{conf_threshold:.0%}"),
    ]
    for label, val in summary_rows:
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
            f'border-bottom:1px solid var(--border);font-size:0.85rem">'
            f'<span style="color:var(--text-muted)">{label}</span>'
            f'<span style="font-family:JetBrains Mono;color:var(--accent)">{val}</span></div>',
            unsafe_allow_html=True)

    if st.button("↺ Reset session", key="reset"):
        st.session_state.total_detections = 0
        st.session_state.frames_processed = 0
        st.session_state.alert_count = 0
        st.session_state.detection_log = []
        log_entry("Session reset", "info")
        st.rerun()