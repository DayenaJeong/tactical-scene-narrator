# streamlit_app.py

import streamlit as st
import cv2
import time
import queue
import threading
from pathlib import Path
import random
import numpy as np
import pandas as pd
from ultralytics import YOLO
import speech_recognition as sr
from datetime import datetime

try:
    from tactical_scene_narrator import (
        detect_objects,
        summarize_counts,
        query_llava,
        extract_salute,
        reverse_geocode,
        caption_worker,
    )
except ImportError:
    st.error("`tactical_scene_narrator.py` 파일을 찾을 수 없습니다. 두 파일이 같은 폴더에 있는지 확인하세요.")
    st.stop()

# --- 0. 다국어 및 UI 설정 ---
LANGUAGES = {
    "kr": {
        "title": "🎖️ 전장 시각관제 시스템(SALUTE-VIEW)",
        "login_title": "Login",
        "username": "사용자명",
        "role": "직책",
        "login_btn": "로그인",
        "logout_btn": "로그아웃",
        "roles": {"Operator": "작전병", "Analyst": "분석관", "Commander": "지휘관"},
        "sidebar_title": "임무 파라미터",
        "lang_select": "언어 선택",
        "upload_video": "임무 영상 업로드",
        "demo_mode": "데모 모드",
        "active_video": "현재 분석 중인 영상",
        "gps_coords": "GPS 좌표",
        "analysis_settings": "분석 설정",
        "caption_interval": "장면 분석 주기 (프레임)",
        "live_feed": "실시간 피드",
        "salute_report": "SALUTE 보고",
        "report_id": "보고 ID",
        "severity": "중요도",
        "size": "규모 (S)",
        "activity": "활동 (A)",
        "location": "위치 (L)",
        "unit": "부대 (U)",
        "time": "시간 (T)",
        "equipment": "장비 (E)",
        "voice_input": "음성으로 활동 입력",
        "map": "작전 지도",
        "status": "시스템 상태",
        "gps_status": "GPS 상태",
        "llm_status": "LLM 상태",
        "event_log": "이벤트 로그",
        "unclassified": "비분류",
        "login_success": "로그인 성공",
        "report_generated": "보고서 생성",
        "voice_listening": "음성 인식 중...",
        "voice_error": "음성 인식 오류",
    },
    "en": {
        "title": "🎖️ TACTICAL SCENE NARRATOR [v3.0]",
        "login_title": "ACCESS AUTHORIZATION",
        "username": "Username",
        "role": "Role",
        "login_btn": "Login",
        "logout_btn": "Logout",
        "roles": {"Operator": "Operator", "Analyst": "Analyst", "Commander": "Commander"},
        "sidebar_title": "MISSION PARAMETERS",
        "lang_select": "Select Language",
        "upload_video": "Upload Mission Video",
        "demo_mode": "DEMO MODE",
        "active_video": "Currently Analyzing Video",
        "gps_coords": "GPS COORDINATES",
        "analysis_settings": "ANALYSIS SETTINGS",
        "caption_interval": "Caption Interval (Frames)",
        "live_feed": "LIVE FEED",
        "salute_report": "SALUTE REPORT",
        "report_id": "REPORT ID",
        "severity": "SEVERITY",
        "size": "S - SIZE",
        "activity": "A - ACTIVITY",
        "location": "L - LOCATION",
        "unit": "U - UNIT",
        "time": "T - TIME",
        "equipment": "E - EQUIPMENT",
        "voice_input": "Input Activity via Voice",
        "map": "LOCATION MAP",
        "status": "SYSTEM STATUS",
        "gps_status": "GPS STATUS",
        "llm_status": "LLM STATUS",
        "event_log": "EVENT LOG",
        "unclassified": "UNCLASSIFIED",
        "login_success": "Login successful",
        "report_generated": "Report generated",
        "voice_listening": "Listening...",
        "voice_error": "Voice recognition error",
    }
}

# --- 1. 페이지 설정 및 커스텀 CSS ---
st.set_page_config(page_title="Tactical Scene Narrator", layout="wide")
st.markdown("""
<style>
    /* 전체 앱 배경 및 폰트 */
    .stApp {
        background-color: #010D1A;
        color: #00FF41; /* 밝은 녹색 */
    }
    h1, h2, h3 {
        color: #00FF41;
        text-shadow: 0 0 5px #00FF41;
        font-family: 'Courier New', Courier, monospace;
    }
    .st-emotion-cache-16txtl3 { /* 사이드바 배경 */
        background-color: rgba(1, 13, 26, 0.8);
    }
    .salute-item {
        background-color: #021a30;
        border-left: 5px solid #00FF41;
        padding: 10px 15px;
        margin-bottom: 8px;
        border-radius: 3px;
    }
    .salute-label {
        color: #7DF9FF; /* 청록색 */
        font-size: 0.9em;
        font-weight: bold;
        font-family: 'Courier New', Courier, monospace;
    }
    .salute-value {
        color: #FFFFFF;
        font-size: 1.1em;
        font-family: 'Consolas', 'Menlo', 'Monaco', monospace;
        word-wrap: break-word;
        margin-top: 5px;
    }
    .st-emotion-cache-16txtl3, .st-emotion-cache-1y4p8pa, .st-emotion-cache-1d3w5bk { color: #00FF41; }
    .st-emotion-cache-1v0mbdj, .st-emotion-cache-1r6slb0 { color: #00FF41; }
    .stSlider > div > div > div > div { color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)


def display_salute_field(label, value, icon=""):
    st.markdown(f"""
    <div class="salute-item">
        <span class="salute-label">{icon} {label}</span>
        <div class="salute-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# --- 2. 세션 상태 및 로그인 관리 (RBAC) ---
if 'lang' not in st.session_state:
    st.session_state.lang = "kr"
T = LANGUAGES[st.session_state.lang]

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = "Operator"
    st.session_state.event_log = []

def add_log(event):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.event_log.insert(0, f"[{timestamp}] {st.session_state.username} ({T['roles'][st.session_state.role]}): {event}")

# 로그인 UI
if not st.session_state.logged_in:
    st.title(T['login_title'])
    with st.form("login_form"):
        username = st.text_input(T['username'], "user01")
        role = st.selectbox(T['role'], ["Operator", "Analyst", "Commander"], index=2, format_func=lambda r: T['roles'][r])
        submitted = st.form_submit_button(T['login_btn'])
        if submitted:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = role
            add_log(T['login_success'])
            st.rerun()
    st.stop()

# --- 3. 앱 타이틀 및 사이드바 설정 ---
st.title(T['title'])
st.markdown(f"<div style='background-color: green; color: white; text-align: center; padding: 5px;'>-- {T['unclassified']} --</div>", unsafe_allow_html=True)

source = None
with st.sidebar:
    st.header(T['sidebar_title'])
    
    # 언어 선택
    selected_lang_label = st.radio(T['lang_select'], ["한국어", "English"], index=0 if st.session_state.lang == "kr" else 1)
    new_lang = "kr" if selected_lang_label == "한국어" else "en"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

    st.button(T['logout_btn'], on_click=lambda: st.session_state.update(logged_in=False, initialized=False))
    st.divider()

    # 영상 소스 선택 (업로드 우선)
    st.header(T['upload_video'])
    uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"], label_visibility="collapsed")
    
    temp_dir = Path("temp_videos")
    temp_dir.mkdir(exist_ok=True)

    if uploaded_file:
        temp_file_path = temp_dir / uploaded_file.name
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        source = str(temp_file_path)
        st.info(f"{T['active_video']}: {uploaded_file.name}")
    else:
        # 데모 비디오 폴더 확인 및 선택
        demo_dir = Path("UAV123_10fps")
        if demo_dir.is_dir() and any(demo_dir.iterdir()):
             # 데모 비디오를 일관되게 선택하기 위해 session_state에 저장
            if 'demo_video' not in st.session_state:
                st.session_state.demo_video = str(random.choice(list(demo_dir.glob("*.mp4"))))
            source = st.session_state.demo_video
            st.info(f"🎬 {T['demo_mode']}: {Path(source).name}")
        else:
            st.warning("데모 비디오 폴더('UAV123_10fps/data_seq')를 찾을 수 없거나 비어있습니다.")

    st.divider()
    
    # 분석 파라미터 설정
    st.header(T['gps_coords'])
    lat = st.number_input("🛰️ Latitude", value=37.4485, format="%.6f")
    lon = st.number_input("🛰️ Longitude", value=126.9526, format="%.6f")
    
    st.header(T['analysis_settings'])
    sample_rate = st.slider(f"👁️ {T['caption_interval']}", 1, 120, 30)

if not source:
    st.error("분석할 영상 소스를 선택해주세요. (파일 업로드 또는 데모 폴더 확인)")
    st.stop()
    
# --- 4. 모델 로드 및 초기 설정 ---
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8s.pt")

yolo = load_yolo_model()
loc_name = reverse_geocode(lat, lon)

# 세션 상태 초기화 (소스가 바뀌면 초기화되도록 `source`를 키로 사용)
if 'initialized' not in st.session_state or st.session_state.get('source_path') != source:
    st.session_state.initialized = True
    st.session_state.source_path = source
    st.session_state.cap = cv2.VideoCapture(source)
    st.session_state.frame_id = 0
    st.session_state.last_caption = "… 분석 대기 중 …"
    st.session_state.caption_status = "🟢 IDLE"
    st.session_state.activity_override = ""
    st.session_state.report = {} # 리포트 정보 저장
    
    # 백그라운드 스레드가 이미 있으면 종료 신호 보내기
    if 'stop_event' in st.session_state and st.session_state.stop_event:
        st.session_state.stop_event.set()

    # 새 스레드 시작
    st.session_state.stop_event = threading.Event()
    st.session_state.q_in = queue.Queue(maxsize=8)
    st.session_state.q_out = queue.Queue()
    
    worker = threading.Thread(
        target=caption_worker,
        args=(st.session_state.q_in, st.session_state.q_out, st.session_state.stop_event),
        daemon=True
    )
    worker.start()

# --- 5. 메인 UI 레이아웃 및 플레이스홀더 ---
col_vid, col_info = st.columns([2.5, 1.5])

with col_vid:
    st.subheader(T['live_feed'])
    video_slot = st.empty()
    map_slot   = st.empty() 

with col_info:
    st.subheader(T['salute_report'])
    salute_slots = {
        "ID": st.empty(), "SEV": st.empty(), "S": st.empty(), "A": st.empty(), 
        "L": st.empty(), "U": st.empty(), "T": st.empty(), "E": st.empty()
    }

    if st.button(f"🎤 {T['voice_input']}"):
        r = sr.Recognizer()
        with sr.Microphone() as mic_source:
            st.toast(T['voice_listening'])
            try:
                audio = r.listen(mic_source, timeout=5, phrase_time_limit=5)
                lang_code = "ko-KR" if st.session_state.lang == "kr" else "en-US"
                text = r.recognize_google(audio, language=lang_code)
                st.session_state.activity_override = text
                add_log(f"음성 입력 성공: '{text}'")
            except Exception as e:
                st.toast(f"{T['voice_error']}: {e}")
                add_log(f"음성 입력 오류: {e}")
    st.divider()
    
    st.subheader(T['map'])
    map_slot = st.empty()
    st.divider()
    
    st.subheader(T['status'])
    status_slots = {"gps": st.empty(), "caption": st.empty()}

# --- 6. 이벤트 로그 (Audit Trail) ---
if st.session_state.role == "Commander":
    with st.expander(f"📜 {T['event_log']}", expanded=False):
        st.text("\n".join(st.session_state.event_log))

# --- 7. 프레임 단위 처리 ---
if not st.session_state.cap.isOpened():
    st.warning("영상 스트림이 종료되었거나 열리지 않습니다.")
else:
    ret, frame = st.session_state.cap.read()
    if not ret:
        st.success("임무 영상 분석이 완료되었습니다.")
        st.session_state.cap.release()
    else:
        # 1) 객체 탐지
        dets = detect_objects(yolo, frame)
        n_person, n_vehicle, _ = summarize_counts(dets)

        # 2) 캡션 큐에 프레임 추가
        if st.session_state.frame_id % sample_rate == 0 and not st.session_state.q_in.full():
            st.session_state.q_in.put(frame.copy())
            st.session_state.caption_status = "🟡 ANALYZING..."

        # 3) 캡션 결과 가져오기
        try:
            new_caption = st.session_state.q_out.get_nowait()
            if new_caption:
                st.session_state.last_caption = new_caption
                st.session_state.caption_status = "🟢 IDLE"
                
                # 새 캡션으로 SALUTE 리포트 생성 및 저장
                has_weapon = any(w in st.session_state.last_caption.lower() for w in ("rifle", "gun", "weapon"))
                report = extract_salute(st.session_state.last_caption, n_person, n_vehicle, has_weapon, loc_name)
                st.session_state.report = report # 세션에 리포트 저장
                add_log(f"{T['report_generated']} (ID: {report.report_id})")

        except queue.Empty:
            pass

        # 4) 저장된 리포트 불러오기 및 음성 입력 적용
        report = st.session_state.get('report')
        if report:
             # 음성 입력이 있으면 Activity 항목 덮어쓰기
            if st.session_state.activity_override:
                report.activity = st.session_state.activity_override
                add_log(f"Activity 수동 변경 -> '{report.activity}'")
                st.session_state.activity_override = "" # 한번 사용 후 초기화

            # 5) 화면 업데이트
            vis = frame.copy()
            for x1, y1, x2, y2, _, cls in dets:
                col = (0, 255, 0) if int(cls) == 0 else (0, 255, 255)
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            video_slot.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # RBAC에 따른 정보 표시
            if st.session_state.role in ["Analyst", "Commander"]:
                with salute_slots["ID"]: display_salute_field(T['report_id'], report.report_id, "🆔")
                with salute_slots["SEV"]: display_salute_field(T['severity'], report.severity, "⚠️")
            
            with salute_slots["S"]: display_salute_field(T['size'], report.size, "🧑‍🤝‍🧑")
            with salute_slots["A"]: display_salute_field(T['activity'], report.activity, "🏃")
            with salute_slots["L"]: display_salute_field(T['location'], report.location, "📍")
            with salute_slots["U"]: display_salute_field(T['unit'], report.unit, "❓")
            with salute_slots["T"]: display_salute_field(T['time'], report.time, "⏱️")
            with salute_slots["E"]: display_salute_field(T['equipment'], report.equipment, "🔧")
            
            map_df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
            map_slot.map(map_df, zoom=14, use_container_width=True)

        # 상태 정보는 항상 표시
        with status_slots["gps"]: display_salute_field(T['gps_status'], "LOCK ACQUIRED", "🛰️")
        with status_slots["caption"]: display_salute_field(T['llm_status'], st.session_state.caption_status, "🧠")

        st.session_state.frame_id += 1
        
        # 다음 프레임 처리를 위해 스크립트 재실행 예약
        time.sleep(0.01) # CPU 사용률을 낮추기 위한 약간의 지연
        st.rerun()