"""
Voice-Powered Healthcare Policy Assistant
A Streamlit application for querying insurance policy documents using voice or text.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY_SET = bool(os.getenv("OPENAI_API_KEY"))

from services.rag import reset_rag_service
reset_rag_service()

st.set_page_config(
    page_title="Voice Powered Healthcare Policy Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background-color: #1a1a2e;
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    .main-header {
        text-align: center;
        padding: 1rem 1.5rem;
        background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white !important;
        font-size: 1.75rem;
        font-weight: 600;
        margin: 0;
    }
    .main-header p {
        color: #e0d4fc !important;
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a78bfa;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3d3d5c;
    }
    .answer-section {
        background: #ede9fe;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #7c3aed;
        margin-top: 1rem;
        color: #000000 !important;
        line-height: 1.6;
    }
    .answer-section * {
        color: #000000 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #7c3aed;
    }
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown label,
    [data-testid="stSidebar"] .stMarkdown li {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: white !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #7c3aed;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 500;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #6d28d9;
    }
    .stMarkdown, .stText, p, span, label, h1, h2, h3, h4 {
        color: #e0e0e0 !important;
    }
    .stTextInput input {
        background-color: #2d2d4a;
        color: #ffffff;
        border: 1px solid #4a4a6a;
    }
    .stTextInput input::placeholder {
        color: #8888aa;
    }
    .disclaimer {
        background-color: #ede9fe;
        border: 1px solid #7c3aed;
        border-radius: 6px;
        padding: 12px 16px;
        color: #5b21b6;
        font-size: 13px;
        margin-top: 1.5rem;
        text-align: center;
    }
    .streamlit-expanderHeader {
        background-color: #2d2d4a;
        border-radius: 8px;
    }
    [data-testid="stAlert"] {
        background-color: #2d2d4a;
        border: 1px solid #4a4a6a;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    defaults = {
        "status": "Ready",
        "stt_latency": None,
        "rag_latency": None,
        "tts_latency": None,
        "transcript_original": None,
        "transcript": None,
        "redactions": [],
        "was_redacted": False,
        "selected_plan": None,
        "retrieved_chunks": [],
        "response": None,
        "audio_response": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

AVAILABLE_PLANS = [
    ("All Plans", None),
    ("Independence", "SBC_8_Independence.pdf"),
    ("UCSF", "SBC_7_UCSF.pdf"),
    ("BlueShield Illinois", "SBC_9_BlueShield_Illinois.pdf"),
    ("AmeriHealth", "SBC_3_AmeriHealth.pdf"),
    ("BlueShield Oklahoma", "SBC_10_BlueShield_Oklahoma.pdf"),
    ("Blue Cross", "SBC_5_Blue Cross.pdf"),
    ("Sierra Pacific", "SBC_6_Sierra_Pacific.pdf"),
    ("University of Rochester", "SBC_4_Univ_Rochester.pdf"),
    ("Healthcare.gov Sample", "SBC_1_healthcare.gov.pdf"),
    ("National Health Council", "SBC_2_nationalhealthcouncil.pdf"),
]

init_session_state()

with st.sidebar:
    st.markdown("### Select Plan")
    st.markdown("Choose an insurance plan to search")
    
    plan_options = [p[0] for p in AVAILABLE_PLANS]
    selected_index = st.selectbox(
        "Insurance Plan",
        range(len(plan_options)),
        format_func=lambda x: plan_options[x],
        key="plan_selector",
        label_visibility="collapsed"
    )
    selected_plan_name, selected_plan_file = AVAILABLE_PLANS[selected_index]
    st.session_state.selected_plan = selected_plan_file
    
    st.markdown("---")
    
    if selected_plan_file:
        st.success(f"Searching: **{selected_plan_name}**")
    else:
        st.info("Searching all available plans")
    
    st.markdown("---")
    st.markdown("### Available Plans")
    for name, _ in AVAILABLE_PLANS[1:]:
        st.markdown(f"- {name}")

st.markdown("""
<div class="main-header">
    <h1>Voice Powered Healthcare Policy Assistant</h1>
    <p>Ask questions about your insurance policy using voice or text</p>
</div>
""", unsafe_allow_html=True)

if not OPENAI_API_KEY_SET:
    st.error("OpenAI API key not configured. Add OPENAI_API_KEY to your .env file.")
    st.stop()

col_voice, col_text = st.columns(2)

with col_voice:
    st.markdown('<div class="section-title">Voice Input</div>', unsafe_allow_html=True)
    audio_data = st.audio_input(
        "Record your question",
        key="audio_recorder",
        help="Click to record, click again to stop"
    )
    if audio_data is not None:
        st.audio(audio_data, format="audio/wav")

with col_text:
    st.markdown('<div class="section-title">Text Input</div>', unsafe_allow_html=True)
    text_input = st.text_input(
        "Type your question",
        placeholder="Type your question",
        label_visibility="collapsed"
    )

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    submit_clicked = st.button("Ask Question", use_container_width=True, type="primary")

if submit_clicked:
    query_text = None
    
    if audio_data is not None:
        try:
            from services.stt import transcribe_audio
            from services.sanitizer import sanitize_transcript
            
            audio_bytes = audio_data.getvalue()
            
            with st.spinner("Transcribing..."):
                stt_result = transcribe_audio(audio_bytes, "recording.wav")
            
            if stt_result.success:
                st.session_state.transcript_original = stt_result.text
                st.session_state.stt_latency = stt_result.latency_ms
                sanitized = sanitize_transcript(stt_result.text)
                query_text = sanitized.sanitized_text
                st.session_state.transcript = query_text
                st.session_state.was_redacted = sanitized.was_redacted
            else:
                st.error(f"Transcription failed: {stt_result.error}")
        except Exception as e:
            st.error(f"STT Error: {str(e)}")
    
    elif text_input.strip():
        from services.sanitizer import sanitize_transcript
        sanitized = sanitize_transcript(text_input.strip())
        query_text = sanitized.sanitized_text
        st.session_state.transcript = query_text
        st.session_state.transcript_original = text_input.strip()
        st.session_state.was_redacted = sanitized.was_redacted
    
    if query_text:
        try:
            from services.rag import query_rag
            from services.tts import text_to_speech
            
            plan_filter = st.session_state.selected_plan
            
            with st.spinner("Searching policy documents..."):
                rag_result = query_rag(query_text, plan_filter=plan_filter)
            
            st.session_state.rag_latency = rag_result.latency_ms
            st.session_state.response = rag_result.answer
            st.session_state.retrieved_chunks = [
                chunk.to_dict() for chunk in rag_result.chunks
            ]
            
            if rag_result.success and rag_result.answer:
                with st.spinner("Generating voice response..."):
                    tts_result = text_to_speech(rag_result.answer)
                
                if tts_result.success:
                    st.session_state.audio_response = tts_result.audio_bytes
                    st.session_state.tts_latency = tts_result.latency_ms
            
            st.session_state.status = "Complete"
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
        
        st.rerun()
    else:
        st.warning("Please record audio or type a question first.")

if st.session_state.response:
    st.markdown("---")
    
    st.markdown('<div class="section-title">Answer</div>', unsafe_allow_html=True)
    
    if st.session_state.transcript:
        st.markdown(f"**Question:** {st.session_state.transcript}")
    
    st.markdown(f"""
    <div class="answer-section">
        {st.session_state.response}
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.audio_response:
        st.audio(st.session_state.audio_response, format="audio/mp3", autoplay=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Ask Another Question", use_container_width=True):
            st.session_state.transcript = ""
            st.session_state.transcript_original = ""
            st.session_state.response = ""
            st.session_state.audio_response = None
            st.session_state.retrieved_chunks = []
            st.rerun()
    
    if st.session_state.retrieved_chunks:
        st.markdown("---")
        st.markdown('<div class="section-title">Source Documents</div>', unsafe_allow_html=True)
        
        for i, chunk in enumerate(st.session_state.retrieved_chunks[:3]):
            score = chunk.get('score', 0)
            score_pct = int(score * 100)
            source = chunk.get('source', 'Unknown')
            page = chunk.get('page', '?')
            
            with st.expander(f"{'[Top Match] ' if i == 0 else ''}{source} - Page {page} ({score_pct}% match)"):
                st.markdown(chunk.get("content", ""))

st.markdown("""
<div class="disclaimer">
    <strong>Administrative assistance only.</strong> 
    This tool provides policy information and does not offer medical advice. 
    For medical questions, direct callers to their healthcare provider.
</div>
""", unsafe_allow_html=True)
