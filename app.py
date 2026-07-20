
# -*- coding: utf-8 -*-
"""HTU Advisor — reviewed mobile-first Streamlit interface."""

from pathlib import Path
import io

import pandas as pd
import speech_recognition as sr
import streamlit as st

from quiz_data import (
    SCENARIOS,
    MAJOR_ADMISSION_FAMILY,
    MAJOR_NAMES_AR,
    check_eligibility,
    requires_physics,
    scenarios_to_feature_vector,
)
from rag import AdmissionRAG


# ---------------------------------------------------------------------------
# Page configuration and state
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HTU Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_STATE = {
    "lang": "en",
    "theme": "light",
    "page": "home",
    "quiz_step": -1,
    "quiz_answers": {},
    "chat_history": [],
    "chat_context": {},
    "home_question": "",
    "voice_recorder_nonce": 0,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def tr(en: str, ar: str) -> str:
    return ar if st.session_state.lang == "ar" else en


def navigate(page: str) -> None:
    st.session_state.page = page
    st.rerun()


# ---------------------------------------------------------------------------
# Accessible visual system
# ---------------------------------------------------------------------------
THEMES = {
    "light": {
        "background": "#F5F6F8",
        "surface": "#FFFFFF",
        "surface_alt": "#F1F3F5",
        "text": "#14161A",
        "muted": "#667085",
        "accent": "#B5122B",
        "accent_hover": "#94091E",
        "accent_soft": "rgba(181, 18, 43, 0.08)",
        "border": "#E4E7EC",
        "success": "#067647",
        "warning": "#8A4B00",
        "focus": "#2563EB",
    },
    "dark": {
        "background": "#0E1014",
        "surface": "#16191F",
        "surface_alt": "#1C2029",
        "text": "#ECEDEE",
        "muted": "#9BA1AC",
        "accent": "#F26178",
        "accent_hover": "#FF7C90",
        "accent_soft": "rgba(242, 97, 120, 0.14)",
        "border": "#262B33",
        "success": "#3DD68C",
        "warning": "#FFC46B",
        "focus": "#7AA2FF",
    },
}
C = THEMES[st.session_state.theme]

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');

    :root {{
        --bg: {C["background"]};
        --surface: {C["surface"]};
        --surface-alt: {C["surface_alt"]};
        --text: {C["text"]};
        --muted: {C["muted"]};
        --accent: {C["accent"]};
        --accent-hover: {C["accent_hover"]};
        --accent-soft: {C["accent_soft"]};
        --border: {C["border"]};
        --success: {C["success"]};
        --focus: {C["focus"]};
        --radius-sm: 10px;
        --radius-md: 14px;
        --radius-lg: 20px;
        --shadow-sm: 0 1px 2px rgba(16,24,40,.05);
        --shadow-md: 0 6px 20px rgba(16,24,40,.07);
    }}

    html, body, [class*="css"], .stApp,
    button, input, textarea, select {{
        font-family: 'Inter', 'IBM Plex Sans Arabic', -apple-system,
            BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    }}

    .stApp {{ background: var(--bg); color: var(--text); }}

    .block-container {{
        max-width: 1060px;
        padding-top: 1.25rem;
        padding-bottom: 4rem;
    }}

    h1, h2, h3, h4 {{ color: var(--text); letter-spacing: -0.01em; }}
    p, label, span, li {{ color: var(--text); }}
    a {{ color: var(--accent); }}

    /* ---- Top bar ---- */
    .app-header {{
        display: flex; align-items: center; gap: .7rem;
        min-width: 0; padding: .3rem 0;
    }}
    .brand {{ display: flex; align-items: center; gap: .7rem; min-width: 0; }}
    .brand-mark {{
        display: grid; place-items: center;
        width: 40px; height: 40px;
        border-radius: 11px;
        background: var(--accent); color: #fff;
        font-size: 1.25rem; font-weight: 800;
        box-shadow: var(--shadow-sm); flex: 0 0 auto;
    }}
    .brand-title {{ font-size: 1rem; font-weight: 700; line-height: 1.1; }}
    .brand-subtitle {{ color: var(--muted); font-size: .78rem; margin-top: .12rem; }}

    /* ---- Hero ---- */
    .hero {{
        position: relative;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: clamp(1.5rem, 4vw, 2.75rem);
        margin: .5rem 0 1.75rem;
        box-shadow: var(--shadow-sm);
        overflow: hidden;
    }}
    .hero::before {{
        content: "";
        position: absolute;
        inset-block-start: 0; inset-inline-start: 0;
        width: 4px; height: 100%;
        background: var(--accent);
    }}
    .eyebrow {{
        color: var(--accent); font-weight: 700;
        letter-spacing: .06em; text-transform: uppercase;
        font-size: .74rem; margin-bottom: .7rem;
    }}
    .hero h1 {{
        margin: 0;
        font-size: clamp(1.8rem, 4.2vw, 2.9rem);
        line-height: 1.1; font-weight: 800; max-width: 44ch;
    }}
    .hero p {{
        color: var(--muted);
        font-size: clamp(.98rem, 1.6vw, 1.1rem);
        line-height: 1.65; max-width: 58ch; margin: 1rem 0 0;
    }}

    /* ---- Section headings ---- */
    .section-heading {{ margin: 2.25rem 0 1rem; }}
    .section-heading h2 {{ margin: 0 0 .25rem; font-size: 1.3rem; font-weight: 700; }}
    .section-heading p {{ color: var(--muted); margin: 0; font-size: .95rem; }}

    /* ---- Cards ---- */
    .feature-card, .content-card, .result-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1.35rem; height: 100%;
        box-shadow: var(--shadow-sm);
    }}
    .feature-card {{ transition: border-color .15s ease, box-shadow .15s ease, transform .15s ease; }}
    .feature-card:hover {{
        border-color: var(--accent);
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }}
    .feature-card h3 {{ margin: .85rem 0 .35rem; font-size: 1.05rem; font-weight: 650; }}
    .feature-card p, .content-card p {{ color: var(--muted); line-height: 1.6; font-size: .93rem; margin: 0; }}
    .feature-icon {{
        display: grid; place-items: center;
        width: 42px; height: 42px; border-radius: 11px;
        background: var(--accent-soft); color: var(--accent);
    }}
    .feature-icon svg {{ width: 22px; height: 22px; }}

    .result-card .eyebrow {{ margin-bottom: .4rem; }}
    .result-card h2 {{ margin: 0 0 .5rem; font-size: 1.6rem; }}
    .result-card p {{ color: var(--muted); margin: 0; }}

    /* ---- Steps ---- */
    .step-row {{ display: flex; gap: .8rem; align-items: flex-start; margin-bottom: 1rem; }}
    .step-row:last-child {{ margin-bottom: 0; }}
    .step-number {{
        flex: 0 0 30px; width: 30px; height: 30px;
        display: grid; place-items: center; border-radius: 50%;
        background: var(--accent-soft); color: var(--accent);
        font-weight: 700; font-size: .9rem;
    }}
    .step-copy strong {{ display: block; margin-bottom: .15rem; font-weight: 650; }}
    .step-copy span {{ color: var(--muted); font-size: .92rem; }}

    /* ---- Notices ---- */
    .status-note {{
        border-inline-start: 3px solid var(--accent);
        background: var(--surface);
        border-radius: var(--radius-sm);
        padding: .85rem 1rem; color: var(--text);
        box-shadow: var(--shadow-sm); font-size: .92rem;
    }}

    /* ---- Trust strip ---- */
    .trust-strip {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: .8rem; margin: 1rem 0 1.5rem;
    }}
    .trust-item {{
        background: var(--surface); border: 1px solid var(--border);
        border-radius: var(--radius-md); padding: 1rem 1.1rem;
        box-shadow: var(--shadow-sm);
    }}
    .trust-item strong {{ display: block; font-size: .92rem; margin-bottom: .25rem; font-weight: 650; }}
    .trust-item span {{ color: var(--muted); font-size: .82rem; line-height: 1.5; }}

    /* ---- Voice ---- */
    .voice-panel {{
        background: var(--surface); border: 1px solid var(--border);
        border-radius: var(--radius-md); padding: 1.1rem;
        margin: .75rem 0 1rem; box-shadow: var(--shadow-sm);
    }}
    .voice-panel h4 {{ margin: 0 0 .3rem; font-weight: 650; }}
    .voice-panel p {{ color: var(--muted); margin: 0; font-size: .9rem; }}

    /* ---- Chat empty ---- */
    .chat-empty {{
        text-align: center; padding: 2.5rem 1rem;
        background: var(--surface); border: 1px dashed var(--border);
        border-radius: var(--radius-md); margin: 1rem 0;
    }}
    .chat-empty .chat-icon {{ color: var(--muted); margin-bottom: .6rem; }}
    .chat-empty .chat-icon svg {{ width: 34px; height: 34px; }}
    .chat-empty h3 {{ font-weight: 650; }}
    .chat-empty p {{ color: var(--muted); max-width: 52ch; margin: .3rem auto 0; }}

    /* ---- Answer meta ---- */
    .answer-meta {{ display: flex; flex-wrap: wrap; gap: .45rem; margin-top: .7rem; }}
    .meta-badge {{
        display: inline-flex; align-items: center; gap: .3rem;
        min-height: 26px; padding: .18rem .6rem; border-radius: 999px;
        background: var(--surface-alt); border: 1px solid var(--border);
        color: var(--muted); font-size: .76rem; font-weight: 600;
    }}

    .small-muted {{ color: var(--muted); font-size: .85rem; }}
    .footer {{
        margin-top: 3rem; padding-top: 1.25rem;
        border-top: 1px solid var(--border);
        color: var(--muted); font-size: .82rem;
    }}

    /* ---- Buttons ---- */
    .stButton > button, .stFormSubmitButton > button {{
        min-height: 42px; border-radius: var(--radius-sm);
        font-weight: 600; font-size: .92rem; padding: .55rem 1.1rem;
        transition: background .15s ease, color .15s ease,
            border-color .15s ease, transform .08s ease, box-shadow .15s ease;
    }}
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primaryFormSubmit"] {{
        border: 1px solid var(--accent); background: var(--accent);
        color: #fff; box-shadow: var(--shadow-sm);
    }}
    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button[kind="primaryFormSubmit"]:hover {{
        background: var(--accent-hover); border-color: var(--accent-hover);
        color: #fff; box-shadow: var(--shadow-md);
    }}
    .stButton > button[kind="secondary"],
    .stFormSubmitButton > button[kind="secondaryFormSubmit"] {{
        border: 1px solid var(--border); background: var(--surface); color: var(--text);
    }}
    .stButton > button[kind="secondary"]:hover,
    .stFormSubmitButton > button[kind="secondaryFormSubmit"]:hover {{
        border-color: var(--accent); color: var(--accent); background: var(--accent-soft);
    }}
    .stButton > button:active, .stFormSubmitButton > button:active {{ transform: translateY(1px); }}

    .stButton > button:focus-visible,
    input:focus-visible, textarea:focus-visible,
    [role="radiogroup"] *:focus-visible {{
        outline: 3px solid var(--focus) !important; outline-offset: 2px;
    }}

    /* ---- Segmented control (primary nav) ---- */
    [data-testid="stSegmentedControl"] {{ width: 100%; }}
    [data-testid="stSegmentedControl"] > div {{ width: 100%; }}
    [data-testid="stSegmentedControl"] button {{
        flex: 1 1 0; min-height: 40px; font-weight: 600;
    }}

    /* ---- Inputs ---- */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-baseweb="select"] > div, textarea {{
        min-height: 42px; border-radius: var(--radius-sm) !important;
    }}

    /* ---- Chat ---- */
    [data-testid="stChatMessage"] {{
        background: var(--surface); border: 1px solid var(--border);
        border-radius: var(--radius-md); padding: .75rem .9rem;
        margin-bottom: .65rem; box-shadow: var(--shadow-sm);
    }}
    [data-testid="stChatInput"] {{ background: var(--surface); border-top: 1px solid var(--border); }}

    [data-testid="stSidebar"] {{ display: none; }}

    [data-testid="stProgress"] div[role="progressbar"] > div {{ background: var(--accent); }}

    /* ---- Mobile ---- */
    @media (max-width: 720px) {{
        .block-container {{ padding-inline: 1rem; padding-top: .75rem; }}
        .hero {{ border-radius: var(--radius-md); padding: 1.4rem; }}
        .stButton > button, .stFormSubmitButton > button {{ width: 100%; min-height: 46px; }}
        .trust-strip {{ grid-template-columns: 1fr; }}
        [data-testid="stSegmentedControl"] button {{ font-size: .8rem; padding-inline: .2rem; }}
        [data-testid="stChatInput"] textarea {{ font-size: 16px !important; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

if st.session_state.lang == "ar":
    st.markdown(
        """
        <style>
        html { direction: rtl; }
        .stApp, .block-container { direction: rtl; text-align: right; }
        /* Mirror the order of every horizontal column block */
        [data-testid="stHorizontalBlock"] { direction: rtl; }
        [data-testid="stChatInput"] { direction: rtl; }
        [data-testid="stChatMessage"] { direction: rtl; text-align: right; }
        /* Right-align typed content (logical props already flip borders/padding) */
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-baseweb="select"] > div,
        textarea { text-align: right; }
        .hero h1, .hero p, .eyebrow,
        .section-heading h2, .section-heading p { text-align: right; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
DATASET_PATH = Path("htu_majors_interests.csv")


def dataset_version() -> str:
    if not DATASET_PATH.exists():
        return "missing"
    stat = DATASET_PATH.stat()
    return f"{stat.st_mtime_ns}-{stat.st_size}"


@st.cache_resource
def load_model(_dataset_version: str):
    from model_manager import load_current_model

    return load_current_model(
        auto_retrain=True,
        minimum_accuracy=0.60,
    )


@st.cache_resource
def load_rag():
    return AdmissionRAG()



def transcribe_recording(audio_file, language: str) -> tuple[str | None, str | None]:
    """Transcribe a Streamlit microphone recording using SpeechRecognition."""
    if audio_file is None:
        return None, None

    recognizer = sr.Recognizer()
    try:
        audio_bytes = io.BytesIO(audio_file.getvalue())
        with sr.AudioFile(audio_bytes) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio_data = recognizer.record(source)

        locale = "ar-JO" if language == "ar" else "en-US"
        transcript = recognizer.recognize_google(
            audio_data,
            language=locale,
        )
        return transcript.strip(), None
    except sr.UnknownValueError:
        return None, tr(
            "I could not understand the recording. Speak closer to the microphone and try again.",
            "لم أتمكن من فهم التسجيل. تحدث بالقرب من الميكروفون وحاول مرة أخرى.",
        )
    except sr.RequestError:
        return None, tr(
            "Voice transcription is temporarily unavailable. Check the internet connection or type your question.",
            "خدمة تحويل الصوت إلى نص غير متاحة حالياً. تحقق من الإنترنت أو اكتب سؤالك.",
        )
    except (ValueError, OSError) as error:
        return None, tr(
            f"The recording could not be processed: {error}",
            f"تعذر معالجة التسجيل: {error}",
        )


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
PAGES = {
    "home": tr("Home", "الرئيسية"),
    "discover": tr("Discover your major", "اكتشف تخصصك"),
    "eligibility": tr("Check eligibility", "تحقق من الأهلية"),
    "advisor": tr("Ask the advisor", "اسأل المستشار"),
}

# Top bar: brand on the leading side, language + theme controls trailing.
# All three stay visible on mobile (they used to hide inside the sidebar).
brand_col, controls_col = st.columns([3, 2])
with brand_col:
    st.markdown(
        f"""
        <div class="app-header">
            <div class="brand">
                <div class="brand-mark" aria-label="HTU Advisor home">H</div>
                <div>
                    <div class="brand-title">{tr("HTU Advisor", "مستشار جامعة الحسين التقنية")}</div>
                    <div class="brand-subtitle">{tr("Major choice and admission guidance", "اختيار التخصص وإرشادات القبول")}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with controls_col:
    lang_col, theme_col = st.columns(2)
    with lang_col:
        if st.button(
            tr("العربية", "English"),
            key="lang_toggle",
            use_container_width=True,
            help=tr("Switch language", "تغيير اللغة"),
        ):
            st.session_state.lang = "ar" if st.session_state.lang == "en" else "en"
            st.rerun()
    with theme_col:
        theme_label = (
            tr("Dark", "داكن")
            if st.session_state.theme == "light"
            else tr("Light", "فاتح")
        )
        if st.button(
            theme_label,
            key="theme_toggle",
            use_container_width=True,
            help=tr("Switch theme", "تغيير المظهر"),
        ):
            st.session_state.theme = (
                "dark" if st.session_state.theme == "light" else "light"
            )
            st.rerun()

# Primary navigation as a clean segmented tab bar. A sentinel keeps the
# widget's stored value pointed at the current page — this reseeds on
# programmatic navigation (home cards) and on language switches without
# clobbering a genuine user tap.
nav_labels = list(PAGES.values())
if (
    st.session_state.get("nav_synced_page") != st.session_state.page
    or st.session_state.get("nav_selection") not in nav_labels
):
    st.session_state.nav_selection = PAGES[st.session_state.page]
    st.session_state.nav_synced_page = st.session_state.page

nav_choice = st.segmented_control(
    tr("Primary navigation", "التنقل الرئيسي"),
    options=nav_labels,
    key="nav_selection",
    label_visibility="collapsed",
    width="stretch",
)
if nav_choice and nav_choice != PAGES[st.session_state.page]:
    target_page = list(PAGES.keys())[nav_labels.index(nav_choice)]
    st.session_state.page = target_page
    st.session_state.nav_synced_page = target_page
    st.rerun()

st.write("")


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------
if st.session_state.page == "home":
    st.markdown(
        f"""
        <section class="hero">
            <div class="eyebrow">{tr("Student decision support", "دعم قرار الطالب")}</div>
            <h1>{tr("Choose your path with clearer information.", "اختر مسارك بمعلومات أوضح.")}</h1>
            <p>
                {tr(
                    "Discover a suitable major, check admission eligibility, and search verified HTU information without moving between multiple documents.",
                    "اكتشف التخصص المناسب، وتحقق من شروط القبول، وابحث في معلومات جامعة الحسين التقنية الموثقة دون التنقل بين ملفات متعددة.",
                )}
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="trust-strip">
            <div class="trust-item">
                <strong>{tr("Verified datasets", "بيانات موثقة")}</strong>
                <span>{tr("Answers use the available HTU policies and study plans.", "تعتمد الإجابات على سياسات وخطط الجامعة المتاحة.")}</span>
            </div>
            <div class="trust-item">
                <strong>{tr("Fast task access", "وصول سريع")}</strong>
                <span>{tr("Core tasks are available in one click from the home page.", "المهام الأساسية متاحة بنقرة واحدة من الصفحة الرئيسية.")}</span>
            </div>
            <div class="trust-item">
                <strong>{tr("Transparent results", "نتائج شفافة")}</strong>
                <span>{tr("Advisor answers include confidence and source information.", "تتضمن إجابات المستشار مستوى الثقة ومعلومات المصادر.")}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="section-heading">
            <h2>{tr("What do you need today?", "ماذا تحتاج اليوم؟")}</h2>
            <p>{tr("Each task is reachable directly from this page.", "يمكنك الوصول إلى كل مهمة مباشرة من هذه الصفحة.")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    card_columns = st.columns(3)
    cards = [
        (
            "discover",
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><polygon points="15.5 8.5 13 13 8.5 15.5 11 11 15.5 8.5"/></svg>',
            tr("Discover your major", "اكتشف تخصصك"),
            tr(
                "Answer short scenario questions and receive a ranked major recommendation.",
                "أجب عن مواقف قصيرة واحصل على توصية مرتبة للتخصصات.",
            ),
            tr("Start discovery", "ابدأ الاكتشاف"),
        ),
        (
            "eligibility",
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l7 3v5c0 4.4-3 7.6-7 9-4-1.4-7-4.6-7-9V6l7-3z"/><path d="M9 12l2 2 4-4"/></svg>',
            tr("Check eligibility", "تحقق من الأهلية"),
            tr(
                "Enter only the essential grades and certificate details needed for the check.",
                "أدخل فقط العلامات ونوع الشهادة الضرورية للتحقق.",
            ),
            tr("Check now", "تحقق الآن"),
        ),
        (
            "advisor",
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M20 4H4a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3v4l4-4h9a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1z"/></svg>',
            tr("Ask the advisor", "اسأل المستشار"),
            tr(
                "Search admission, fees, majors, courses, study plans, interviews, and deadlines.",
                "ابحث عن القبول والرسوم والتخصصات والمواد والخطط والمقابلات والمواعيد.",
            ),
            tr("Open advisor", "افتح المستشار"),
        ),
    ]

    for column, (page, icon, title, description, action) in zip(card_columns, cards):
        with column:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon" aria-hidden="true">{icon}</div>
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                action,
                key=f"home_{page}",
                use_container_width=True,
                type="primary",
            ):
                navigate(page)

    st.markdown(
        f"""
        <div class="section-heading">
            <h2>{tr("Ask immediately", "اسأل مباشرة")}</h2>
            <p>{tr("Use the search field below to go straight to the advisor.", "استخدم مربع البحث للانتقال مباشرة إلى المستشار.")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("home_search_form", clear_on_submit=False):
        home_question = st.text_input(
            tr("Search HTU information", "ابحث في معلومات الجامعة"),
            placeholder=tr(
                "Example: What courses are in Technical Computer Science?",
                "مثال: ما مواد الدرجة التقنية في علم الحاسوب؟",
            ),
        )
        search_submitted = st.form_submit_button(
            tr("Search with the advisor", "ابحث مع المستشار"),
            use_container_width=True,
            type="primary",
        )

    if search_submitted:
        if not home_question.strip():
            st.warning(
                tr(
                    "Type a question before searching.",
                    "اكتب سؤالاً قبل البحث.",
                )
            )
        else:
            st.session_state.home_question = home_question.strip()
            st.session_state.page = "advisor"
            st.rerun()

    st.markdown(
        f"""
        <div class="section-heading">
            <h2>{tr("How it works", "كيف يعمل النظام؟")}</h2>
        </div>
        <div class="content-card">
            <div class="step-row">
                <div class="step-number">1</div>
                <div class="step-copy">
                    <strong>{tr("Choose one task", "اختر مهمة واحدة")}</strong>
                    <span>{tr("Major discovery, eligibility, or information search.", "اكتشاف التخصص أو التحقق أو البحث عن معلومة.")}</span>
                </div>
            </div>
            <div class="step-row">
                <div class="step-number">2</div>
                <div class="step-copy">
                    <strong>{tr("Provide the minimum information", "أدخل الحد الأدنى من المعلومات")}</strong>
                    <span>{tr("The interface asks only for data required for that task.", "تطلب الواجهة فقط البيانات اللازمة للمهمة.")}</span>
                </div>
            </div>
            <div class="step-row">
                <div class="step-number">3</div>
                <div class="step-copy">
                    <strong>{tr("Review the result and source", "راجع النتيجة والمصدر")}</strong>
                    <span>{tr("Chatbot answers show confidence and supporting sources.", "تعرض إجابات المستشار مستوى الثقة والمصادر الداعمة.")}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Major discovery
# ---------------------------------------------------------------------------
elif st.session_state.page == "discover":
    st.title(tr("Discover your major", "اكتشف تخصصك"))
    st.caption(
        tr(
            "A short scenario-based quiz. There are no right or wrong answers.",
            "اختبار قصير قائم على مواقف. لا توجد إجابات صحيحة أو خاطئة.",
        )
    )

    if st.session_state.quiz_step == -1:
        st.markdown(
            f"""
            <div class="content-card">
                <h3>{tr("Before you begin", "قبل أن تبدأ")}</h3>
                <p>{tr(
                    "Choose the option that feels most natural to you. The result is guidance, not an admission decision.",
                    "اختر الخيار الأقرب لطبيعتك. النتيجة إرشادية وليست قرار قبول.",
                )}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            tr("Start the quiz", "ابدأ الاختبار"),
            key="start_quiz",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.quiz_step = 0
            st.rerun()

    elif 0 <= st.session_state.quiz_step < len(SCENARIOS):
        index = st.session_state.quiz_step
        scenario = SCENARIOS[index]

        st.progress(
            index / len(SCENARIOS),
            text=tr(
                f"Question {index + 1} of {len(SCENARIOS)}",
                f"السؤال {index + 1} من {len(SCENARIOS)}",
            ),
        )

        st.markdown(
            f"""
            <div class="content-card">
                <h3>{scenario[st.session_state.lang]}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        labels = [
            option[st.session_state.lang]
            for option in scenario["options"]
        ]
        saved_choice = st.session_state.quiz_answers.get(scenario["id"])
        selected = st.radio(
            tr("Choose one option", "اختر خياراً واحداً"),
            labels,
            index=saved_choice if saved_choice is not None else None,
            key=f"quiz_{scenario['id']}",
        )
        if selected is None:
            st.caption(
                tr(
                    "Select an option to continue.",
                    "اختر خياراً للمتابعة.",
                )
            )

        back_column, next_column = st.columns(2)
        with back_column:
            if st.button(
                tr("Back", "السابق"),
                key=f"quiz_back_{scenario['id']}",
                disabled=index == 0,
                use_container_width=True,
            ):
                st.session_state.quiz_step -= 1
                st.rerun()

        with next_column:
            if st.button(
                tr("Next", "التالي"),
                key=f"quiz_next_{scenario['id']}",
                use_container_width=True,
                type="primary",
                disabled=selected is None,
            ):
                st.session_state.quiz_answers[scenario["id"]] = labels.index(selected)
                st.session_state.quiz_step += 1
                st.rerun()

    else:
        with st.spinner(tr("Preparing your result…", "جاري تجهيز النتيجة…")):
            model, label_encoder, feature_columns = load_model(dataset_version())
            vector = scenarios_to_feature_vector(
                st.session_state.quiz_answers,
                feature_columns,
            )
            inputs = pd.DataFrame([vector], columns=feature_columns)
            predicted_index = model.predict(inputs)[0]
            predicted_major = label_encoder.inverse_transform(
                [predicted_index]
            )[0]

            probabilities = None
            try:
                probabilities = model.predict_proba(inputs)[0]
            except (AttributeError, ValueError):
                pass

        display_major = (
            MAJOR_NAMES_AR.get(predicted_major, predicted_major)
            if st.session_state.lang == "ar"
            else predicted_major
        )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="eyebrow">{tr("Best current match", "أفضل تطابق حالي")}</div>
                <h2>{display_major}</h2>
                <p>{tr(
                    "Use this as a starting point. Review the study plan and admission requirements before deciding.",
                    "استخدم النتيجة كنقطة بداية، وراجع الخطة وشروط القبول قبل اتخاذ القرار.",
                )}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if probabilities is not None:
            st.subheader(tr("Top matches", "أفضل التطابقات"))
            ranked = sorted(
                zip(label_encoder.classes_, probabilities),
                key=lambda item: -item[1],
            )[:3]

            for major, probability in ranked:
                name = (
                    MAJOR_NAMES_AR.get(major, major)
                    if st.session_state.lang == "ar"
                    else major
                )
                st.write(f"**{name}**")
                st.progress(float(probability), text=f"{probability * 100:.0f}%")

        action_columns = st.columns(2)
        with action_columns[0]:
            if st.button(
                tr("Review admission eligibility", "تحقق من شروط القبول"),
                use_container_width=True,
                type="primary",
            ):
                navigate("eligibility")
        with action_columns[1]:
            if st.button(
                tr("Restart quiz", "أعد الاختبار"),
                use_container_width=True,
            ):
                st.session_state.quiz_step = -1
                st.session_state.quiz_answers = {}
                st.rerun()


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------
elif st.session_state.page == "eligibility":
    st.title(tr("Check admission eligibility", "تحقق من أهلية القبول"))
    st.caption(
        tr(
            "Only essential fields are requested. Results are preliminary and should be verified with HTU.",
            "نطلب الحقول الضرورية فقط. النتيجة أولية ويجب التحقق منها مع الجامعة.",
        )
    )

    major_options = list(MAJOR_ADMISSION_FAMILY.keys())
    certificate_labels = {
        "tawjihi": tr("Jordanian Tawjihi", "توجيهي أردني"),
        "igcse": tr("British IGCSE", "IGCSE بريطانية"),
        "ib": tr("International Baccalaureate", "بكالوريا دولية IB"),
        "american": tr("American ACT / AP / SAT II", "أمريكية ACT / AP / SAT II"),
        "btec": "BTEC",
    }
    tier_labels = {
        "bachelor": tr("Bachelor", "بكالوريوس"),
        "tech": tr("Technical degree", "درجة تقنية"),
        "applied": tr("Technician degree", "درجة فني"),
    }

    with st.form("eligibility_form"):
        left, right = st.columns(2)

        with left:
            major = st.selectbox(
                tr("Major", "التخصص"),
                major_options,
                format_func=lambda value: (
                    MAJOR_NAMES_AR.get(value, value)
                    if st.session_state.lang == "ar"
                    else {
                        "CS": "Computer Science",
                        "Cybersecurity": "Cyber Security",
                        "Architecture Engineering": "Architectural Engineering",
                        "Game Design & Development": "Game Design and Development",
                    }.get(value, value)
                ),
            )
            certificate = st.selectbox(
                tr("Secondary certificate", "نوع الشهادة الثانوية"),
                list(certificate_labels),
                format_func=lambda value: certificate_labels[value],
            )

        with right:
            degree = st.selectbox(
                tr("Requested degree", "الدرجة المطلوبة"),
                list(tier_labels),
                format_func=lambda value: tier_labels[value],
            )
            average = st.number_input(
                tr("Overall average (%)", "المعدل العام (%)"),
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=0.5,
            )

        math_value = None
        physics_value = None

        if certificate in {"tawjihi", "ib"}:
            maximum = 100.0 if certificate == "tawjihi" else 7.0
            default = 80.0 if certificate == "tawjihi" else 5.0

            if requires_physics(major):
                grade_left, grade_right = st.columns(2)
                with grade_left:
                    math_value = st.number_input(
                        tr("Mathematics grade", "علامة الرياضيات"),
                        min_value=0.0,
                        max_value=maximum,
                        value=default,
                    )
                with grade_right:
                    physics_value = st.number_input(
                        tr("Physics grade", "علامة الفيزياء"),
                        min_value=0.0,
                        max_value=maximum,
                        value=default,
                    )
            else:
                math_value = st.number_input(
                    tr("Mathematics grade", "علامة الرياضيات"),
                    min_value=0.0,
                    max_value=maximum,
                    value=default,
                )
                physics_value = None
                st.caption(
                    tr(
                        "Physics is not a separate minimum requirement for computing programmes.",
                        "الفيزياء ليست شرط علامة منفصل لتخصصات كلية الحاسوب والمعلومات.",
                    )
                )

        elif certificate == "igcse":
            grades = ["A*", "A", "B", "C", "D", "E"]
            if requires_physics(major):
                grade_left, grade_right = st.columns(2)
                with grade_left:
                    math_value = st.selectbox(
                        tr("Mathematics grade", "علامة الرياضيات"),
                        grades,
                        index=2,
                    )
                with grade_right:
                    physics_value = st.selectbox(
                        tr("Physics grade", "علامة الفيزياء"),
                        grades,
                        index=2,
                    )
            else:
                math_value = st.selectbox(
                    tr("Mathematics grade", "علامة الرياضيات"),
                    grades,
                    index=2,
                )
                physics_value = None

        elif certificate == "btec":
            math_value = st.selectbox(
                tr("Mathematics result", "نتيجة الرياضيات"),
                ["Pass", "Merit", "Distinction"],
                index=1,
            )

        elif certificate == "american":
            st.info(
                tr(
                    "Enter at least one available score for mathematics and physics.",
                    "أدخل علامة واحدة متوفرة على الأقل للرياضيات والفيزياء.",
                )
            )
            score_columns = st.columns(3)
            with score_columns[0]:
                act_math = st.number_input("ACT Math", 0, 36, 0)
                act_physics = st.number_input("ACT Physics", 0, 36, 0)
            with score_columns[1]:
                ap_math = st.number_input("AP Math", 0, 5, 0)
                ap_physics = st.number_input("AP Physics", 0, 5, 0)
            with score_columns[2]:
                sat_math = st.number_input("SAT II Math", 0, 800, 0)
                sat_physics = st.number_input("SAT II Physics", 0, 800, 0)

            math_value = {
                "act": act_math or None,
                "ap": ap_math or None,
                "sat2": sat_math or None,
            }
            physics_value = {
                "act": act_physics or None,
                "ap": ap_physics or None,
                "sat2": sat_physics or None,
            }

        submitted = st.form_submit_button(
            tr("Check eligibility", "تحقق من الأهلية"),
            use_container_width=True,
            type="primary",
        )

    if submitted:
        if certificate == "american":
            has_math = any(math_value.values())
            has_physics = any(physics_value.values())
            physics_needed = requires_physics(major)
            if not has_math or (physics_needed and not has_physics):
                message = (
                    tr(
                        "Enter at least one mathematics score and one physics score.",
                        "أدخل علامة رياضيات واحدة وعلامة فيزياء واحدة على الأقل.",
                    )
                    if physics_needed
                    else tr(
                        "Enter at least one mathematics score.",
                        "أدخل علامة رياضيات واحدة على الأقل.",
                    )
                )
                st.error(message)
                st.stop()

        with st.spinner(tr("Checking requirements…", "جاري التحقق من الشروط…")):
            eligible, reason = check_eligibility(
                major,
                certificate,
                degree,
                math_value,
                physics_value,
                average,
            )

        if eligible is None:
            st.info(
                tr(
                    "A verified rule is not available for this exact combination. Contact Admissions and Registration.",
                    "لا توجد قاعدة موثقة لهذه الحالة بالتحديد. تواصل مع القبول والتسجيل.",
                )
            )
        elif eligible:
            st.success(reason)
        else:
            st.error(reason)


# ---------------------------------------------------------------------------
# Advisor chatbot
# ---------------------------------------------------------------------------
else:
    st.title(tr("Ask the HTU advisor", "اسأل مستشار الجامعة"))
    st.caption(
        tr(
            "Ask about admissions, majors, courses, plans, interviews, fees, scholarships, and contacts.",
            "اسأل عن القبول والتخصصات والمواد والخطط والمقابلات والرسوم والمنح والتواصل.",
        )
    )

    rag = load_rag()

    heading_col, clear_col = st.columns([5, 1])
    with heading_col:
        st.markdown(
            f"""
            <div class="status-note">
                {tr(
                    "Answers are grounded in the available HTU datasets. Open Sources under an answer to verify it.",
                    "تعتمد الإجابات على بيانات الجامعة المتاحة. افتح المصادر أسفل الإجابة للتحقق منها.",
                )}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with clear_col:
        if st.button(
            tr("Clear chat", "مسح المحادثة"),
            use_container_width=True,
        ):
            st.session_state.chat_history = []
            st.session_state.chat_context = {}
            st.rerun()

    quick_categories = {
        tr("Apply and documents", "التقديم والوثائق"): [
            tr("How do I start an application to HTU?", "كيف أبدأ طلب الالتحاق؟"),
            tr("What are all application steps?", "ما خطوات التقديم كاملة؟"),
            tr("Which documents are required?", "ما الوثائق المطلوبة؟"),
            tr("Can I edit my application?", "هل يمكن تعديل الطلب؟"),
        ],
        tr("Admission eligibility", "شروط القبول"): [
            tr("What is the minimum bachelor average?", "ما الحد الأدنى لمعدل البكالوريوس؟"),
            tr("What are the IGCSE requirements?", "ما شروط IGCSE؟"),
            tr("What are the IB requirements?", "ما شروط IB؟"),
            tr("What are the BTEC requirements?", "ما شروط BTEC؟"),
            tr("What are the Electrical Engineering requirements?", "ما شروط الهندسة الكهربائية؟"),
        ],
        tr("Majors and comparisons", "التخصصات والمقارنات"): [
            tr("Which majors are available at HTU?", "ما التخصصات المتاحة؟"),
            tr("What is Computer Science about?", "ما طبيعة علم الحاسوب؟"),
            tr("What is Cyber Security about?", "ما طبيعة الأمن السيبراني؟"),
            tr("What is Architectural Engineering about?", "ما طبيعة الهندسة المعمارية؟"),
            tr("Compare Computer Science and Cyber Security.", "قارن بين علم الحاسوب والأمن السيبراني."),
        ],
        tr("Courses and study plans", "المواد والخطط الدراسية"): [
            tr("What courses are in Computer Science BSc?", "ما مواد بكالوريوس علم الحاسوب؟"),
            tr("What courses are in Technical Computer Science?", "ما مواد الدرجة التقنية في علم الحاسوب؟"),
            tr("What courses are in Cyber Security BSc?", "ما مواد بكالوريوس الأمن السيبراني؟"),
            tr("How many credit hours are in Electrical Engineering?", "كم عدد ساعات الهندسة الكهربائية؟"),
            tr("What are the course prerequisites?", "ما المتطلبات السابقة للمواد؟"),
        ],
        tr("Interview and assessment", "المقابلة والتقييم"): [
            tr("What does the admission test assess?", "ماذا يقيس اختبار القبول؟"),
            tr("What is required in the introductory video?", "ما المطلوب في الفيديو التعريفي؟"),
            tr("Do activities add bonus points?", "هل الأنشطة تضيف نقاطاً؟"),
            tr("How is the competitive score calculated?", "كيف تحسب علامة القبول التنافسية؟"),
        ],
        tr("Fees, aid, dates, and contact", "الرسوم والمنح والمواعيد والتواصل"): [
            tr("How much is the registration fee?", "كم رسوم التسجيل؟"),
            tr("How much is each credit hour?", "كم رسوم الساعة؟"),
            tr("Which scholarships are available?", "ما المنح المتاحة؟"),
            tr("When does application open and close?", "متى يبدأ وينتهي التقديم؟"),
            tr("Where is HTU located?", "أين تقع الجامعة؟"),
        ],
    }

    with st.expander(
        tr("Prepared questions", "أسئلة جاهزة"),
        expanded=not st.session_state.chat_history,
    ):
        category_tabs = st.tabs(list(quick_categories))
        for tab, category in zip(category_tabs, quick_categories):
            with tab:
                chip_columns = st.columns(2)
                for position, question in enumerate(quick_categories[category]):
                    with chip_columns[position % 2]:
                        if st.button(
                            question,
                            key=f"quick_{category}_{position}",
                            use_container_width=True,
                        ):
                            st.session_state.pending_quick_question = question
                            st.rerun()

    if not st.session_state.chat_history:
        st.markdown(
            f"""
            <div class="chat-empty">
                <div class="chat-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M20 4H4a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3v4l4-4h9a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1z"/></svg></div>
                <h3>{tr("Start with a question or your voice", "ابدأ بسؤال مكتوب أو بصوتك")}</h3>
                <p>{tr(
                    "Choose a prepared question, type below, or record a short voice question using the microphone panel.",
                    "اختر سؤالاً جاهزاً، أو اكتب في الأسفل، أو سجل سؤالاً صوتياً قصيراً من لوحة الميكروفون.",
                )}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    input_mode = st.segmented_control(
        tr("Question input method", "طريقة إدخال السؤال"),
        options=[
            tr("Keyboard", "الكتابة"),
            tr("Microphone", "الميكروفون"),
        ],
        default=tr("Keyboard", "الكتابة"),
        key="advisor_input_mode",
    )

    voice_question = None
    if input_mode == tr("Microphone", "الميكروفون"):
        st.markdown(
            f"""
            <div class="voice-panel">
                <h4>{tr("Record your question", "سجل سؤالك")}</h4>
                <p>{tr(
                    "Allow microphone access, speak clearly, then stop the recording. Transcription requires an internet connection.",
                    "اسمح بالوصول إلى الميكروفون، وتحدث بوضوح، ثم أوقف التسجيل. تحويل الصوت إلى نص يحتاج اتصالاً بالإنترنت.",
                )}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        recording = st.audio_input(
            tr("Microphone recording", "تسجيل الميكروفون"),
            key=f"advisor_microphone_{st.session_state.voice_recorder_nonce}",
        )
        if recording is not None:
            with st.spinner(
                tr(
                    "Converting your voice to text…",
                    "جاري تحويل الصوت إلى نص…",
                )
            ):
                transcript, voice_error = transcribe_recording(
                    recording,
                    st.session_state.lang,
                )

            if voice_error:
                st.error(voice_error)
            elif transcript:
                st.caption(
                    tr(
                        "Transcription ready — review it before sending.",
                        "النص جاهز — راجعه قبل الإرسال.",
                    )
                )
                edited_transcript = st.text_area(
                    tr(
                        "Review or edit the transcription",
                        "راجع النص أو عدله",
                    ),
                    value=transcript,
                    key="voice_transcript_editor",
                    height=90,
                )
                if st.button(
                    tr(
                        "Send voice question",
                        "أرسل السؤال الصوتي",
                    ),
                    use_container_width=True,
                    key="send_voice_question",
                    type="primary",
                ):
                    if edited_transcript.strip():
                        st.session_state.pending_voice_question = (
                            edited_transcript.strip()
                        )
                        st.session_state.voice_recorder_nonce += 1
                        st.rerun()
                    else:
                        st.warning(
                            tr(
                                "The transcription is empty.",
                                "النص المحول فارغ.",
                            )
                        )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            sources = message.get("sources", [])
            if sources:
                with st.expander(tr("Sources", "المصادر")):
                    for source in sources:
                        st.write(f"- {source}")
                confidence = message.get("confidence")
                if confidence:
                    st.markdown(
                        f"""
                        <div class="answer-meta">
                            <span class="meta-badge">✓ {tr("Grounded answer", "إجابة مدعومة")}</span>
                            <span class="meta-badge">{tr("Confidence", "الثقة")}: {confidence}</span>
                            <span class="meta-badge">{len(sources)} {tr("sources", "مصادر")}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    pending_voice = st.session_state.pop("pending_voice_question", None)
    pending_question = st.session_state.pop("pending_quick_question", None)
    initial_question = st.session_state.home_question
    st.session_state.home_question = ""

    typed_question = st.chat_input(
        tr(
            "Type a specific HTU question",
            "اكتب سؤالاً محدداً عن الجامعة",
        )
    )
    prompt = (
        pending_voice
        or voice_question
        or pending_question
        or initial_question
        or typed_question
    )

    if prompt:
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(
                tr(
                    "Searching verified HTU sources…",
                    "جاري البحث في مصادر الجامعة…",
                )
            ):
                result = rag.answer(
                    prompt,
                    context=st.session_state.chat_context,
                )

            st.markdown(result["answer"])
            unique_sources = list(
                dict.fromkeys(result.get("sources", []))
            )
            if unique_sources:
                with st.expander(tr("Sources", "المصادر")):
                    for source in unique_sources:
                        st.write(f"- {source}")
                st.markdown(
                    f"""
                    <div class="answer-meta">
                        <span class="meta-badge">✓ {tr("Grounded answer", "إجابة مدعومة")}</span>
                        <span class="meta-badge">{tr("Confidence", "الثقة")}: {result.get("confidence", "unknown")}</span>
                        <span class="meta-badge">{len(unique_sources)} {tr("sources", "مصادر")}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.session_state.chat_context = result.get(
            "context",
            st.session_state.chat_context,
        )
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": unique_sources,
                "confidence": result.get("confidence"),
            }
        )
        st.rerun()


st.markdown(
    f"""
    <div class="footer">
        {tr(
            "HTU Advisor prototype · Not an official admission decision system · Verify final information with HTU.",
            "نموذج مستشار HTU · ليس نظام قرار قبول رسمي · تحقق من المعلومات النهائية مع الجامعة.",
        )}
    </div>
    """,
    unsafe_allow_html=True,
)
