import os
import re
import time
from typing import Any

import requests
import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="헤이 친구 AI 스피커",
    page_icon="🎧",
    layout="wide",
)

WIDGET_KEY = "ai_speaker_component_widget"
APP_STATE_KEY = "ai_speaker_app_state"
EVENT_KEY = "ai_speaker_event_key"
DEFAULT_WAKE_WORD = "헤이 친구"


def normalize_text(value: str) -> str:
    return (value or "").replace("\u3000", " ").strip().lower()


def detect_wake_word(text: str) -> bool:
    normalized = normalize_text(text)
    return "헤이친구" in normalized or "헤이 친구" in normalized


def detect_stop_command(text: str) -> bool:
    normalized = normalize_text(text)
    return "종료" in normalized or "그만" in normalized or "quit" in normalized


def get_groq_client() -> OpenAI | None:
    api_key = st.session_state.get("groq_api_key", "").strip()
    if not api_key:
        api_key = st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def get_search_results(query: str) -> list[dict[str, str]]:
    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_redirect": 1,
                "no_html": 1,
                "skip_disambig": 1,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        items: list[dict[str, str]] = []
        heading = data.get("Heading")
        abstract = data.get("AbstractText")
        abstract_url = data.get("AbstractURL")
        if abstract:
            items.append({"title": heading or "요약", "snippet": abstract, "url": abstract_url or ""})
        for topic in data.get("RelatedTopics", [])[:4]:
            if isinstance(topic, dict):
                title = topic.get("Text", "").split(" - ")[0].strip() or "관련 정보"
                snippet = topic.get("Text", "").strip()
                url = topic.get("FirstURL", "")
                if snippet:
                    items.append({"title": title, "snippet": snippet, "url": url})
        return items[:5]
    except Exception:
        return []


def build_system_prompt() -> str:
    return (
        "당신은 한국어 친근한 AI 스피커 '헤이 친구'입니다. "
        "기본적으로 따뜻하고 간결하게 답하고, 사용자가 질문에 대해 최신 정보를 원하면 "
        "검색 결과를 기반으로 사실과 출처를 함께 설명하되 추정하지 마세요. "
        "최근 대화 맥락을 유지하고 대답은 한국어로 합니다."
    )


def generate_reply(user_text: str) -> str:
    search_query = user_text.strip()
    if re.search(r"검색|웹검색|정보|최신|뉴스|뭐가|어디", search_query):
        results = get_search_results(search_query)
        if results:
            snippets = []
            for idx, item in enumerate(results[:3], start=1):
                title = item.get("title", f"검색 결과 {idx}")
                snippet = item.get("snippet", "").strip()
                url = item.get("url", "")
                if url:
                    snippets.append(f"{idx}. {title}: {snippet} — {url}")
                else:
                    snippets.append(f"{idx}. {title}: {snippet}")
            return "최신 정보는 다음과 같습니다.\n" + "\n".join(snippets)
        return "검색 결과를 가져오지 못했습니다. 잠시 후 다시 시도해 주세요."

    client = get_groq_client()
    if client is None:
        return (
            "Groq API 키가 설정되지 않아 기본 안내 응답으로 동작 중입니다. "
            "Streamlit Secrets의 GROQ_API_KEY를 추가하면 실제 AI 응답을 받을 수 있습니다."
        )

    messages = [{"role": "system", "content": build_system_prompt()}]
    for item in st.session_state.get("messages", []):
        messages.append({"role": item["role"], "content": item["text"]})
    messages.append({"role": "user", "content": user_text})

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.7,
            max_tokens=400,
        )
        content = response.choices[0].message.content.strip()
        return content or "답변을 생성하지 못했습니다. 다시 말씀해 주세요."
    except Exception:
        return "현재 AI 응답 생성에 실패했습니다. 잠시 후 다시 시도해 주세요."


def add_message(role: str, text: str) -> None:
    messages = st.session_state.setdefault("messages", [])
    messages.append({"role": role, "text": text, "time": time.strftime("%H:%M")})
    if len(messages) > 20:
        st.session_state["messages"] = messages[-20:]


def ensure_state() -> None:
    st.session_state.setdefault("messages", [
        {"role": "assistant", "text": "반가워요! 오늘은 무엇을 도와드릴까요?", "time": time.strftime("%H:%M")}
    ])
    st.session_state.setdefault("reminders", [])
    st.session_state.setdefault("last_tts_text", "")
    st.session_state.setdefault("last_seen_transcript", "")
    st.session_state.setdefault("wake_word", DEFAULT_WAKE_WORD)
    st.session_state.setdefault("manual_input", "")
    st.session_state.setdefault("groq_api_key", "")
    st.session_state.setdefault("conversation_started", False)
    st.session_state.setdefault("voice_active", False)


def sync_component_app_state() -> None:
    default_state = {
        "status": "idle",
        "transcript": "",
        "wakeWord": "미감지",
        "lastCommand": "대기",
        "event_id": "",
    }
    current_state = st.session_state.get(APP_STATE_KEY, default_state)
    if not isinstance(current_state, dict):
        current_state = default_state.copy()
    st.session_state[APP_STATE_KEY] = current_state


def process_voice_event(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        return
    if not st.session_state.get("conversation_started", False):
        return

    transcript = payload.get("transcript", "").strip()
    wake_word = payload.get("wakeWord", "미감지")
    status = payload.get("status", "idle")
    last_command = payload.get("lastCommand", "대기")

    if not transcript:
        return

    if transcript == st.session_state.get("last_seen_transcript"):
        return

    st.session_state["last_seen_transcript"] = transcript

    if detect_stop_command(transcript):
        add_message("assistant", "호출 대기 상태로 돌아갑니다. 무엇을 도와드릴까요?")
        st.session_state["last_tts_text"] = "호출 대기 상태로 돌아갑니다. 무엇을 도와드릴까요?"
        st.session_state["last_seen_transcript"] = ""
        return

    if detect_wake_word(transcript) or wake_word == "감지":
        add_message("user", transcript)
        reply = generate_reply(transcript.replace(DEFAULT_WAKE_WORD, "").strip() or transcript)
        add_message("assistant", reply)
        st.session_state["last_tts_text"] = reply
        st.session_state["wake_word"] = DEFAULT_WAKE_WORD
    else:
        add_message("user", transcript)
        reply = generate_reply(transcript)
        add_message("assistant", reply)
        st.session_state["last_tts_text"] = reply

    st.session_state[APP_STATE_KEY] = {
        "status": status,
        "transcript": transcript,
        "wakeWord": wake_word,
        "lastCommand": last_command,
        "event_id": payload.get("event_id", ""),
    }


def add_reminder() -> None:
    title = st.session_state.get("reminder_title", "").strip()
    when = st.session_state.get("reminder_when", "")
    if not title or not when:
        return
    reminder = {"title": title, "when": when, "notified": False}
    st.session_state["reminders"].append(reminder)
    st.session_state["reminder_title"] = ""
    st.session_state["reminder_when"] = ""


def render_reminders() -> None:
    reminders = st.session_state.get("reminders", [])
    for idx, item in enumerate(reminders):
        with st.container():
            st.markdown(f"- {item['title']} · {item['when']}")
            if st.button(f"삭제 {idx + 1}", key=f"del_reminder_{idx}"):
                reminders.pop(idx)
                st.session_state["reminders"] = reminders
                st.rerun()


def render_sidebar_voice_status() -> None:
    state = st.session_state.get(APP_STATE_KEY, {})
    status = state.get("status", "idle")
    label_map = {
        "idle": "대기 중",
        "listening": "듣는 중",
        "result_ready": "인식 완료",
        "error": "오류",
        "processing": "처리 중",
    }
    transcript = state.get("transcript", "") or "없음"
    wake_word = state.get("wakeWord", "미감지")
    last_command = state.get("lastCommand", "대기")
    st.subheader("음성 상태")
    st.markdown(
        f"""
        <div style="background:#f3f4f6;border-radius:12px;padding:12px;color:#111827;line-height:1.8;">
          <div>상태: <strong>{label_map.get(status, '대기 중')}</strong></div>
          <div>최근 인식: <strong>{transcript}</strong></div>
          <div>호출어: <strong>{wake_word}</strong></div>
          <div>최근 명령: <strong>{last_command}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_component() -> None:
    component_state = st.session_state.get(APP_STATE_KEY, {
        "status": "idle",
        "transcript": "",
        "wakeWord": "미감지",
        "lastCommand": "대기",
        "event_id": "",
        "conversation_started": st.session_state.get("conversation_started", False),
    })
    tts_text = st.session_state.get("last_tts_text", "")

    def on_payload_change() -> None:
        if isinstance(st.session_state.get(WIDGET_KEY, {}), dict):
            payload = st.session_state[WIDGET_KEY].get("payload")
            if isinstance(payload, dict):
                st.session_state[APP_STATE_KEY] = payload

    component_state["conversation_started"] = st.session_state.get("conversation_started", False)

    component_factory = st.components.v2.component(
        "ai_speaker_component",
        html="<div id='root'></div>",
        js=r"""
        function normalizeText(value) {
          return String(value || '').trim().toLowerCase().replace(/\s+/g, '');
        }

        function makeEventId() {
          return 'evt_' + Date.now() + '_' + Math.random().toString(16).slice(2, 10);
        }

        function evaluateTranscript(text) {
          const normalized = normalizeText(text);
          return {
            wakeDetected: normalized.includes('헤이친구') || normalized.includes('헤이 친구'),
            stopDetected: normalized.includes('종료') || normalized.includes('그만') || normalized.includes('quit'),
            normalized,
          };
        }

        function speakText(text) {
          if (!text) return;
          const synth = window.speechSynthesis;
          if (!synth) return;
          synth.cancel();
          const utterance = new SpeechSynthesisUtterance(text);
          utterance.lang = 'ko-KR';
          utterance.rate = 1.0;
          utterance.pitch = 1.1;
          const voices = synth.getVoices();
          const preferred = voices.find(v => /ko|kr/i.test(v.lang) && /(female|woman|여성|여자)/i.test(v.name))
            || voices.find(v => /ko|kr/i.test(v.lang));
          if (preferred) utterance.voice = preferred;
          synth.speak(utterance);
        }

        export default function(component) {
          const { parentElement, data, setStateValue } = component;
          const root = parentElement.querySelector('#root');
          if (!root) return;

          const current = data && data.payload ? data.payload : {
            status: 'idle',
            transcript: '',
            wakeWord: '미감지',
            lastCommand: '대기',
            event_id: '',
            conversation_started: false,
          };
          const conversationStarted = !!(data && data.conversation_started) || !!current.conversation_started;
          const ttsText = (data && data.tts_text) || '';

          const updateState = (nextPayload) => {
            const merged = {
              ...current,
              ...nextPayload,
              event_id: nextPayload && nextPayload.event_id ? nextPayload.event_id : makeEventId(),
            };
            current.status = merged.status;
            current.transcript = merged.transcript ?? current.transcript;
            current.wakeWord = merged.wakeWord ?? current.wakeWord;
            current.lastCommand = merged.lastCommand ?? current.lastCommand;
            setStateValue('payload', merged);
          };

          const render = () => {
            const status = current.status || 'idle';
            const transcript = current.transcript || '';
            const wakeWord = current.wakeWord || '미감지';
            const lastCommand = current.lastCommand || '대기';
            const support = !!(window.SpeechRecognition || window.webkitSpeechRecognition);

            root.innerHTML = `
              <div style="background:#f3f4f6;border-radius:16px;padding:14px;box-shadow:0 8px 18px rgba(0,0,0,0.04);color:#111827;">
                <div style="font-size:13px;color:#4b5563;margin-bottom:6px;">음성 인식 상태</div>
                <div style="font-size:12px;line-height:1.7;">
                  <div>브라우저 지원: <strong>${support ? '지원' : '미지원'}</strong></div>
                  <div>상태: <strong>${status}</strong></div>
                  <div>결과: <strong>${transcript || '없음'}</strong></div>
                  <div>호출어: <strong>${wakeWord}</strong></div>
                  <div>최근 명령: <strong>${lastCommand}</strong></div>
                </div>
              </div>
            `;
          };

          const startListening = () => {
            const RecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!RecognitionCtor) {
              updateState({ status: 'error', lastCommand: '브라우저 미지원' });
              return null;
            }

            const recognition = new RecognitionCtor();
            recognition.lang = 'ko-KR';
            recognition.continuous = false;
            recognition.interimResults = false;

            recognition.onstart = () => updateState({ status: 'listening', lastCommand: '듣는 중' });
            recognition.onresult = (event) => {
              const lastResult = event.results[event.results.length - 1];
              const text = lastResult[0].transcript.trim();
              const info = evaluateTranscript(text);
              updateState({
                status: 'result_ready',
                transcript: text,
                wakeWord: info.wakeDetected ? '감지' : '미감지',
                lastCommand: info.stopDetected ? '종료' : '대화',
              });
            };
            recognition.onerror = () => updateState({ status: 'error', lastCommand: '음성 인식 오류' });
            recognition.onend = () => {
              if (conversationStarted) {
                try {
                  recognition.start();
                } catch (err) {
                  updateState({ status: 'idle', lastCommand: '재시도' });
                }
              } else {
                updateState({ status: 'idle', lastCommand: '대기' });
              }
            };

            try {
              recognition.start();
            } catch (err) {
              updateState({ status: 'idle', lastCommand: '재시도' });
            }
            return recognition;
          };

          if (ttsText) {
            speakText(ttsText);
          }

          if (conversationStarted) {
            startListening();
          }

          render();
        }
        """,
        css="",
        isolate_styles=True,
    )

    result = component_factory(
        key=WIDGET_KEY,
        data={
            "payload": component_state,
            "conversation_started": st.session_state.get("conversation_started", False),
        },
        default={
            "payload": component_state,
        },
        on_payload_change=on_payload_change,
    )

    if result is not None and isinstance(result, dict):
        next_payload = result.get("payload")
        if isinstance(next_payload, dict):
            st.session_state[APP_STATE_KEY] = next_payload

    component_state = st.session_state.get(APP_STATE_KEY, {
        "status": "idle",
        "transcript": "",
        "wakeWord": "미감지",
        "lastCommand": "대기",
        "event_id": "",
        "conversation_started": st.session_state.get("conversation_started", False),
    })
    process_voice_event(component_state)


ensure_state()

st.title("🎧 헤이 친구 AI 스피커")
st.caption("음성 인식 + 대화 + 검색 + 일정 알림을 한 번에")

with st.sidebar:
    st.subheader("대화 제어")
    if st.button(
        "대화 시작" if not st.session_state.get("conversation_started", False) else "대화 종료",
        use_container_width=True,
    ):
        if st.session_state.get("conversation_started", False):
            st.session_state["conversation_started"] = False
            st.session_state["voice_active"] = False
            add_message("assistant", "대화가 종료되었습니다. 필요하면 다시 시작해 주세요.")
            st.session_state["last_tts_text"] = "대화가 종료되었습니다. 필요하면 다시 시작해 주세요."
        else:
            st.session_state["conversation_started"] = True
            st.session_state["voice_active"] = True
            st.session_state["last_seen_transcript"] = ""
            add_message("assistant", "대화가 시작되었습니다. 무엇을 도와드릴까요?")
            st.session_state["last_tts_text"] = "대화가 시작되었습니다. 무엇을 도와드릴까요?"

    render_sidebar_voice_status()
    render_component()
    st.caption("음성 인식 상태는 여기에서 확인합니다.")

    st.subheader("설정")
    st.text_input("GROQ API Key", key="groq_api_key", type="password", help="브라우저 세션 메모리에만 저장됩니다. 서버 로그에는 남기지 않습니다.")
    st.text_input("호출어", key="wake_word")
    st.text_input("음성 안내 문구", key="manual_input")
    if st.button("직접 답변 전송"):
        value = st.session_state.manual_input.strip()
        if value:
            add_message("user", value)
            reply = generate_reply(value)
            add_message("assistant", reply)
            st.session_state["last_tts_text"] = reply
            st.session_state.manual_input = ""

    st.subheader("일정/리마인더")
    st.text_input("제목", key="reminder_title")
    st.text_input("시간", key="reminder_when")
    if st.button("추가"):
        add_reminder()
    render_reminders()

    st.subheader("보안")
    st.caption("입력한 Groq 키는 현재 브라우저 세션 메모리에만 저장되며, 서버 로그에는 남기지 않습니다.")

main_col, side_col = st.columns([2, 1])
with main_col:
    st.subheader("대화 기록")
    if not st.session_state.get("conversation_started", False):
        st.info("대화 시작 버튼을 누르면 음성 대화가 활성화됩니다.")
    for item in st.session_state.get("messages", []):
        with st.chat_message(item["role"]):
            st.write(item["text"])
            st.caption(item["time"])


if st.button("대화 기록 삭제"):
    st.session_state["messages"] = [
        {"role": "assistant", "text": "반가워요! 오늘은 무엇을 도와드릴까요?", "time": time.strftime("%H:%M")}
    ]
    st.session_state["last_tts_text"] = ""

for reminder in st.session_state.get("reminders", []):
    if not reminder.get("notified") and reminder.get("when"):
        if reminder["when"] <= time.strftime("%Y-%m-%d %H:%M"):
            st.toast(f"리마인더: {reminder['title']}")
            reminder["notified"] = True
            st.session_state["reminders"] = st.session_state["reminders"]

st.caption("개인용 서비스: 브라우저 세션과 서버 메모리 내에서만 동작하며, 세션 종료 후에는 저장되지 않습니다.")
