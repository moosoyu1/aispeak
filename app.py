import os
import re
import time
from typing import Any

import requests
import streamlit as st

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


def resolve_groq_api_key() -> str:
    cached = (st.session_state.get("groq_api_key") or "").strip()
    if cached:
        return cached

    secret_value = ""
    try:
        if hasattr(st, "secrets"):
            secret_value = (st.secrets.get("GROQ_API_KEY") or "").strip()
    except Exception:
        secret_value = ""

    if secret_value:
        st.session_state["groq_api_key"] = secret_value
        return secret_value

    env_value = (os.getenv("GROQ_API_KEY") or "").strip()
    if env_value:
        st.session_state["groq_api_key"] = env_value
        return env_value

    st.session_state["groq_api_key"] = ""
    return ""


def has_groq_key() -> bool:
    return bool(resolve_groq_api_key())


def call_groq_chat(messages: list[dict[str, str]]) -> str:
    api_key = resolve_groq_api_key()
    if not api_key:
        return ""

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 400,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        return ""

    first_choice = choices[0]
    message = first_choice.get("message") if isinstance(first_choice, dict) else getattr(first_choice, "message", None)
    content = extract_message_text(message if isinstance(message, (dict, list, str)) else getattr(message, "content", message))
    return content.strip() if content else ""


def save_groq_key(value: str | None) -> None:
    cleaned = (value or "").strip()
    st.session_state["groq_api_key"] = cleaned
    return None


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


def extract_message_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = extract_message_text(item)
            if text:
                parts.append(text)
        return " ".join(parts).strip()
    if isinstance(value, dict):
        for key in ("text", "content"):
            if key in value:
                text = extract_message_text(value[key])
                if text:
                    return text
        return ""
    if hasattr(value, "content"):
        return extract_message_text(getattr(value, "content"))
    if hasattr(value, "text"):
        return extract_message_text(getattr(value, "text"))
    text = str(value).strip()
    return text


def format_groq_error(exc: Exception) -> str:
    message = str(exc).lower()
    if "401" in message or "api key" in message or "authentication" in message or "unauthorized" in message:
        return "Groq API 인증에 실패했습니다. 키를 다시 확인하고, 새로 입력한 뒤 다시 질문해 주세요."
    if "429" in message or "rate limit" in message or "too many requests" in message:
        return "Groq 요청 한도를 초과했습니다. 잠시 후 다시 시도해 주세요."
    if "404" in message or "model" in message or "not found" in message:
        return "현재 Groq 모델이 사용할 수 없습니다. 잠시 후 다시 시도하거나 다른 모델 사용 여부를 확인해 주세요."
    if "connection" in message or "timed out" in message or "network" in message:
        return "Groq 서버 연결에 실패했습니다. 잠시 후 다시 시도해 주세요."
    if "ascii" in message or "unicode" in message or "codec" in message:
        return "Groq 요청 인코딩 처리에 실패했습니다. 질문을 짧게 다시 말해주시거나 키를 다시 입력해 주세요."
    return f"현재 AI 응답 생성에 실패했습니다. 원인: {exc}"


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

    if not has_groq_key():
        return (
            "Groq API 키가 설정되지 않아 기본 안내 응답으로 동작 중입니다. "
            "Streamlit Secrets의 GROQ_API_KEY를 추가하면 실제 AI 응답을 받을 수 있습니다."
        )

    messages = [{"role": "system", "content": build_system_prompt()}]
    for item in st.session_state.get("messages", []):
        messages.append({"role": item["role"], "content": item["text"]})
    messages.append({"role": "user", "content": user_text})

    try:
        content = call_groq_chat(messages)
        return content or "답변을 생성하지 못했습니다. 다시 말씀해 주세요."
    except Exception as exc:
        return format_groq_error(exc)


def add_message(role: str, text: str) -> None:
    messages = st.session_state.setdefault("messages", [])
    messages.append({"role": role, "text": text, "time": time.strftime("%H:%M")})
    if len(messages) > 20:
        st.session_state["messages"] = messages[-20:]


def clean_tts_text(text: str) -> str:
    value = (text or "").strip()
    value = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", " ", value)
    value = value.replace("\n", " ")
    value = re.sub(r"[.!?,;:…\-_=+\[\]{}()/\\'\"<>|~`@#$%^&*]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def queue_voice_reply(message: str) -> None:
    text = clean_tts_text(message)
    if not text:
        st.session_state["last_tts_text"] = ""
        st.session_state["tts_last_played"] = ""
        st.session_state["voice_active"] = False
        return
    if text == st.session_state.get("tts_last_played", ""):
        return
    st.session_state["last_tts_text"] = text
    st.session_state["tts_last_played"] = text
    st.session_state["voice_active"] = True


def ensure_state() -> None:
    st.session_state.setdefault("messages", [
        {"role": "assistant", "text": "반가워요! 오늘은 무엇을 도와드릴까요?", "time": time.strftime("%H:%M")}
    ])
    st.session_state.setdefault("reminders", [])
    st.session_state.setdefault("last_tts_text", "")
    st.session_state.setdefault("tts_last_played", "")
    st.session_state.setdefault("tts_test_text", "테스트 음성이 정상적으로 재생되고 있습니다.")
    st.session_state.setdefault("last_seen_transcript", "")
    st.session_state.setdefault("wake_word", DEFAULT_WAKE_WORD)
    st.session_state.setdefault("manual_input", "")
    st.session_state.setdefault("groq_api_key", "")
    st.session_state.setdefault("conversation_started", False)
    st.session_state.setdefault("voice_active", False)

    if not st.session_state.get("groq_api_key"):
        st.session_state["groq_api_key"] = resolve_groq_api_key()


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


def stop_conversation(message: str = "대화가 종료되었습니다. 필요하면 다시 시작해 주세요.") -> None:
    st.session_state["conversation_started"] = False
    st.session_state["voice_active"] = False
    st.session_state["last_seen_transcript"] = ""
    st.session_state["last_tts_text"] = ""
    st.session_state["tts_last_played"] = ""
    st.session_state[APP_STATE_KEY] = {
        "status": "idle",
        "transcript": "",
        "wakeWord": "미감지",
        "lastCommand": "종료",
        "event_id": f"stop_{int(time.time() * 1000)}",
        "conversation_started": False,
    }


def process_voice_event(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        return

    transcript = (payload.get("transcript") or "").strip()
    wake_word = payload.get("wakeWord", "미감지")
    status = payload.get("status", "idle")
    last_command = payload.get("lastCommand", "대기")

    if not transcript:
        return

    if transcript == st.session_state.get("last_seen_transcript"):
        return

    st.session_state["last_seen_transcript"] = transcript

    if detect_stop_command(transcript):
        stop_message = "대화가 종료되었습니다. 필요하면 다시 시작해 주세요."
        add_message("assistant", stop_message)
        stop_conversation(stop_message)
        queue_voice_reply(stop_message)
        st.session_state["last_seen_transcript"] = ""
        return

    is_wake_word = detect_wake_word(transcript) or wake_word == "감지"
    if not st.session_state.get("conversation_started", False) and not is_wake_word:
        return

    if is_wake_word:
        st.session_state["conversation_started"] = True
        st.session_state["voice_active"] = True
        st.session_state[APP_STATE_KEY] = {
            "status": "listening",
            "transcript": transcript,
            "wakeWord": "감지",
            "lastCommand": "호출어",
            "event_id": payload.get("event_id", ""),
            "conversation_started": True,
        }

    add_message("user", transcript)
    user_text = transcript.replace(DEFAULT_WAKE_WORD, "").strip() if is_wake_word else transcript
    user_text = user_text.strip() or transcript

    reply = generate_reply(user_text)
    add_message("assistant", reply)
    queue_voice_reply(reply)
    st.session_state["wake_word"] = DEFAULT_WAKE_WORD

    st.session_state[APP_STATE_KEY] = {
        "status": status,
        "transcript": transcript,
        "wakeWord": wake_word,
        "lastCommand": last_command,
        "event_id": payload.get("event_id", ""),
        "conversation_started": st.session_state.get("conversation_started", False),
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
    component_state["conversation_started"] = st.session_state.get("conversation_started", False)
    tts_text = st.session_state.get("last_tts_text", "")

    def on_payload_change() -> None:
        widget_value = st.session_state.get(WIDGET_KEY, {})
        if isinstance(widget_value, dict):
            payload = widget_value.get("payload")
            if isinstance(payload, dict):
                existing = st.session_state.get(APP_STATE_KEY, {})
                if not isinstance(existing, dict) or payload != existing:
                    st.session_state[APP_STATE_KEY] = payload
                st.session_state["conversation_started"] = bool(payload.get("conversation_started", st.session_state.get("conversation_started", False)))
                if payload.get("tts_text") == "":
                    st.session_state["last_tts_text"] = ""

                # Only process a new browser transcript payload. Calling this on every
                # rerun replays the same recognition event and creates the loop that
                # freezes the app and breaks isolated TTS playback.
                if payload.get("transcript"):
                    process_voice_event(payload)

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
          const normalized = String(text).trim();
          if (!normalized) return;

          const runSpeak = () => {
            try {
              synth.cancel();
              if (synth.paused) synth.resume();
            } catch (err) {
              // ignored
            }

            const utterance = new SpeechSynthesisUtterance(normalized);
            utterance.lang = 'ko-KR';
            utterance.rate = 1.0;
            utterance.pitch = 1.1;
            utterance.volume = 1.0;

            try {
              const voices = synth.getVoices();
              const preferred = voices.find(v => /ko|kr/i.test(v.lang) && /(female|woman|여성|여자)/i.test(v.name))
                || voices.find(v => /ko|kr/i.test(v.lang));
              if (preferred) utterance.voice = preferred;
            } catch (err) {
              // ignored
            }

            utterance.onend = () => {
              window.__ai_speaker_last_tts = '';
            };
            utterance.onerror = () => {
              window.__ai_speaker_last_tts = '';
            };

            try {
              window.__ai_speaker_last_tts = normalized;
              synth.speak(utterance);
            } catch (err) {
              console.warn('TTS failed', err);
              window.__ai_speaker_last_tts = '';
            }
          };

          try {
            const currentVoices = synth.getVoices();
            if (currentVoices && currentVoices.length === 0) {
              window.speechSynthesis.onvoiceschanged = () => {
                const voices = synth.getVoices();
                if (voices && voices.length > 0) {
                  window.setTimeout(runSpeak, 150);
                }
              };
              window.setTimeout(runSpeak, 250);
            } else {
              window.setTimeout(runSpeak, 150);
            }
          } catch (err) {
            window.setTimeout(runSpeak, 150);
          }
        }

        function playQueuedTts() {
          const pending = window.__ai_speaker_pending_tts || '';
          if (!pending) return;
          window.__ai_speaker_pending_tts = '';
          speakText(pending);
        }

        function queueTtsPlayback(text) {
          if (!text) return;
          const cleanText = String(text).trim();
          if (!cleanText) return;

          window.__ai_speaker_pending_tts = cleanText;

          if (!window.__ai_speaker_tts_gate_initialized) {
            window.__ai_speaker_tts_gate_initialized = true;
            const activateSpeech = () => {
              window.__ai_speaker_tts_ready = true;
              playQueuedTts();
            };
            window.addEventListener('pointerdown', activateSpeech, { once: true, passive: true });
            window.addEventListener('keydown', activateSpeech, { once: true, passive: true });
            window.addEventListener('touchstart', activateSpeech, { once: true, passive: true });
          }

          if (window.__ai_speaker_tts_ready) {
            playQueuedTts();
          }
        }

        function getSpeechState() {
          if (!window.__ai_speaker_state) {
            window.__ai_speaker_state = { recognition: null, activeEventId: null };
          }
          return window.__ai_speaker_state;
        }

        export default function(component) {
          const { parentElement, data, setStateValue } = component;
          const root = parentElement.querySelector('#root');
          if (!root) return;

          const speechState = getSpeechState();

          const current = data && data.payload ? data.payload : {
            status: 'idle',
            transcript: '',
            wakeWord: '미감지',
            lastCommand: '대기',
            event_id: '',
            conversation_started: false,
            tts_text: '',
          };
          const conversationStarted = !!(data && data.conversation_started) || !!current.conversation_started;
          current.conversation_started = conversationStarted;
          const ttsText = current.tts_text || (data && data.tts_text) || '';

          const updateState = (nextPayload) => {
            const merged = {
              ...current,
              ...nextPayload,
              event_id: nextPayload && nextPayload.event_id ? nextPayload.event_id : current.event_id || makeEventId(),
            };
            const changed = (
              current.status !== merged.status ||
              current.transcript !== merged.transcript ||
              current.wakeWord !== merged.wakeWord ||
              current.lastCommand !== merged.lastCommand ||
              !!current.conversation_started !== !!merged.conversation_started ||
              current.event_id !== merged.event_id
            );
            if (!changed) return;
            current.status = merged.status;
            current.transcript = merged.transcript ?? current.transcript;
            current.wakeWord = merged.wakeWord ?? current.wakeWord;
            current.lastCommand = merged.lastCommand ?? current.lastCommand;
            current.conversation_started = merged.conversation_started ?? current.conversation_started;
            current.event_id = merged.event_id;
            setStateValue('payload', merged);
          };

          const stopListening = () => {
            const rec = speechState.recognition;
            if (rec && typeof rec.stop === 'function') {
              try {
                rec.stop();
              } catch (err) {
                // ignored
              }
            }
            speechState.recognition = null;
            speechState.activeEventId = null;
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

            const nextEventId = current.event_id || makeEventId();
            if (speechState.recognition && speechState.activeEventId === nextEventId) {
              return speechState.recognition;
            }

            stopListening();

            const recognition = new RecognitionCtor();
            speechState.recognition = recognition;
            speechState.activeEventId = nextEventId;
            recognition.lang = 'ko-KR';
            recognition.continuous = true;
            recognition.interimResults = true;

            recognition.onstart = () => updateState({ status: 'listening', lastCommand: '듣는 중', event_id: nextEventId });
            recognition.onresult = (event) => {
              const lastResult = event.results[event.results.length - 1];
              if (!lastResult || !lastResult.isFinal) return;
              const text = lastResult[0].transcript.trim();
              if (!text) return;
              const info = evaluateTranscript(text);
              if (info.stopDetected) {
                stopListening();
                const closedPayload = {
                  status: 'idle',
                  transcript: text,
                  wakeWord: '미감지',
                  lastCommand: '종료',
                  conversation_started: false,
                  event_id: nextEventId,
                };
                updateState(closedPayload);
                setStateValue('payload', closedPayload);
                setStateValue('conversation_started', false);
                return;
              }
              updateState({
                status: 'result_ready',
                transcript: text,
                wakeWord: info.wakeDetected ? '감지' : '미감지',
                lastCommand: '대화',
                event_id: nextEventId,
              });
            };
            recognition.onerror = () => updateState({ status: 'error', lastCommand: '음성 인식 오류', event_id: nextEventId });
            recognition.onend = () => {
              speechState.recognition = null;
              speechState.activeEventId = null;
              updateState({ status: 'idle', lastCommand: '대기', event_id: nextEventId });
            };

            try {
              recognition.start();
            } catch (err) {
              speechState.recognition = null;
              speechState.activeEventId = null;
              updateState({ status: 'idle', lastCommand: '재시도', event_id: nextEventId });
            }
            return recognition;
          };

          if (ttsText) {
            queueTtsPlayback(ttsText);
            const nextPayload = { ...current, tts_text: '' };
            setStateValue('payload', nextPayload);
          }

          if (!conversationStarted) {
            stopListening();
            const idlePayload = {
              ...current,
              status: 'idle',
              transcript: '',
              wakeWord: '미감지',
              lastCommand: '대기',
              conversation_started: false,
              event_id: current.event_id || 'idle_state',
            };
            const hasIdleDiff = (
              current.status !== idlePayload.status ||
              current.transcript !== idlePayload.transcript ||
              current.wakeWord !== idlePayload.wakeWord ||
              current.lastCommand !== idlePayload.lastCommand ||
              !!current.conversation_started !== !!idlePayload.conversation_started
            );
            if (hasIdleDiff) {
              setStateValue('payload', idlePayload);
            }
            current.status = idlePayload.status;
            current.transcript = idlePayload.transcript;
            current.wakeWord = idlePayload.wakeWord;
            current.lastCommand = idlePayload.lastCommand;
            current.conversation_started = idlePayload.conversation_started;
            current.event_id = idlePayload.event_id;
          } else if (!speechState.recognition || speechState.activeEventId !== (current.event_id || '')) {
            startListening();
          }

          render();
        }
        """,
        css="",
        isolate_styles=True,
    )

    component_state["tts_text"] = st.session_state.get("last_tts_text", "")
    result = component_factory(
        key=WIDGET_KEY,
        data={
            "payload": component_state,
            "conversation_started": st.session_state.get("conversation_started", False),
            "tts_text": st.session_state.get("last_tts_text", ""),
        },
        default={
            "payload": component_state,
        },
        on_payload_change=on_payload_change,
    )

    if result is not None and isinstance(result, dict):
        next_payload = result.get("payload")
        if isinstance(next_payload, dict):
            existing_payload = st.session_state.get(APP_STATE_KEY, {})
            if not isinstance(existing_payload, dict) or next_payload != existing_payload:
                st.session_state[APP_STATE_KEY] = next_payload
            if next_payload.get("tts_text") == "":
                st.session_state["last_tts_text"] = ""
        if "conversation_started" in result:
            st.session_state["conversation_started"] = bool(result["conversation_started"])

    # Important: do not call process_voice_event() here on every rerun.
    # Browser speech results are already delivered through the component payload,
    # and re-processing the same transcript here creates a rerender loop and
    # breaks the isolated TTS playback flow.


ensure_state()

st.title("🎧 헤이 친구 AI 스피커")
st.caption("음성 인식 + 대화 + 검색 + 일정 알림을 한 번에")

if not has_groq_key():
    st.warning("Groq API 키가 아직 없습니다. 설정에서 API 키를 입력해야 실제 AI 응답이 나옵니다.")

with st.sidebar:
    st.subheader("대화 제어")
    if st.button(
        "대화 시작" if not st.session_state.get("conversation_started", False) else "대화 종료",
        use_container_width=True,
    ):
        if st.session_state.get("conversation_started", False):
            stop_conversation("대화가 종료되었습니다. 필요하면 다시 시작해 주세요.")
            add_message("assistant", "대화가 종료되었습니다. 필요하면 다시 시작해 주세요.")
        else:
            st.session_state["conversation_started"] = True
            st.session_state["voice_active"] = True
            st.session_state["last_seen_transcript"] = ""
            add_message("assistant", "대화가 시작되었습니다. 무엇을 도와드릴까요?")
            st.session_state["last_tts_text"] = "대화가 시작되었습니다. 무엇을 도와드릴까요?"
            st.session_state[APP_STATE_KEY] = {
                "status": "idle",
                "transcript": "",
                "wakeWord": "미감지",
                "lastCommand": "대기",
                "event_id": f"start_{int(time.time() * 1000)}",
                "conversation_started": True,
            }

    render_sidebar_voice_status()
    render_component()
    st.caption("음성 인식 상태는 여기에서 확인합니다.")

    st.subheader("설정")
    if has_groq_key():
        st.success("Groq API 키가 설정되어 있습니다.")
    else:
        st.warning("Groq API 키를 입력해야 실제 AI 응답이 재생됩니다.")

    st.text_input(
        "GROQ API Key",
        key="groq_api_key",
        type="password",
        help="한 번 입력하면 현재 브라우저 세션 동안 유지됩니다. 설정을 변경하기 전까지 다시 입력할 필요가 없습니다.",
        placeholder="API 키를 입력하세요",
    )
    if st.session_state.get("groq_api_key"):
        st.caption("현재 입력된 API 키는 이 세션에서 유지됩니다.")
    st.caption("현재 세션에서 저장된 API 키는 설정을 바꾸기 전까지 유지됩니다.")
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

    st.text_input("음성 테스트 문구", key="tts_test_text")
    if st.button("음성 테스트 재생"):
        st.session_state["last_tts_text"] = st.session_state.get("tts_test_text", "테스트 음성이 정상적으로 재생되고 있습니다.")
        st.session_state["conversation_started"] = True

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
