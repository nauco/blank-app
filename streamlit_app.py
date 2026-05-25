"""
Anthropic API 토큰 사용량 대시보드
=====================================
실행 방법:
  python3 -m pip install streamlit anthropic pandas plotly cryptography
  streamlit run streamlit_app.py

보안 구조:
  - API Key는 Fernet 대칭 암호화 후 로컬 파일에만 저장
  - 암호화 키(master.key)는 최초 실행 시 자동 생성, 절대 공유 금지
  - usage_data.json에는 API Key가 절대 저장되지 않음
  - 화면에서도 API Key는 마스킹 처리
"""

import streamlit as st
import anthropic
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
from cryptography.fernet import Fernet

# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────
KEY_FILE    = "master.key"       # Fernet 암호화 키 (공유 금지)
USERS_FILE  = "users.json"       # 암호화된 API Key 저장
USAGE_FILE  = "usage_data.json"  # 토큰 사용 로그 (API Key 없음)

# ─────────────────────────────────────────────
# 암호화 유틸
# ─────────────────────────────────────────────
def get_fernet() -> Fernet:
    if not Path(KEY_FILE).exists():
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
        os.chmod(KEY_FILE, 0o600)
    with open(KEY_FILE, "rb") as f:
        return Fernet(f.read())

def encrypt(text: str) -> str:
    return get_fernet().encrypt(text.encode()).decode()

def decrypt(token: str) -> str:
    return get_fernet().decrypt(token.encode()).decode()

def mask_key(key: str) -> str:
    if len(key) <= 12:
        return "****"
    return key[:10] + "..." + key[-4:]

# ─────────────────────────────────────────────
# 사용자 데이터 (암호화된 API Key)
# ─────────────────────────────────────────────
def load_users() -> dict:
    if Path(USERS_FILE).exists():
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_users(users: dict):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    os.chmod(USERS_FILE, 0o600)

def register_user(username: str, api_key: str):
    users = load_users()
    users[username] = {
        "api_key_enc": encrypt(api_key),
        "registered_at": datetime.now().isoformat(),
    }
    save_users(users)

def get_api_key(username: str):
    users = load_users()
    if username in users:
        return decrypt(users[username]["api_key_enc"])
    return None

# ─────────────────────────────────────────────
# 사용량 로그 (API Key 저장 안 함)
# ─────────────────────────────────────────────
def load_usage() -> dict:
    if Path(USAGE_FILE).exists():
        with open(USAGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"logs": []}

def save_usage(data: dict):
    with open(USAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def append_log(username, model, input_tokens, output_tokens, preview):
    data = load_usage()
    data["logs"].append({
        "user":          username,
        "timestamp":     datetime.now().isoformat(),
        "model":         model,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "total_tokens":  input_tokens + output_tokens,
        "preview":       preview,
    })
    save_usage(data)

# ─────────────────────────────────────────────
# 프록시 API 호출
# ─────────────────────────────────────────────
def proxy_call(username, model, system_prompt, user_message, max_tokens=1024):
    api_key = get_api_key(username)
    if not api_key:
        raise ValueError("API Key가 등록되지 않은 사용자입니다.")

    client = anthropic.Anthropic(api_key=api_key)
    kwargs = {
        "model":      model,
        "max_tokens": max_tokens,
        "messages":   [{"role": "user", "content": user_message}],
    }
    if system_prompt.strip():
        kwargs["system"] = system_prompt

    response      = client.messages.create(**kwargs)
    input_tokens  = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    reply         = response.content[0].text
    preview       = user_message[:80] + ("..." if len(user_message) > 80 else "")

    append_log(username, model, input_tokens, output_tokens, preview)
    return reply, input_tokens, output_tokens

# ─────────────────────────────────────────────
# 통계 헬퍼
# ─────────────────────────────────────────────
def user_stats(logs: list, username: str) -> dict:
    ul = [l for l in logs if l["user"] == username]
    return {
        "calls":  len(ul),
        "input":  sum(l["input_tokens"]  for l in ul),
        "output": sum(l["output_tokens"] for l in ul),
        "total":  sum(l["total_tokens"]  for l in ul),
        "logs":   ul,
    }

# ─────────────────────────────────────────────
# Streamlit 앱
# ─────────────────────────────────────────────
st.set_page_config(page_title="Token Dashboard", page_icon="📊", layout="wide")

if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── 사이드바 ─────────────────────────────────
with st.sidebar:
    st.title("🔐 로그인")
    users = load_users()

    if users:
        login_name = st.selectbox("등록된 사용자", ["-- 선택 --"] + list(users.keys()))
        if st.button("로그인", use_container_width=True):
            if login_name != "-- 선택 --":
                st.session_state.current_user = login_name
                st.session_state.chat_history = []
                st.rerun()
    else:
        st.info("등록된 사용자가 없습니다.\n'API Key 등록' 탭에서 먼저 등록해주세요.")

    if st.session_state.current_user:
        st.success(f"✅ {st.session_state.current_user} 로그인 중")
        if st.button("로그아웃", use_container_width=True):
            st.session_state.current_user = None
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.caption("📁 로컬 저장 파일")
    st.caption(f"- `{KEY_FILE}` : 암호화 마스터키")
    st.caption(f"- `{USERS_FILE}` : 암호화된 API Key")
    st.caption(f"- `{USAGE_FILE}` : 사용량 로그")
    st.caption("⚠️ master.key는 절대 공유하지 마세요!")

# ── 탭 ───────────────────────────────────────
tab_dashboard, tab_register, tab_chat = st.tabs([
    "📊 대시보드",
    "🔑 API Key 등록",
    "💬 채팅",
])

# ══════════════════════════════════════════════
# TAB 1: 대시보드
# ══════════════════════════════════════════════
with tab_dashboard:
    current = st.session_state.current_user
    usage   = load_usage()
    logs    = usage["logs"]

    if not current:
        st.info("👈 사이드바에서 로그인하세요.")
    elif not logs:
        st.info("아직 사용 기록이 없습니다. 채팅 탭에서 먼저 사용해보세요.")
    else:
        st.header(f"안녕하세요, {current}님 👋")

        # 내 사용량
        st.subheader("📌 내 사용량")
        my = user_stats(logs, current)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 호출",   f"{my['calls']:,}회")
        c2.metric("입력 토큰", f"{my['input']:,}")
        c3.metric("출력 토큰", f"{my['output']:,}")
        c4.metric("총 토큰",   f"{my['total']:,}")

        if my["logs"]:
            df_my = pd.DataFrame(my["logs"])
            df_my["timestamp"] = pd.to_datetime(df_my["timestamp"])
            df_my["시간"] = df_my["timestamp"].dt.strftime("%m/%d %H:%M")

            fig_my = go.Figure()
            fig_my.add_bar(x=df_my["시간"], y=df_my["input_tokens"],  name="입력", marker_color="#4F86F7")
            fig_my.add_bar(x=df_my["시간"], y=df_my["output_tokens"], name="출력", marker_color="#F76F4F")
            fig_my.update_layout(
                barmode="stack", title="호출별 토큰 사용량",
                height=280, margin=dict(t=40, b=10),
                xaxis_title="", yaxis_title="토큰"
            )
            st.plotly_chart(fig_my, use_container_width=True)

            disp = df_my[["시간","model","input_tokens","output_tokens","total_tokens","preview"]].copy()
            disp.columns = ["시간","모델","입력","출력","합계","메시지 미리보기"]
            st.dataframe(disp[::-1], use_container_width=True, hide_index=True)

        st.divider()

        # 팀 전체 요약
        st.subheader("👥 팀 전체 현황")
        df_all = pd.DataFrame(logs)

        team_summary = (
            df_all.groupby("user")
            .agg(
                호출수=("total_tokens", "count"),
                입력토큰=("input_tokens", "sum"),
                출력토큰=("output_tokens", "sum"),
                총토큰=("total_tokens", "sum"),
            )
            .reset_index()
            .rename(columns={"user": "사용자"})
            .sort_values("총토큰", ascending=False)
        )

        col_a, col_b = st.columns(2)
        with col_a:
            fig_pie = px.pie(
                team_summary, names="사용자", values="총토큰",
                title="사용자별 토큰 비중", hole=0.4,
            )
            fig_pie.update_layout(height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            fig_bar = px.bar(
                team_summary, x="사용자", y=["입력토큰", "출력토큰"],
                title="사용자별 토큰 분류", barmode="stack",
                color_discrete_map={"입력토큰": "#4F86F7", "출력토큰": "#F76F4F"},
            )
            fig_bar.update_layout(height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(team_summary, use_container_width=True, hide_index=True)

        csv = df_all.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 전체 로그 CSV 다운로드", data=csv,
            file_name=f"token_usage_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# ══════════════════════════════════════════════
# TAB 2: API Key 등록
# ══════════════════════════════════════════════
with tab_register:
    st.header("🔑 API Key 등록")

    st.info(
        "입력한 API Key는 **Fernet 대칭 암호화** 후 로컬 파일에만 저장됩니다.\n\n"
        "- 네트워크로 전송되지 않습니다.\n"
        "- `master.key` 파일 없이는 복호화 불가합니다.\n"
        "- 화면에 다시 표시되지 않습니다.",
        icon="🔒"
    )

    with st.form("register_form"):
        new_name    = st.text_input("사용자 이름", placeholder="예: 홍길동")
        new_api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
        submitted   = st.form_submit_button("등록하기", use_container_width=True)

        if submitted:
            if not new_name.strip():
                st.error("사용자 이름을 입력해주세요.")
            elif not new_api_key.startswith("sk-ant-"):
                st.error("올바른 Anthropic API Key 형식이 아닙니다. (sk-ant-... 로 시작해야 함)")
            else:
                with st.spinner("API Key 유효성 확인 중..."):
                    try:
                        test_client = anthropic.Anthropic(api_key=new_api_key)
                        test_client.messages.create(
                            model="claude-haiku-4-5-20251001",
                            max_tokens=10,
                            messages=[{"role": "user", "content": "hi"}],
                        )
                        register_user(new_name.strip(), new_api_key)
                        st.success(f"✅ **{new_name}** 님의 API Key가 암호화되어 저장되었습니다!")
                        st.caption(f"저장된 키: `{mask_key(new_api_key)}`")
                    except anthropic.AuthenticationError:
                        st.error("❌ API Key 인증 실패: 키를 다시 확인해주세요.")
                    except Exception as e:
                        st.error(f"❌ 오류 발생: {e}")

    st.divider()
    st.subheader("등록된 사용자 목록")
    users_now = load_users()
    if users_now:
        rows = []
        for uname, udata in users_now.items():
            rows.append({
                "사용자":  uname,
                "API Key": mask_key(decrypt(udata["api_key_enc"])),
                "등록일시": udata.get("registered_at", "")[:19].replace("T", " "),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with st.expander("🗑️ 사용자 삭제"):
            del_name = st.selectbox("삭제할 사용자", list(users_now.keys()), key="del_user")
            if st.button("삭제", type="primary"):
                users_now.pop(del_name, None)
                save_users(users_now)
                st.success(f"{del_name} 삭제 완료")
                st.rerun()
    else:
        st.caption("등록된 사용자가 없습니다.")

# ══════════════════════════════════════════════
# TAB 3: 채팅
# ══════════════════════════════════════════════
with tab_chat:
    st.header("💬 채팅")
    current = st.session_state.current_user

    if not current:
        st.warning("👈 사이드바에서 먼저 로그인하세요.")
    else:
        st.caption(f"**{current}** 님으로 채팅 중 — 토큰 사용량이 자동 기록됩니다.")

        col1, col2 = st.columns([1, 3])
        with col1:
            model = st.selectbox("모델", [
                "claude-sonnet-4-20250514",
                "claude-haiku-4-5-20251001",
                "claude-opus-4-20250514",
            ])
            max_tokens    = st.slider("최대 출력 토큰", 256, 4096, 1024, 256)
            system_prompt = st.text_area("시스템 프롬프트", height=100,
                                         placeholder="당신은 친절한 어시스턴트입니다.")
            if st.button("🗑️ 대화 초기화", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        with col2:
            chat_box = st.container(height=430)
            with chat_box:
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])
                        if msg["role"] == "assistant" and "tokens" in msg:
                            t = msg["tokens"]
                            st.caption(f"입력 {t['input']:,} | 출력 {t['output']:,} tokens")

            user_input = st.chat_input("메시지를 입력하세요...")

            if user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with st.spinner("응답 생성 중..."):
                    try:
                        reply, inp, out = proxy_call(
                            current, model, system_prompt, user_input, max_tokens
                        )
                        st.session_state.chat_history.append({
                            "role":    "assistant",
                            "content": reply,
                            "tokens":  {"input": inp, "output": out},
                        })
                        st.rerun()
                    except anthropic.AuthenticationError:
                        st.error("❌ API Key 인증 실패. 'API Key 등록' 탭에서 키를 다시 등록해주세요.")
                    except ValueError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"❌ 오류: {e}")
