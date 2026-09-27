"""
Project 07. 가상 센서 시뮬레이터 기반 스마트 자세 교정 방석
실행: streamlit run app.py
"""
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sim import (ADVICE, POSTURES, SENSORS, analyze, count_alarms, naive_alarms, simulate)

st.set_page_config(page_title="스마트 자세 교정 방석 시뮬레이터", layout="wide")
st.title("스마트 자세 교정 방석 – 가상 센서 시뮬레이터")
st.caption("실제 압력 센서 없이, 자세 시나리오로 4채널 센서 값을 만들어 판별 알고리즘을 검증합니다.")

# ---------- 사이드바: 사용자·센서 조건 ----------
with st.sidebar:
    st.header("착석자 & 센서 조건")
    weight = st.slider("체중 (kg)", 40, 100, 60)
    bx = st.slider("체형 좌우 편향", -0.2, 0.2, 0.1, 0.01)
    by = st.slider("체형 앞뒤 편향", -0.2, 0.2, 0.05, 0.01)
    noise = st.slider("센서 노이즈", 0.0, 0.1, 0.03, 0.01)
    drift = st.slider("센서 드리프트 (분당)", 0.0, 0.1, 0.02, 0.01)
    st.header("판별 기준")
    radius = st.slider("안전 반경", 0.1, 0.5, 0.3, 0.05)
    hold = st.slider("경보까지 유지 시간 (초)", 5, 30, 15)

# ---------- 시나리오 편집 ----------
st.subheader("① 자세 시나리오")
default = pd.DataFrame({
    "자세": ["바른 자세", "고쳐 앉기(움직임)", "바른 자세", "다리 꼬기(오른쪽)",
            "바른 자세", "앞으로 숙이기", "자리 비움", "바른 자세"],
    "시간(초)": [20, 10, 15, 25, 10, 20, 5, 15],
})
scenario_df = st.data_editor(
    default, num_rows="dynamic", use_container_width=True,
    column_config={"자세": st.column_config.SelectboxColumn(options=list(POSTURES)),
                   "시간(초)": st.column_config.NumberColumn(min_value=1, max_value=120)})

if st.button("시뮬레이션 실행", type="primary"):
    scenario = [(r["자세"], float(r["시간(초)"])) for _, r in scenario_df.dropna().iterrows()]
    t, F, labels = simulate(scenario, weight, (bx, by), noise, drift)
    o = analyze(t, F, radius=radius, hold_sec=hold)

    seated = o["seated"].astype(bool)
    good = seated & (o["bad"] == 0)
    score = good.sum() / max(seated.sum(), 1) * 100
    types = Counter(x for x in o["type"] if x)

    st.subheader("② 결과 요약")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("자세 점수", f"{score:.0f} %")
    c2.metric("개선 알고리즘 경보", f"{count_alarms(o['pwm'])} 회")
    c3.metric("단순 알고리즘 경보", f"{naive_alarms(o['X'], o['Y'], o['total'], radius=radius)} 회",
              help="보정·속도 판별 없이 벗어나는 즉시 울리는 1차 방식")
    top = types.most_common(1)[0][0] if types else "없음"
    c4.metric("가장 많은 불량 유형", top)
    if top in ADVICE:
        st.info(f"맞춤 조언: {ADVICE[top]}")

    # ---------- 방석 위 궤적 + 위상 평면 ----------
    st.subheader("③ 무게중심 궤적과 위상 평면")
    c1, c2 = st.columns(2)
    th = np.linspace(0, 2 * np.pi, 100)
    cushion = go.Figure()
    cushion.add_shape(type="rect", x0=-1, y0=-1, x1=1, y1=1, line=dict(color="gray"))
    cushion.add_trace(go.Scatter(x=radius * np.cos(th), y=radius * np.sin(th), fill="toself",
                                 fillcolor="rgba(0,180,0,0.15)", line=dict(color="green"), name="안전 구역"))
    m = seated & (o["timer"] >= 0)
    cushion.add_trace(go.Scatter(x=o["dx"][m], y=o["dy"][m], mode="lines", name="무게중심 궤적",
                                 line=dict(color="crimson", width=1)))
    for nm, (sx, sy) in SENSORS.items():
        cushion.add_annotation(x=sx * 0.92, y=sy * 0.92, text=nm, showarrow=False)
    cushion.update_layout(title="방석 위 무게중심 (기준 원점 보정 후)",
                          xaxis=dict(range=[-1.1, 1.1], title="좌 ← X → 우"),
                          yaxis=dict(range=[-1.1, 1.1], title="뒤 ← Y → 앞", scaleanchor="x"))
    c1.plotly_chart(cushion, use_container_width=True)

    state = np.where(o["moving"] == 1, "움직이는 중",
                     np.where(o["bad"] == 1, "멈춘 채 벗어남(불량)", "정상"))
    phase = go.Figure()
    for s, col in [("정상", "green"), ("움직이는 중", "orange"), ("멈춘 채 벗어남(불량)", "red")]:
        k = seated & (state == s)
        phase.add_trace(go.Scatter(x=o["dx"][k], y=o["vx"][k], mode="markers", name=s,
                                   marker=dict(color=col, size=4)))
    phase.add_vline(x=radius, line_dash="dash"); phase.add_vline(x=-radius, line_dash="dash")
    phase.update_layout(title="위상 평면 (가로: 위치 X, 세로: 속도 dX/dt)",
                        xaxis_title="좌우 위치 X", yaxis_title="좌우 속도 dX/dt")
    c2.plotly_chart(phase, use_container_width=True)

    # ---------- 시간 그래프 ----------
    st.subheader("④ 시간에 따른 변화")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("무게중심 위치", "불량 자세 타이머 (초)", "진동 모터 PWM"))
    fig.add_trace(go.Scatter(x=t, y=o["dx"], name="X (좌우)"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=o["dy"], name="Y (앞뒤)"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=o["timer"], name="타이머", line=dict(color="purple")), 2, 1)
    fig.add_hline(y=hold, line_dash="dash", row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=o["pwm"], name="PWM", line=dict(color="red")), 3, 1)
    # 실제 자세 구간 표시
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            fig.add_vrect(x0=t[start], x1=t[i - 1], annotation_text=labels[start],
                          annotation_position="top left", opacity=0.05, line_width=0, row=1, col=1)
            start = i
    fig.update_layout(height=700, xaxis3_title="시간 (초)")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("원본 센서 값 보기 (처음 100개)"):
        st.dataframe(pd.DataFrame(F[:100].round(2), columns=list(SENSORS)).assign(시간=t[:100].round(2)))
