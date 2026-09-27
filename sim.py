"""
Project 07 핵심 로직 (센서 시뮬레이터 + 자세 판별 알고리즘)
UI(app.py)와 분리해 두어 테스트하기 쉽게 만듦
"""
import numpy as np

FS = 20  # 샘플링 주파수 (Hz) = 50ms마다 측정
# 방석 네 모서리 센서 위치 (x: 왼쪽 -1 ~ 오른쪽 +1, y: 뒤 -1 ~ 앞 +1)
SENSORS = {"좌앞": (-1, 1), "우앞": (1, 1), "좌뒤": (-1, -1), "우뒤": (1, -1)}

# 자세별 목표 무게중심 (x, y)
POSTURES = {
    "바른 자세": (0.0, 0.0),
    "왼쪽으로 기울기": (-0.5, 0.0),
    "오른쪽으로 기울기": (0.5, 0.0),
    "앞으로 숙이기": (0.0, 0.5),
    "뒤로 기대기": (0.0, -0.5),
    "다리 꼬기(오른쪽)": (0.35, -0.35),
    "고쳐 앉기(움직임)": None,  # 빠르게 흔들리는 동작
    "자리 비움": "vacant",
}


# ---------------- [확장] 가상 센서 시뮬레이터 ----------------
def cog_to_forces(x, y, weight):
    """무게중심 (x, y)가 되도록 네 센서에 힘을 나눠줌.
    F_i = W/4 · (1 + x·x_i)(1 + y·y_i) → ΣF_i = W, ΣF_i·x_i / ΣF_i = x 가 성립"""
    return np.array([weight / 4 * (1 + x * sx) * (1 + y * sy) for sx, sy in SENSORS.values()])


def simulate(scenario, weight=60.0, body_bias=(0.1, 0.05), noise=0.03, drift=0.02, seed=0):
    """scenario: [(자세 이름, 지속 시간 s), ...] → 시간, 4채널 힘, 실제 자세 라벨"""
    rng = np.random.default_rng(seed)
    t_list, F_list, label_list = [], [], []
    cur = np.array(body_bias, dtype=float)
    t = 0.0
    for name, dur in scenario:
        target = POSTURES[name]
        for k in range(int(dur * FS)):
            if target == "vacant":
                F = np.abs(rng.normal(0, 0.5, 4))  # 빈 의자: 거의 0
            else:
                if target is None:  # 고쳐 앉기: 좌우로 크게 왔다 갔다
                    goal = np.array([0.6 * np.sin(2 * np.pi * 0.8 * k / FS), 0.3 * np.cos(2 * np.pi * 0.5 * k / FS)])
                else:
                    goal = np.array(target)
                goal = goal + body_bias
                cur += (goal - cur) * (1 / FS) / 0.4  # 사람 몸은 순간이동 못함 (시상수 0.4s)
                x, y = np.clip(cur, -0.95, 0.95)
                F = cog_to_forces(x, y, weight)
                F *= 1 + drift * (t / 60)  # FSR 필름 눌림에 따른 느린 드리프트
                F += rng.normal(0, noise * weight / 4, 4)
            t_list.append(t); F_list.append(np.clip(F, 0, None)); label_list.append(name)
            t += 1 / FS
    return np.array(t_list), np.array(F_list), label_list


# ---------------- 판별 알고리즘 (Step 1~3) ----------------
def compute_cog(F):
    """모멘트 평형: X = ΣF_i·x_i / ΣF_i"""
    xs = np.array([s[0] for s in SENSORS.values()]); ys = np.array([s[1] for s in SENSORS.values()])
    total = F.sum(axis=1)
    safe = np.where(total > 0, total, 1)
    return (F @ xs) / safe, (F @ ys) / safe, total


def classify(dx, dy):
    """[확장] 쏠린 방향으로 불량 자세 유형 추정"""
    ang = np.degrees(np.arctan2(dy, dx)) % 360
    if 25 <= ang < 65 or 115 <= ang < 155 or 205 <= ang < 245 or 295 <= ang < 335:
        return "다리 꼬기 의심(대각선 쏠림)"
    if 65 <= ang < 115: return "앞으로 숙임"
    if 245 <= ang < 295: return "뒤로 기댐"
    return "오른쪽 쏠림" if dx > 0 else "왼쪽 쏠림"


ADVICE = {
    "다리 꼬기 의심(대각선 쏠림)": "다리를 풀고 양발을 바닥에 붙여보세요.",
    "앞으로 숙임": "허리를 펴고 등받이에 등을 대보세요.",
    "뒤로 기댐": "엉덩이를 의자 안쪽으로 당겨 앉아보세요.",
    "오른쪽 쏠림": "왼쪽 엉덩이에도 체중을 실어보세요.",
    "왼쪽 쏠림": "오른쪽 엉덩이에도 체중을 실어보세요.",
}


def analyze(t, F, vacant_th=5.0, tare_sec=5.0, alpha=0.2, radius=0.3,
            v_th=0.25, hold_sec=15.0):
    X, Y, total = compute_cog(F)
    n = len(t)
    out = {k: np.zeros(n) for k in ["dx", "dy", "vx", "vy", "timer", "pwm", "bad", "moving", "seated"]}
    out["type"] = [""] * n
    zero = None; seat_start = None
    ex = ey = 0.0; pvx = pvy = 0.0; timer = 0.0; alarm_t = None
    prev_ex = prev_ey = None
    for i in range(n):
        seated = total[i] >= vacant_th
        out["seated"][i] = seated
        if not seated:  # 빈 의자 → 모두 초기화, 다음 착석 때 다시 오토 테어
            zero = None; seat_start = None; timer = 0; alarm_t = None; prev_ex = None
            continue
        if seat_start is None:
            seat_start = t[i]; buf = []
        if zero is None:  # Step 1: 착석 후 5초 평균 = 이 사람의 기준 원점
            buf.append((X[i], Y[i]))
            if t[i] - seat_start >= tare_sec:
                zero = np.mean(buf, axis=0); ex, ey = 0.0, 0.0
            continue
        # Step 1: 기준 대비 상대 변위 + EMA 필터
        ex = alpha * (X[i] - zero[0]) + (1 - alpha) * ex
        ey = alpha * (Y[i] - zero[1]) + (1 - alpha) * ey
        # Step 2: 속도(변화율) = 위상 평면의 세로축
        if prev_ex is not None:
            vx = (ex - prev_ex) * FS; vy = (ey - prev_ey) * FS
            pvx = 0.3 * vx + 0.7 * pvx; pvy = 0.3 * vy + 0.7 * pvy  # 속도 LPF
        prev_ex, prev_ey = ex, ey
        speed = np.hypot(pvx, pvy)
        bad = (ex / radius) ** 2 + (ey / radius) ** 2 > 1
        moving = speed > v_th
        # Step 3: 상태 머신 — 멈춘 채로 벗어나 있을 때만 타이머 증가
        if not bad:
            timer = 0; alarm_t = None
        elif not moving:
            timer += 1 / FS
        # 움직이는 중이면 타이머 일시정지 (고쳐 앉기 오경보 방지)
        pwm = 0
        if timer >= hold_sec:
            if alarm_t is None: alarm_t = t[i]
            pwm = min(200, 50 + 150 * (t[i] - alarm_t) / 3)  # 3초간 50→200 부드럽게
        out["dx"][i], out["dy"][i], out["vx"][i], out["vy"][i] = ex, ey, pvx, pvy
        out["timer"][i], out["pwm"][i], out["bad"][i], out["moving"][i] = timer, pwm, bad, moving
        if bad: out["type"][i] = classify(ex, ey)
    out["X"], out["Y"], out["total"] = X, Y, total
    return out


def naive_alarms(X, Y, total, vacant_th=5.0, radius=0.3):
    """비교용 1차 알고리즘: 보정·속도 판별 없이 원점 이탈 즉시 경보"""
    bad = ((X / radius) ** 2 + (Y / radius) ** 2 > 1) & (total >= vacant_th)
    return int(np.sum(np.diff(bad.astype(int)) == 1) + bad[0])


def count_alarms(pwm):
    on = pwm > 0
    return int(np.sum(np.diff(on.astype(int)) == 1) + on[0])
