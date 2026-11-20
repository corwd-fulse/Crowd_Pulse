"""crowdqueue 병목 실험에 LSTM을 더해 직전값·대기열 모형·LightGBM과 같은 표본에서 비교(문 앞 5·10초 뒤 밀도).
입력 = 최근 10초 × LightGBM과 같은 특징 12개(학습 회차 통계로 표준화, 결측 0), 출력 = 지금 밀도 + 변화량. 손실 L1.
평가 = 회차 하나씩 빼고 학습(24겹), 시드 3개 평균±표준편차(명세서 MDL-07).

python -m forecast.bottleneck.crowdqueue          (먼저: LightGBM 결과 pred_bottleneck.csv 생성)
python -m forecast.bottleneck.crowdqueue_lstm
"""
import numpy as np, pandas as pd, torch, torch.nn as nn

from forecast.bottleneck.crowdqueue import DATA, HORIZONS, X, build_dataset

SEQ, SEEDS, EPOCHS = 10, (0, 1, 2), 80


def sequences(df):
    """행마다 같은 회차의 t-9..t 특징(앞부분이 모자라면 0으로 채움). df는 (run, h, sec) 정렬 상태."""
    F = df[X].fillna(0).to_numpy(np.float32)
    out = np.zeros((len(df), SEQ, len(X)), np.float32)
    start = 0
    for _, n in df.groupby(["run", "h"], sort=False).size().items():
        for i in range(n):
            lo = max(0, i - SEQ + 1)
            out[start + i, SEQ - (i - lo + 1):] = F[start + lo:start + i + 1]
        start += n
    return out


class Net(nn.Module):
    def __init__(s, k):
        super().__init__()
        s.lstm = nn.LSTM(k, 32, batch_first=True)
        s.head = nn.Linear(32, 1)

    def forward(s, x, now):
        o, _ = s.lstm(x)
        return now + s.head(o[:, -1]).squeeze(-1)


def fit_predict(xtr, ntr, ytr, xte, nte, seed):
    torch.manual_seed(seed)
    mu, sd = xtr.reshape(-1, xtr.shape[-1]).mean(0), xtr.reshape(-1, xtr.shape[-1]).std(0) + 1e-6
    t = lambda a: torch.tensor(a)
    xtr, xte = t((xtr - mu) / sd), t((xte - mu) / sd)
    net = Net(xtr.shape[-1])
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    ntr_, ytr_ = t(ntr), t(ytr)
    for _ in range(EPOCHS):  # 조기 종료용 검증 회차를 따로 두기엔 회차가 적어 고정 에폭
        for b in torch.randperm(len(xtr)).split(64):
            loss = (net(xtr[b], ntr_[b]) - ytr_[b]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return net(xte, t(nte)).numpy()


def main():
    df = build_dataset().sort_values(["run", "h", "sec"]).reset_index(drop=True)
    lg = pd.read_csv(DATA / "pred_bottleneck.csv")[["run", "h", "sec", "lgb"]]
    df = df.merge(lg, on=["run", "h", "sec"], how="left", validate="1:1")
    assert df.lgb.notna().all(), "LightGBM 결과와 표본이 다르다: crowdqueue를 먼저 다시 실행"
    seq = sequences(df)
    now, y = df.dens.to_numpy(np.float32), df.y_dens.to_numpy(np.float32)
    for s in SEEDS:
        df[f"lstm_s{s}"] = np.nan
        for h in HORIZONS:
            for r in df.run.unique():
                tr = ((df.h == h) & (df.run != r)).to_numpy()
                te = ((df.h == h) & (df.run == r)).to_numpy()
                df.loc[te, f"lstm_s{s}"] = fit_predict(seq[tr], now[tr], y[tr], seq[te], now[te], s)
    for h in HORIZONS:
        s = df[df.h == h]
        hot = s.dens >= 2
        res = {}
        for m, name in (("pers", "직전값"), ("queue", "대기열 모형"), ("lgb", "LightGBM")):
            res[name] = {"전체": f"{(s[m] - s.y_dens).abs().mean():.3f}", "붐빌 때": f"{(s[m] - s.y_dens)[hot].abs().mean():.3f}"}
        for key, mask in (("전체", slice(None)), ("붐빌 때", hot)):
            v = [(s[f"lstm_s{k}"] - s.y_dens)[mask].abs().mean() for k in SEEDS]
            res.setdefault("LSTM (시드 3개)", {})[key] = f"{np.mean(v):.3f} ± {np.std(v):.3f}"
        print(f"\n[{h}초 뒤 문 앞 밀도] MAE(명/㎡) — 전체 {len(s)}초 / 붐빌 때(지금 2명/㎡ 이상) {hot.sum()}초")
        print(pd.DataFrame(res).T.to_string())
    df.to_csv(DATA / "pred_bottleneck_lstm.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
