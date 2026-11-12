"""재학습 없이 저장된 모델(lgbm_1h.txt, lstm_1h.pt)로 10월 테스트를 처음부터 다시 추론해 보고 수치와 비교하고,
미래값을 쓰지 않는지(10/15 이후 값을 0으로 바꿔도 이전 시각 예측이 같은지) 확인한다.
사용: python verify_saved.py [lgbm 모델 파일(기본 lgbm_1h.txt)]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, torch, torch.nn as nn
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I 모드에서도 features_1h를 찾도록
from features_1h import ROOT, TEST_START, HOL, load_features, load_wide

F = Path(__file__).resolve().parent
LGBM = F / (sys.argv[1] if len(sys.argv) > 1 else "lgbm_1h.txt")
CUT = pd.Timestamp("2026-10-15")
L = 168
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- LSTM: train_lstm.py의 Net/batch와 같은 구조(복사). 구조가 다르면 load_state_dict(strict)에서 바로 실패 ---
class Net(nn.Module):
    def __init__(s, n, h=96):
        super().__init__()
        s.emb = nn.Embedding(n, 16)
        s.lstm = nn.LSTM(8, h, num_layers=2, batch_first=True, dropout=0.1)
        s.head = nn.Sequential(nn.Linear(h + 16 + 7, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(s, seq, tgt, j, last):
        o, _ = s.lstm(seq)
        return last + s.head(torch.cat([o[:, -1], s.emb(j), tgt], -1)).squeeze(-1)


def lstm_predict(w, scale, ts, js):
    """w: 원본 또는 교란된 wide. ts,js: 예측할 (시각 위치, 동 위치). 반환: 실제 인구 단위 예측."""
    r = (w / scale).to_numpy(np.float32)
    hol = pd.Series(w.index.date).isin(HOL).astype(np.float32).to_numpy()
    hour, dow = w.index.hour.to_numpy(), w.index.dayofweek.to_numpy()
    tf = np.stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), np.sin(2 * np.pi * dow / 7),
                   np.cos(2 * np.pi * dow / 7), hol, np.roll(hol, -24), np.roll(hol, 24)], 1).astype(np.float32)
    R, TF, off = torch.tensor(r, device=dev), torch.tensor(tf, device=dev), torch.arange(-L, 0, device=dev)
    net = Net(r.shape[1]).to(dev)
    net.load_state_dict(torch.load(F / "lstm_1h.pt"))
    net.eval(); out = []
    t_all, j_all = torch.tensor(ts, device=dev), torch.tensor(js, device=dev)
    with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
        for i in range(0, len(t_all), 4096):
            t, j = t_all[i:i + 4096], j_all[i:i + 4096]
            idx = t[:, None] + off
            assert (idx < t[:, None]).all()
            seq = torch.cat([R[idx, j[:, None]].unsqueeze(-1), TF[idx]], -1)
            out.append(net(seq, TF[t], j, R[t - 1, j]).float().cpu())
    return torch.cat(out).numpy() * scale.to_numpy()[js]


def metrics(y, p):
    e = p - y
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), sMAPE=(2 * np.abs(e) / (np.abs(y) + np.abs(p))).mean() * 100)


w = load_wide()
w.columns = w.columns.astype(str)
d, cols, scale, (_, _, te) = load_features(w)
booster = lgb.Booster(model_file=str(LGBM))
sc = scale.reindex(te.adm_cd.astype(str)).to_numpy()
te = te.assign(adm_cd=te.adm_cd.astype(str), actual=te.y.to_numpy() * sc)
te["lightgbm"] = booster.predict(te[cols]) * sc
tpos = {t: i for i, t in enumerate(w.index)}
jpos = {c: i for i, c in enumerate(w.columns)}
ts, js = te.datetime.map(tpos).to_numpy(), te.adm_cd.map(jpos).to_numpy()
te["lstm"] = lstm_predict(w, scale, ts, js)

rep = pd.read_parquet(F / "pred_test_1h.parquet")
rep["adm_cd"] = rep.adm_cd.astype(str)
m = te.merge(rep, on=["datetime", "adm_cd"], suffixes=("", "_rep"))
assert len(m) == len(te) == 317_688, (len(m), len(te))
rows = {}
for name, col in (("LightGBM", "lightgbm"), ("LSTM", "lstm")):
    now, old = metrics(m.actual.to_numpy(), m[col].to_numpy()), metrics(m.actual_rep.to_numpy(), m[col + "_rep"].to_numpy())
    rows[f"{name} 재추론"] = now; rows[f"{name} 보고값"] = old
    print(f"{name}: 보고된 예측과 최대 절대 차이 {np.abs(m[col] - m[col + '_rep']).max():.3f}명")
print("\n[재추론 vs 보고값, 테스트 10월]"); print(pd.DataFrame(rows).T.round(2).to_string())

# --- 미래값 교란 테스트: 10/15 이후 값을 0으로 바꾼 데이터로 다시 예측 ---
w2 = w.copy()
w2.loc[w2.index >= CUT] = 0.0
d2, cols2, scale2, (_, _, te2) = load_features(w2)
assert np.allclose(scale2, scale)
te2 = te2.assign(adm_cd=te2.adm_cd.astype(str))
te2["lightgbm2"] = booster.predict(te2[cols2]) * scale2.reindex(te2.adm_cd).to_numpy()
smp = m.sample(40_000, random_state=0)  # LSTM은 표본으로
ts2, js2 = smp.datetime.map(tpos).to_numpy(), smp.adm_cd.map(jpos).to_numpy()
smp = smp.assign(lstm2=lstm_predict(w2, scale2, ts2, js2))
c = m.merge(te2[["datetime", "adm_cd", "lightgbm2"]], on=["datetime", "adm_cd"])
c = c.merge(smp[["datetime", "adm_cd", "lstm2"]], on=["datetime", "adm_cd"], how="left")
before, after = c.datetime < CUT, c.datetime >= CUT
res = {
    "LightGBM 10/15 이전 최대 차이": np.abs(c.lightgbm2 - c.lightgbm)[before].max(),
    "LightGBM 10/15 이후 평균 차이(교란이 먹히는지)": np.abs(c.lightgbm2 - c.lightgbm)[after].mean(),
    "LSTM 10/15 이전 최대 차이": np.abs(c.lstm2 - c.lstm)[before].max(),
    "LSTM 10/15 이후 평균 차이": np.abs(c.lstm2 - c.lstm)[after].mean(),
}
print("\n[미래값 교란 테스트] (10/15 이후 값을 0으로 변경)")
for k, v in res.items():
    print(f"  {k}: {v:.4f}")
ok = res["LightGBM 10/15 이전 최대 차이"] < 1e-6 and res["LSTM 10/15 이전 최대 차이"] < 1e-2 \
    and res["LightGBM 10/15 이후 평균 차이(교란이 먹히는지)"] > 1 and res["LSTM 10/15 이후 평균 차이"] > 1
print("판정:", "통과(미래값을 쓰지 않음, 교란은 이후 시각에만 영향)" if ok else "실패 - 확인 필요")
