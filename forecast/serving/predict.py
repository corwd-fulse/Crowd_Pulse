"""1시간 뒤 생활인구 예측 (LightGBM + LSTM 36에폭). CPU에서 동작한다. 보통은 pipeline.run_forecast()가 이 클래스를 부른다.
입력: 행정동별 시간별 인구(명) 이력 (index=1시간 간격 시각, columns=행정동코드), 마지막 행이 예측 시각 T의 직전 시각(T-1시간).
출력: 행정동별 T시각 예측(명)과 상태(status). 이력이 모자라거나(insufficient_history) 중간에 비었거나(gap_in_history)
학습한 427개 동이 아니면(unknown_area) 그 동은 예측값을 비우고 이유를 적는다.
특징 생성은 학습 때(features_1h.py, train_lstm.py)와 같은 정의이며 test_parity.py, test_pipeline.py가 저장된 예측과 일치하는지 확인한다."""
import gzip, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, lightgbm as lgb, holidays

A = Path(__file__).resolve().parent / "artifacts"
L = 168
LAGS = (1, 2, 3, 6, 12, 24, 48, 72, 96, 120, 144, 168)
SUBSTITUTE = {pd.Timestamp("2026-09-28").date()}  # 추석 대체공휴일: holidays 라이브러리에 없어 학습 때도 수동 추가했다


def holiday_dates(years):
    return set(pd.to_datetime(list(holidays.KR(years=list(years)))).date) | SUBSTITUTE


def history_status(window):
    """window: (L, n) 최근 L시간 값. 동별 상태: ok / insufficient_history(앞쪽만 비어 있음 = 관측 시작이 늦음) / gap_in_history(중간이 빔)."""
    nan = np.isnan(window)
    first = np.where(nan.all(0), L, nan.argmin(0))  # 처음으로 값이 있는 위치
    after = np.array([nan[f:, k].any() for k, f in enumerate(first)])
    return np.where(~nan.any(0), "ok", np.where(after, "gap_in_history", "insufficient_history"))


class Net(nn.Module):  # train_lstm.py와 같은 구조 (가중치 키가 같아야 load_state_dict가 통과한다)
    def __init__(s, n, h=96):
        super().__init__()
        s.emb = nn.Embedding(n, 16)
        s.lstm = nn.LSTM(8, h, num_layers=2, batch_first=True, dropout=0.1)
        s.head = nn.Sequential(nn.Linear(h + 16 + 7, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(s, seq, tgt, j, last):
        o, _ = s.lstm(seq)
        return last + s.head(torch.cat([o[:, -1], s.emb(j), tgt], -1)).squeeze(-1)


def load_booster(path=None):
    """LightGBM 모델: artifacts의 gzip본(lgbm_1h.txt.gz)을 기본으로, 원본 .txt 경로도 받는다."""
    p = Path(path) if path else (A / "lgbm_1h.txt.gz" if (A / "lgbm_1h.txt.gz").exists() else A / "lgbm_1h.txt")
    if p.suffix == ".gz":
        return lgb.Booster(model_str=gzip.decompress(p.read_bytes()).decode("utf-8"))
    return lgb.Booster(model_file=str(p))


class Predictor:
    def __init__(self, lgbm_path=None, lstm_path=None, device="cpu"):
        self.dongs = json.loads((A / "dong_order.json").read_text(encoding="utf-8"))  # LSTM 동 임베딩 인덱스 = 이 순서
        self.cols = json.loads((A / "feature_cols.json").read_text(encoding="utf-8"))  # LightGBM 입력 특징 순서
        self.scale = pd.read_csv(A / "scale.csv", dtype={"adm_cd": str}).set_index("adm_cd")["scale"].reindex(self.dongs)  # 학습 구간 동별 평균(명)
        self.device = torch.device(device)
        self.net = Net(len(self.dongs)).to(self.device)
        self.net.load_state_dict(torch.load(lstm_path or A / "lstm_1h_36ep.pt", map_location=self.device))  # GPU에서 저장된 가중치도 CPU에서 로드
        self.net.eval()
        self.booster = load_booster(lgbm_path)

    def _calendar(self, times):
        """times: DatetimeIndex -> (n,7) [sin hour, cos hour, sin dow, cos dow, 오늘 공휴일, 다음날 공휴일, 전날 공휴일]"""
        hol = holiday_dates(range(times.year.min() - 1, times.year.max() + 2))
        f = lambda off: pd.Series((times + pd.Timedelta(days=off)).date).isin(hol).astype(np.float32).to_numpy()
        h, d = times.hour.to_numpy(), times.dayofweek.to_numpy()
        return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7), f(0), f(1), f(-1)], 1).astype(np.float32)

    def predict(self, history, target_time):
        T = pd.Timestamp(target_time)
        h = history.copy(); h.columns = h.columns.astype(str); h = h.sort_index()
        if h.index[-1] != T - pd.Timedelta(hours=1):
            raise ValueError(f"이력의 마지막 시각({h.index[-1]})이 예측 시각 직전({T - pd.Timedelta(hours=1)})이어야 합니다.")
        tail = h.index[-(L + 1):]
        if not (pd.Series(tail).diff().dropna() == pd.Timedelta(hours=1)).all():
            raise ValueError("이력의 시각이 1시간 간격으로 연속되지 않았습니다.")
        unknown = [c for c in h.columns if c not in set(self.dongs)]
        known = h.reindex(columns=self.dongs)  # 입력에 없는 동은 전부 NaN -> insufficient_history
        n = len(self.dongs)
        status = np.full(n, "insufficient_history", dtype=object)
        if len(known) >= L:
            status = history_status(known.iloc[-L:].to_numpy(np.float64))
        p_lgbm, p_lstm = np.full(n, np.nan), np.full(n, np.nan)
        ok = np.where(status == "ok")[0]
        if len(ok):
            sc = self.scale.to_numpy()[ok]
            r = known.iloc[-(L + 1):, ok].to_numpy(np.float64) / sc  # 비율값. 맨 아래 행 = T-1시간
            # --- LightGBM 특징 (features_1h.load_features와 같은 정의) ---
            feat = {f"lag_{k}h": r[-k] for k in LAGS}
            feat["diff_1h"], feat["diff_3h"] = r[-1] - r[-2], r[-1] - r[-4]
            for m in (24, 168):
                feat[f"mean_{m}h"], feat[f"std_{m}h"] = r[-m:].mean(0), r[-m:].std(0, ddof=1)
            feat["surge_vs_week"] = r[-1] / r[-(L + 1)] if len(r) > L else np.full(len(ok), np.nan)  # 169시간 전이 없으면 NaN(학습 때도 허용)
            cal = self._calendar(pd.DatetimeIndex([T]))[0]
            X = pd.DataFrame(feat)
            X["is_hol"], X["is_hol_next"], X["is_hol_prev"] = cal[4], cal[5], cal[6]
            X["hour"], X["dow"] = T.hour, T.dayofweek
            X["dong"] = pd.Categorical([self.dongs[i] for i in ok], categories=self.dongs)
            p_lgbm[ok] = self.booster.predict(X[self.cols]) * sc
            # --- LSTM (train_lstm.py의 batch와 같은 구성) ---
            tf_hist = self._calendar(h.index[-L:])
            seq = np.concatenate([r[-L:].T[:, :, None], np.broadcast_to(tf_hist, (len(ok), L, 7))], -1).astype(np.float32)
            dv = self.device
            with torch.no_grad():
                out = self.net(torch.from_numpy(seq).to(dv), torch.from_numpy(np.broadcast_to(cal, (len(ok), 7)).copy()).to(dv),
                               torch.from_numpy(ok).to(dv), torch.from_numpy(r[-1].astype(np.float32)).to(dv))
            p_lstm[ok] = out.cpu().numpy() * sc
        res = pd.DataFrame({"adm_cd": self.dongs, "target_time": T, "lightgbm": p_lgbm, "lstm": p_lstm, "status": status})
        if unknown:
            res = pd.concat([res, pd.DataFrame({"adm_cd": unknown, "target_time": T, "lightgbm": np.nan, "lstm": np.nan, "status": "unknown_area"})], ignore_index=True)
        return res
