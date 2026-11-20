"""영상 구역 인원 LSTM(학습 forecast/training/train_video.py · 서비스 video_pipeline.py 공용).
입력: 최근 SEQ_L초 시퀀스(비율값), 출력: 예측 시간별 변화량 비율. 예측 인원 = 지금 값 + 출력 × ref (생활인구 LSTM과 같은 잔차 구조)."""
import numpy as np
import torch
import torch.nn as nn


class VideoNet(nn.Module):
    def __init__(s, n_in, hidden=32, n_out=3):
        super().__init__()
        s.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        s.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, n_out))

    def forward(s, x):
        o, _ = s.lstm(x)
        return s.head(o[:, -1])


def lstm_predict(net, S, batch=4096):
    """S[행, SEQ_L, 특징] -> 출력[행, 예측 시간 수] (numpy)."""
    dev = next(net.parameters()).device
    net.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(S), batch):
            out.append(net(torch.from_numpy(np.ascontiguousarray(S[b:b + batch], np.float32)).to(dev)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, net.head[-1].out_features), np.float32)
