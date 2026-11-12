"""서울 생활인구(OA-23016) 행정동 시간대 원본 CSV 읽기·정제.
학습 데이터 생성(preprocess_living_pop.py)과 추론 파이프라인(serving/pipeline.py)이 같은 규칙을 쓰도록 한곳에 둔다."""
import io, re, zipfile
from pathlib import Path
import pandas as pd

# 인코딩이 달라도 컬럼을 찾도록 위치 사용 (헤더가 깨져도 앞 4개 순서는 고정: 날짜, 시간, 행정동, 총생활인구)
MONTH = re.compile(r"^250_LOCAL_RESD_ADMDONG_(\d{6})(\.zip)?$")
INTERP_LIMIT_H = 2  # 2시간 이하 결측만 선형보간


def read_csv_bytes(b):
    for enc in ("utf-8-sig", "cp949"):
        try:
            text = b.decode(enc)
            break
        except UnicodeDecodeError:
            pass
    else:
        raise ValueError("인코딩 판별 실패")
    text = text.replace(";", ",")  # 일부 일별 파일(2026-07-25~)은 데이터 행만 ';' 구분자
    return pd.read_csv(io.StringIO(text), dtype=str, usecols=[0, 1, 2, 3], header=0,
                       names=["date", "hour", "adm_cd", "total_pop"])


def _zip_members(p):
    with zipfile.ZipFile(p) as z:
        for n in sorted(z.namelist()):
            if n.endswith(".csv"):
                yield n, z.read(n)


def sources(raw_dir):
    """raw_dir 아래 월별 폴더(250_LOCAL_RESD_ADMDONG_YYYYMM)나 ZIP. ' (1)' 중복본은 이름이 맞지 않아 무시된다."""
    for p in sorted(Path(raw_dir).iterdir()):
        if not MONTH.match(p.name):
            continue
        if p.is_dir():
            for f in sorted(p.glob("*.csv")):
                yield f.name, f.read_bytes()
        else:
            yield from _zip_members(p)


def iter_sources(path):
    """추론 입력용: CSV 파일 / ZIP / 폴더 / 이들의 목록. 폴더 안에 월별 폴더가 있으면 sources()와 같은 규칙, 아니면 폴더의 *.csv."""
    if isinstance(path, (list, tuple)):
        for p in path:
            yield from iter_sources(p)
        return
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_dir():
        if any(MONTH.match(c.name) for c in p.iterdir()):
            yield from sources(p)
        else:
            for f in sorted(p.glob("*.csv")):
                yield f.name, f.read_bytes()
    elif p.suffix.lower() == ".zip":
        yield from _zip_members(p)
    else:
        yield p.name, p.read_bytes()


def parse(frames):
    """read_csv_bytes 결과 목록 -> (long[datetime, adm_cd, total_pop], 동별 중복 건수, 원래 행 수). 중복은 마지막 값 사용."""
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    df["datetime"] = pd.to_datetime(df["date"] + df["hour"].str.zfill(2), format="%Y%m%d%H")
    df["total_pop"] = pd.to_numeric(df["total_pop"], errors="coerce")
    df["adm_cd"] = df["adm_cd"].str.strip()
    dup_n = df[df.duplicated(["datetime", "adm_cd"], keep="last")].groupby("adm_cd").size()
    df = df.drop_duplicates(["datetime", "adm_cd"], keep="last")  # 월별 CSV와 일별 CSV 겹침 제거
    return df[["datetime", "adm_cd", "total_pop"]], dup_n, n0


def to_wide(df, interp_limit=INTERP_LIMIT_H):
    """long -> wide(index=1시간 간격 datetime, columns=adm_cd). 반환: (보간 후 wide, 원래 결측 마스크, 보간된 칸 마스크)"""
    wide = df.pivot(index="datetime", columns="adm_cd", values="total_pop")
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="h"))
    wide.index.name = "datetime"
    missing = wide.isna()
    wide = wide.interpolate(limit=interp_limit, limit_area="inside")
    return wide, missing, missing & wide.notna()
