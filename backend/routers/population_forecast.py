"""Read-only population inference; no video jobs, DB writes, or original CSV output."""
import asyncio
import csv
import io
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from app.services.population_forecast import broker, catalog, ForecastError

router = APIRouter(prefix='/api/population-forecast', tags=['Population forecast'])
KST = timezone(timedelta(hours=9))
MAX_BYTES = 10*1024*1024


class Observation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    observed_at: datetime
    area_id: str = Field(pattern=r'^\d{8}$')
    estimate: float | None = Field(ge=0, allow_inf_nan=False)

    @field_validator('observed_at')
    @classmethod
    def hourly(cls, value):
        value = value.replace(tzinfo=KST) if value.tzinfo is None else value.astimezone(KST)
        if value.minute or value.second or value.microsecond:
            raise ValueError('생활인구 이력은 정시 1시간 단위여야 합니다.')
        return value


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    observations: list[Observation] = Field(min_length=1, max_length=100000)

    @model_validator(mode='after')
    def bounded_history(self):
        keys = {(r.area_id, r.observed_at) for r in self.observations}
        if len(keys) != len(self.observations):
            raise ValueError('같은 지역·시각의 중복 입력은 허용하지 않습니다.')
        if len({r.area_id for r in self.observations}) > 427:
            raise ValueError('최대 427개 지역을 지원합니다.')
        times = [r.observed_at for r in self.observations]
        if max(times)-min(times) > timedelta(days=366):
            raise ValueError('한 번에 최대 366일 이력을 입력하세요.')
        return self


def canonical(request):
    output = io.StringIO(newline='')
    writer = csv.writer(output)
    writer.writerow(['observed_at', 'area_id', 'estimate'])
    for row in sorted(request.observations, key=lambda r: (r.observed_at, r.area_id)):
        writer.writerow([row.observed_at.isoformat(), row.area_id, row.estimate])
    return output.getvalue().encode('utf8')


def parse_csv(content):
    for encoding in ('utf-8-sig', 'cp949'):
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            pass
    else:
        raise ValueError('CSV 인코딩은 UTF-8 또는 CP949여야 합니다.')
    reader = csv.reader(io.StringIO(text.replace(';', ',')))
    header = next(reader, [])
    if {'sec', 'zone', 'count'} <= set(header):
        raise ValueError('영상 ROI CSV는 생활인구 입력이 아닙니다.')
    standard = {'observed_at', 'area_id', 'estimate'} <= set(header)
    if len(header) < 4 and not standard:
        raise ValueError('observed_at,area_id,estimate 또는 서울 생활인구 앞 4열이 필요합니다.')
    rows = []
    for cells in reader:
        if not cells or all(not c.strip() for c in cells):
            continue
        if len(rows) >= 100000:
            raise ValueError('최대 100,000개 관측 행을 지원합니다.')
        if standard:
            time, area, estimate = (cells[header.index(k)].strip() for k in ('observed_at', 'area_id', 'estimate'))
        else:
            day, hour, area, estimate = cells[:4]
            time = datetime.strptime(day.strip()+hour.strip().zfill(2), '%Y%m%d%H').replace(tzinfo=KST)
        rows.append(Observation(observed_at=time, area_id=area.strip(), estimate=None if not estimate.strip() else float(estimate)))
    return rows


def parse_upload(content, filename):
    if filename.lower().endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries = [e for e in archive.infolist() if e.filename.lower().endswith('.csv') and not e.is_dir()]
            if not entries or len(entries) > 128 or sum(e.file_size for e in entries) > 32*1024*1024:
                raise ValueError('ZIP은 CSV 1~128개, 압축 해제 총 32MiB 이하여야 합니다.')
            rows = []
            for entry in sorted(entries, key=lambda e: e.filename):
                rows.extend(parse_csv(archive.read(entry)))
                if len(rows) > 100000:
                    raise ValueError('최대 100,000개 관측 행을 지원합니다.')
    elif filename.lower().endswith('.csv'):
        rows = parse_csv(content)
    else:
        raise ValueError('CSV 또는 ZIP 파일을 선택하세요.')
    return ForecastRequest(observations=rows)


async def predict(request):
    try:
        future = await asyncio.to_thread(broker.submit, canonical(request), '.csv')
        return await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), timeout=95)
    except ForecastError as exc:
        raise HTTPException(exc.http_status, detail={'code': exc.code, 'message': exc.message}) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, detail={'code': 'timeout', 'message': '예측 대기 시간을 초과했습니다. 잠시 후 다시 조회하세요.'}) from exc


@router.get('/catalog')
def get_catalog():
    return catalog()


@router.post('/predict')
async def predict_history(request: ForecastRequest):
    return await predict(request)


@router.post('/upload')
async def predict_file(file: Annotated[UploadFile, File()]):
    try:
        content = await file.read(MAX_BYTES+1)
        if len(content) > MAX_BYTES:
            raise HTTPException(413, detail={'code': 'input_too_large', 'message': '입력 파일은 10MiB 이하여야 합니다.'})
        request = await asyncio.to_thread(parse_upload, content, file.filename or '')
    except (ValueError, IndexError, KeyError, zipfile.BadZipFile, RuntimeError) as exc:
        raise HTTPException(422, detail={'code': 'invalid_history', 'message': str(exc)[:300]}) from exc
    finally:
        await file.close()
    return await predict(request)
