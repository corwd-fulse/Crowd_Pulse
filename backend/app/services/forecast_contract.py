"""Future integration contract only. Does not load, execute or retrain artifacts."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ForecastContract(BaseModel):
    model_config=ConfigDict(extra='forbid')
    model_version:str=Field(min_length=1)
    artifact_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    input_columns:list[str]=Field(min_length=1)
    target_kind:Literal['administrative_population','venue_total','venue_zone','center_arrivals']
    target_zone_id:str|None=None
    horizons_seconds:list[int]=Field(min_length=1)
    training_unit:str=Field(min_length=1)
    camera_or_zone_identifiers:list[str]=Field(min_length=1)
    minimum_history_seconds:int=Field(gt=0)
    python_version:str=Field(min_length=1)
    package_versions:dict[str,str]=Field(min_length=1)
    preprocessing_version:str=Field(min_length=1)
    validation_metrics:dict=Field(min_length=1)
    example_output:dict=Field(min_length=1)

    @model_validator(mode='after')
    def scoped(self):
        if self.target_kind in ('venue_zone','center_arrivals') and not self.target_zone_id:
            raise ValueError('A zone-scoped target needs its explicit zone ID')
        if any(h<=0 for h in self.horizons_seconds):raise ValueError('Positive horizons required')
        return self


def validate_center_output(contract:ForecastContract,output:dict,center_zone_id:str):
    if contract.target_kind not in ('venue_zone','center_arrivals') or contract.target_zone_id!=center_zone_id:
        raise ValueError('This model does not predict the selected center target')
    if output.get('target_zone_id')!=center_zone_id or output.get('horizon_seconds') not in contract.horizons_seconds:
        raise ValueError('Output scope/horizon does not match contract')
    value=output.get('value')
    import math
    if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or value<0:
        raise ValueError('Nonnegative finite prediction required')
    return {**output,'model_version':contract.model_version,'source':'external_forecast_model',
            'target_kind':contract.target_kind,'artifact_sha256':contract.artifact_sha256}


def validate_hybrid_compatibility(left: ForecastContract, right: ForecastContract):
    """A preparation guard only: never mix administrative and video predictions."""
    fields = ('target_kind', 'target_zone_id', 'training_unit', 'horizons_seconds',
              'camera_or_zone_identifiers', 'preprocessing_version')
    for field in fields:
        a, b = getattr(left, field), getattr(right, field)
        if isinstance(a, list):
            a, b = set(a), set(b)
        if a != b:
            raise ValueError(f'Hybrid models disagree on {field}')
    return {'compatible_contract': True, 'enabled': False,
            'reason': 'Requires aligned observation timestamps, supplied inference outputs and validated blending weights'}
