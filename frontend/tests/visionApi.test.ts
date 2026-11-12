// @vitest-environment node
import {afterEach, expect, it} from 'vitest';
import {api, getVisionResultBundle} from '../src/services/api';
import sample from './fixtures/visionBundle.json';

const original = api.defaults.adapter;
afterEach(() => {api.defaults.adapter = original;});

it('fetches all pages and preserves the real response fields', async () => {
  const offsets: number[]=[];
  api.defaults.adapter=async (config) => {
    let data: unknown;
    if (config.url?.endsWith('/results')) data=sample.summary;
    else if (config.url?.endsWith('/zone-metrics')) {
      const offset=config.params.offset;
      offsets.push(offset);
      data={items:sample.metrics.slice(0,3).slice(offset,offset+2),limit:2,offset,direction_presentation:sample.summary.direction_presentation};
    } else data={items:sample.scores,limit:1000,offset:0};
    return {data,status:200,statusText:'OK',headers:{},config};
  };
  const result=await getVisionResultBundle(sample.summary.analysis_job_id);
  expect(offsets).toEqual([0,2]);
  expect(result.metrics).toHaveLength(3);
  expect(result.metrics[0].mean_person_count).toBe(3.9);
  expect(result.scores[0].risk_level).toBe('낮음');
});

it('rejects mixed-job responses instead of showing another result', async () => {
  api.defaults.adapter=async (config) => ({
    data:config.url?.endsWith('/results') ? {...sample.summary,analysis_job_id:'wrong'} : {items:[],limit:1000,offset:0},
    status:200,statusText:'OK',headers:{},config,
  });
  await expect(getVisionResultBundle(sample.summary.analysis_job_id)).rejects.toThrow('다른 작업');
});
