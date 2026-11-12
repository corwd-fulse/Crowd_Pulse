// @vitest-environment node
import {expect, it} from 'vitest';
import {getAnalysisJob, getVisionResultBundle} from '../src/services/api';

it.skipIf(process.env.RUN_LIVE_API !== '1')('reads the persisted Worker test result through the frontend API client', async () => {
  const id='9bfe14ca-8932-465e-9347-4a50c0fa4904';
  const job=await getAnalysisJob(id);
  expect(job.status).toBe('succeeded');
  const data=await getVisionResultBundle(id);
  expect(data.metrics).toHaveLength(34);
  expect(data.scores).toHaveLength(4);
  expect(Object.keys(data.summary.artifacts)).toHaveLength(5);
  expect(data.summary.analysis_mode).toBe('test');
  expect(data.summary.roi_snapshot.approved_for_analysis).toBe(false);
  expect(data.summary.direction_presentation.quality).toBe('unvalidated');
  expect(data.metrics.filter((row) => row.low_confidence && row.flow_angle_deg===null)).toHaveLength(10);
  expect(data.metrics.every((row) => !row.flow_direction_confirmed && !row.individual_direction_confirmed)).toBe(true);
},20000);
