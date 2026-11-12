import {spawnSync} from 'node:child_process';

// Enable the opt-in GET-only integration test in this subprocess only.
const result = spawnSync(process.execPath, ['node_modules/vitest/vitest.mjs', 'run', 'tests/visionLive.test.ts', 'tests/visionLiveUI.test.tsx'], {
  stdio: 'inherit', env: {...process.env, RUN_LIVE_API: '1'},
});
process.exit(result.status ?? 1);
