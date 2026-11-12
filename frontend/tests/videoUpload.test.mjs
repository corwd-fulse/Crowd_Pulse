import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {test} from 'node:test';
import ts from 'typescript';
import {validateVideo, MAX_VIDEO_SIZE} from '../src/services/videoUpload.ts';

test('backend extensions are accepted regardless of case or browser MIME type', () => {
  for (const name of ['clip.mp4', 'clip.MOV', 'clip.avi', 'clip.webm', 'clip.MP4']) {
    assert.equal(validateVideo({name, size: 1, type: ''}), null);
  }
  for (const name of ['clip.mkv', 'clip.mp4.exe', 'clip', 'clip.mp4.']) {
    assert.match(validateVideo({name, size: 1}), /형식/);
  }
});

test('200MiB boundary matches backend, including its acceptance of empty files', () => {
  for (const size of [0, MAX_VIDEO_SIZE - 1, MAX_VIDEO_SIZE]) {
    assert.equal(validateVideo({name: 'clip.mp4', size}), null);
  }
  assert.match(validateVideo({name: 'clip.mp4', size: MAX_VIDEO_SIZE + 1}), /200MiB/);
});

// Load the Vite module under Node without needing a browser or live upload server.
const require = createRequire(import.meta.url);
const source = await readFile(new URL('../src/services/api.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023},
}).outputText
  .replace('import.meta.env', '({})')
  .replace("'axios'", JSON.stringify(pathToFileURL(require.resolve('axios')).href));
const {api, uploadVideo} = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);

test('upload sends the original file as multipart and returns actual response fields', async () => {
  const file = new File(['video fixture'], 'camera.MP4', {type: 'video/mp4'});
  const response = {
    video_id: 'server-id', filename: file.name, saved_filename: 'server-id.mp4',
    file_size: file.size, status: 'uploaded', data_source: 'user_upload',
  };
  api.defaults.adapter = async (config) => {
    assert.equal(config.method, 'post');
    assert.equal(config.url, '/api/videos/upload');
    assert.equal(config.timeout, 120000);
    assert.ok(config.data instanceof FormData);
    const uploaded = config.data.get('file');
    assert.equal(uploaded.name, file.name);
    assert.equal(await uploaded.text(), await file.text());
    return {data: response, status: 201, statusText: 'Created', headers: {}, config};
  };
  assert.deepEqual(await uploadVideo(file), response);
});

test('upload failure propagates so the UI can show an error and permit retry', async () => {
  api.defaults.adapter = async () => {throw new Error('upload failed');};
  await assert.rejects(uploadVideo(new File(['x'], 'clip.mp4')), /upload failed/);
});
