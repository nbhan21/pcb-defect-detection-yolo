import test from 'node:test';
import assert from 'node:assert/strict';
import { csvForDetections, formatApiError, createRequestGate } from '../../../web/helpers.mjs';

test('CSV records quote commas, newlines and double quotes without breaking columns', () => {
  const csv = csvForDetections([
    { class_id: 0, class_name: 'mouse,"bite"\nlabel', confidence: 0.875, bbox_xyxy: [1, 2, 30, 40] },
  ]);
  assert.equal(csv, 'class_id,class_name,confidence,x_min,y_min,x_max,y_max\r\n0,"mouse,""bite""\nlabel",0.875,1,2,30,40\r\n');
});

test('zero detections still produce a downloadable header-only CSV', () => {
  assert.equal(csvForDetections([]), 'class_id,class_name,confidence,x_min,y_min,x_max,y_max\r\n');
});

test('CSV neutralizes formula-like values from unexpected class metadata', () => {
  assert.match(csvForDetections([{ class_id: 0, class_name: '=IMPORT(1)', confidence: 0.5, bbox_xyxy: [0, 0, 1, 1] }]), /'=IMPORT\(1\)/);
});

test('FastAPI validation arrays become readable field errors', () => {
  assert.equal(formatApiError({ detail: [{ loc: ['body', 'confidence'], msg: 'Input should be less than or equal to 0.95' }] }), 'confidence: Input should be less than or equal to 0.95');
});

test('API error supports plain detail, health message and malformed response fallback', () => {
  assert.equal(formatApiError({ detail: 'Model belum siap.' }), 'Model belum siap.');
  assert.equal(formatApiError({ message: 'Checkpoint tidak ditemukan.' }), 'Checkpoint tidak ditemukan.');
  assert.equal(formatApiError(null, 'Coba lagi.'), 'Coba lagi.');
});

test('a superseded request cannot commit a result', () => {
  const gate = createRequestGate();
  const first = gate.next();
  assert.equal(gate.isCurrent(first), true);
  const second = gate.next();
  assert.equal(gate.isCurrent(first), false);
  assert.equal(gate.isCurrent(second), true);
});

test('new file or changed confidence invalidates the active request and download eligibility', () => {
  const gate = createRequestGate();
  const token = gate.next();
  gate.invalidate();
  assert.equal(gate.isCurrent(token), false);
  const replacement = gate.next();
  assert.equal(gate.isCurrent(replacement), true);
});
