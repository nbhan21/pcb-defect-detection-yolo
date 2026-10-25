const CSV_HEADER = ['class_id', 'class_name', 'confidence', 'x_min', 'y_min', 'x_max', 'y_max'];

function csvCell(value) {
  let text = value == null ? '' : String(value);
  if (/^[=+@\t\r]/.test(text) || (typeof value === 'string' && /^-/.test(text))) text = `'${text}`;
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

export function csvForDetections(detections) {
  const rows = detections.map((item) => [item.class_id, item.class_name, item.confidence, ...item.bbox_xyxy]);
  return [CSV_HEADER, ...rows].map((row) => row.map(csvCell).join(',')).join('\r\n') + '\r\n';
}

export function formatApiError(payload, fallback = 'Permintaan gagal. Silakan coba lagi.') {
  if (typeof payload?.detail === 'string') return payload.detail;
  if (Array.isArray(payload?.detail)) {
    const errors = payload.detail.map((error) => {
      const field = Array.isArray(error.loc) ? error.loc.filter((part) => part !== 'body').join('.') : '';
      return [field, error.msg].filter(Boolean).join(': ');
    }).filter(Boolean);
    if (errors.length) return errors.join('; ');
  }
  if (typeof payload?.message === 'string') return payload.message;
  return fallback;
}

export function createRequestGate() {
  let generation = 0;
  return {
    next() { generation += 1; return generation; },
    invalidate() { generation += 1; },
    isCurrent(token) { return token === generation; },
  };
}
