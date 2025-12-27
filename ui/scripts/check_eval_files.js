const fs = require('fs');
const path = require('path');

const dataset = 'jinx_references';
const base = path.join(process.cwd(), 'datasets', dataset);
const entries = fs.readdirSync(base, { withFileTypes: true });
for (const e of entries) {
  if (!e.isFile() || !e.name.endsWith('.json')) continue;
  const p = path.join(base, e.name);
  try {
    const contents = fs.readFileSync(p, 'utf-8');
    const parsed = JSON.parse(contents);
    if (parsed && parsed.datasets && typeof parsed.datasets === 'object') {
      const cfg = parsed.config || {};
      const modelName = cfg.model || (cfg.model_config && cfg.model_config.name_or_path) || null;
      if (modelName) {
        const st = fs.statSync(p);
        console.log(`${e.name} -> model: ${modelName}, mtime: ${new Date(st.mtimeMs).toISOString()}`);
        continue;
      }
    }
    console.log(`${e.name} -> SKIPPED (not a valid eval JSON with model)`);
  } catch (err) {
    console.log(`${e.name} -> SKIPPED (parse error: ${err.message})`);
  }
}