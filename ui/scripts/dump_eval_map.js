const fs = require('fs');
const path = require('path');

const p = path.join(process.cwd(), 'datasets', 'jinx_references', 'Z-Image-Turbo.json');
const js = JSON.parse(fs.readFileSync(p, 'utf-8'));
const dsKey = Object.keys(js.datasets)[0];
console.log('DatasetKey:', dsKey);
const itemStats = js.datasets[dsKey].item_stats || {};
const map = {};
for (const [pathKey, stats] of Object.entries(itemStats)) {
  const parts = String(pathKey).split(/[\\/]/);
  const b = parts[parts.length - 1];
  const raw = typeof stats.average_loss_raw !== 'undefined' && stats.average_loss_raw !== null ? Number(stats.average_loss_raw) : undefined;
  const norm = typeof stats.average_loss !== 'undefined' && stats.average_loss !== null ? Number(stats.average_loss) : undefined;
  const abl = typeof stats.average_ablation_delta !== 'undefined' && stats.average_ablation_delta !== null ? Number(stats.average_ablation_delta) : undefined;
  map[b] = { raw, norm, abl };
}
console.log(map);
