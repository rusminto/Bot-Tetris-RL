// Checks that the userscript computes the same policy features as reinforcement-learning/src/tetris_sim.py,
// which the policies are trained and evaluated in.  Run: node tests/features_parity.js
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..', '..');

// Load policies.js the way build.py embeds it, with dummy weights
function loadPolicies() {
  const src = fs.readFileSync(path.join(ROOT, 'userscript', 'src', 'policies.js'), 'utf8')
    .replace('__CEM_WEIGHTS__', '{}').replace('__DQN_V1_WEIGHTS__', '{}').replace('__DQN_V2_WEIGHTS__', '{}');
  return new Function(`${src}\nreturn { extractCemFeatures, calculateDqnV1Features, calculateDqnV2Features };`)();
}

// Small deterministic PRNG so failures are reproducible
function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomCase(rand) {
  // rows[y] bitmasks, y = 0 is the bottom; mix of dense, sparse and "Tetris well" boards
  const rows = new Array(20).fill(0);
  const height = Math.floor(rand() * 21);
  const density = rand();
  const well = rand() < 0.4 ? Math.floor(rand() * 10) : -1;
  for (let y = 0; y < height; y++) {
    let r = 0;
    for (let x = 0; x < 10; x++) if (x !== well && rand() < density) r |= 1 << x;
    if (well >= 0 && rand() < 0.7) r = 1023 & ~(1 << well);
    rows[y] = r;
  }
  const hold = [null, 'I', 'J', 'L', 'O', 'S', 'T', 'Z'][Math.floor(rand() * 8)];
  return { rows, landing: Math.round(rand() * 40) / 2, cleared: Math.floor(rand() * 5), eroded: Math.floor(rand() * 5), level: 1 + Math.floor(rand() * 30), hold };
}

function toBoard(rows) {
  const board = [];
  for (let r = 0; r < 20; r++) {
    const row = new Uint8Array(10);
    for (let x = 0; x < 10; x++) row[x] = (rows[19 - r] >> x) & 1;
    board.push(row);
  }
  return board;
}

const PY = `
import json, sys
sys.path.insert(0, ${JSON.stringify(path.join(ROOT, 'reinforcement-learning', 'src'))})
from tetris_sim import Matrix, cem_features
m = Matrix()
out = []
for c in json.load(open(sys.argv[1])):
    rows = c["rows"] + [0] * 4
    cem = list(cem_features(m, rows, c["landing"], c["cleared"], c["eroded"], c["hold"]))
    holes, bump, total, top = m.dqn_features(rows)
    v1 = [c["cleared"], holes, bump, total]
    v2 = v1 + [top, max(0, min(30, c["level"]) - 1) / 29.0]
    out.append({"cem": cem, "v1": v1, "v2": v2})
print(json.dumps(out))
`;

function main() {
  const js = loadPolicies();
  const rand = mulberry32(12345);
  const cases = Array.from({ length: 3000 }, () => randomCase(rand));
  const tmp = path.join(os.tmpdir(), `tetris-parity-${process.pid}.json`);
  fs.writeFileSync(tmp, JSON.stringify(cases));
  let expected;
  try {
    expected = JSON.parse(execFileSync('python3', ['-c', PY, tmp], { maxBuffer: 64 << 20 }).toString());
  } finally {
    fs.unlinkSync(tmp);
  }
  let failures = 0;
  cases.forEach((c, i) => {
    const board = toBoard(c.rows);
    const got = {
      cem: js.extractCemFeatures(board, c.landing, c.cleared, c.eroded, c.hold),
      v1: js.calculateDqnV1Features(board, c.cleared),
      v2: js.calculateDqnV2Features(board, c.cleared, c.level),
    };
    for (const k of ['cem', 'v1', 'v2']) {
      const a = Array.from(got[k]);
      const b = expected[i][k];
      if (a.length !== b.length || a.some((v, j) => Math.abs(v - b[j]) > 1e-9)) {
        if (failures++ < 5) console.log(`case ${i} ${k}: js ${JSON.stringify(a)} python ${JSON.stringify(b)} rows ${JSON.stringify(c.rows)}`);
      }
    }
  });
  console.log(failures ? `FAIL: ${failures} mismatches` : `ok: ${cases.length} boards, CEM (13) and DQN v1/v2 features match`);
  process.exit(failures ? 1 : 0);
}

main();
