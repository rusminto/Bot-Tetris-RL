// Headless end-to-end test: plays the offline game with the built bot in Firefox and reports progress.
//
//   cd userscript && npm install && npx playwright install firefox
//   node tests/e2e.js                          # normal game from Level 1 until the Marathon ends
//   node tests/e2e.js --force-20g --seconds 120 # 0 ms gravity and 150 ms lock delay from the first piece
//   node tests/e2e.js --engine dqn_v2           # cem (default), dqn_v2 or dqn_v1
//   node tests/e2e.js --bot path/to/other.user.js   # test another build instead of dist/
//
// Set PLAYWRIGHT_FIREFOX_PATH to use a specific Firefox binary.
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { firefox } = require('playwright');

const ROOT = path.resolve(__dirname, '..', '..');

function parseArgs(argv) {
  const opts = { seconds: 600, port: 8790, engine: 'cem', force20g: false, bot: null, screenshot: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--force-20g') opts.force20g = true;
    else if (a === '--seconds') opts.seconds = Number(argv[++i]);
    else if (a === '--port') opts.port = Number(argv[++i]);
    else if (a === '--engine') opts.engine = argv[++i];
    else if (a === '--bot') opts.bot = path.resolve(argv[++i]);
    else if (a === '--screenshot') opts.screenshot = path.resolve(argv[++i]);
    else throw new Error(`unknown argument ${a}`);
  }
  return opts;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Runs inside the game iframe: current stats from the player and the bot
function readGameState() {
  const p = window.__tetrisPlayer;
  const m = window.__tetrisModel;
  let lines = 0, level = 0, score = 0;
  for (let i = 0; p.getNumComponents && i < p.getNumComponents(); i++) {
    const c = p.getComponentAtIndex(i);
    if (c && c.getStatValue) lines = Math.max(lines, c.getStatValue(2059679406) || 0);
    if (c && c.getCurrentLevelIndex) level = Math.max(level, c.getCurrentLevelIndex() + 1);
    if (c && c.getScore) score = Math.max(score, c.getScore() || 0);
  }
  const mx = p.getMatrix();
  let height = 0;
  for (let y = 0; y < 24; y++) for (let x = 0; x < 10; x++) if (mx.getMinoAt(x, y) !== null) height = y + 1;
  return {
    active: p.isGameActive(), level, lines, score, height,
    fallMs: m.mNormalFallSpeedMSEC, lockMs: m.mLockTimeMSEC,
    bot: window.__tetrisBotStats || null,
  };
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const server = spawn('python3', [path.join(ROOT, 'userscript', 'serve_offline.py'), '--port', String(opts.port)], { stdio: ['ignore', 'ignore', 'inherit'] });
  const browser = await firefox.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_FIREFOX_PATH || undefined });
  let exitCode = 1;
  try {
    await sleep(800);
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    const warnings = [];
    page.on('console', (m) => { if (m.text().includes('[TetrisRL]') && m.type() !== 'log') warnings.push(m.text()); });
    if (opts.bot) {
      const body = fs.readFileSync(opts.bot, 'utf8');
      await page.route('**/tetris_bot.user.js', (r) => r.fulfill({ body, contentType: 'application/javascript' }));
    }
    await page.goto(`http://127.0.0.1:${opts.port}/index.html`);

    let frame = null;
    for (let i = 0; i < 100 && !frame; i++) {
      frame = page.frames().find((f) => f.url().includes('game.html'));
      if (!frame) await sleep(100);
    }
    if (!frame) throw new Error('game iframe not found');
    for (let i = 0; i < 300; i++) {
      if (await frame.evaluate(() => !!(window.__bpsApp && window.__bpsKeyConverter)).catch(() => false)) break;
      await sleep(200);
    }
    await frame.evaluate((engine) => window.postMessage({ type: 'TETRIS_BOT_CMD', cmd: 'engine', val: engine }, '*'), opts.engine);

    // Press Start (via the bot's own command) until the first piece is live
    for (let i = 0; i < 20; i++) {
      if (await frame.evaluate(() => !!window.__tetrisModel).catch(() => false)) break;
      await frame.evaluate(() => window.postMessage({ type: 'TETRIS_BOT_CMD', cmd: 'start', val: true }, '*'));
      await sleep(1500);
    }
    if (opts.force20g) {
      await frame.evaluate(() => {
        const m = window.__tetrisModel;
        Object.defineProperty(m, 'mNormalFallSpeedMSEC', { get: () => 0, set: () => {}, configurable: true });
        Object.defineProperty(m, 'mLockTimeMSEC', { get: () => 150, set: () => {}, configurable: true });
      });
    }

    const perLevel = {};
    const t0 = Date.now();
    let state = null;
    while ((Date.now() - t0) / 1000 < opts.seconds) {
      await sleep(5000);
      state = await frame.evaluate(readGameState);
      perLevel[state.level] = { fallMs: state.fallMs, lockMs: state.lockMs };
      console.log(`t=${Math.round((Date.now() - t0) / 1000)}s ${JSON.stringify(state)}`);
      if (!state.active) break;
    }
    if (opts.screenshot) await page.screenshot({ path: opts.screenshot });
    const summary = {
      engine: opts.engine, force20g: opts.force20g, ended: state && !state.active,
      level: state && state.level, lines: state && state.lines, score: state && state.score,
      finalHeight: state && state.height, bot: state && state.bot, warnings: warnings.length, timing: perLevel,
    };
    console.log('SUMMARY ' + JSON.stringify(summary));
    for (const w of warnings.slice(0, 5)) console.log('warning: ' + w);
    // The Marathon ends at 300 lines: a game that ended earlier topped out
    exitCode = state && (state.active || state.lines >= 300) ? 0 : 2;
  } finally {
    await browser.close();
    server.kill();
  }
  process.exit(exitCode);
}

main().catch((e) => { console.error(e); process.exit(1); });
