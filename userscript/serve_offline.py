"""
Serves the offline copy of the game (the tetris/ submodule) with the built userscript injected,
so the bot runs without a userscript manager and without modifying the game files.

    python3 userscript/build.py
    python3 userscript/serve_offline.py          # then open http://localhost:8000/
"""

import argparse
import functools
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GAME_DIR = ROOT / "tetris"
BOT_FILE = ROOT / "dist" / "tetris_bot.user.js"
BOT_URL = "/tetris_bot.user.js"

# The bot has to run after SystemJS loads (it hooks System.register) and before the game imports
# its modules, which is right after the import map in game.html
ANCHOR = '<script src="src/import-map.json" type="systemjs-importmap" charset="utf-8"></script>'
# Like a userscript manager, also run it in the top page (index.html), which shows the HUD around the game iframe
TOP_ANCHOR = '</head>'


class GameHandler(SimpleHTTPRequestHandler):
    inject_bot = True

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if self.inject_bot and path == BOT_URL:
            return self._send(BOT_FILE.read_bytes(), "application/javascript")
        if self.inject_bot and path.endswith("/game.html"):
            return self._inject(path, ANCHOR, f'{ANCHOR}\n\t\t<script src="{BOT_URL}"></script>')
        if self.inject_bot and path in ("/", "/index.html"):
            return self._inject("/index.html", TOP_ANCHOR, f'  <script src="{BOT_URL}"></script>\n{TOP_ANCHOR}')
        return super().do_GET()

    def _inject(self, path, anchor, replacement):
        page = Path(self.translate_path(path))
        if not page.is_file():
            return super().do_GET()
        html = page.read_text(encoding="utf-8").replace(anchor, replacement, 1)
        return self._send(html.encode("utf-8"), "text/html; charset=utf-8")

    def _send(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Serve the offline game with the bot injected")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--no-bot", action="store_true", help="Serve the game unmodified (e.g. to use Tampermonkey)")
    args = parser.parse_args()

    if not (GAME_DIR / "index.html").exists():
        sys.exit("tetris/ is empty: run `git submodule update --init` to fetch the offline game first")
    if not args.no_bot and not BOT_FILE.exists():
        sys.exit("dist/tetris_bot.user.js not found: run `python3 userscript/build.py` first")

    GameHandler.inject_bot = not args.no_bot
    handler = functools.partial(GameHandler, directory=str(GAME_DIR))
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"Serving {GAME_DIR} on http://{'localhost' if args.bind == '127.0.0.1' else args.bind}:{args.port}/"
          f"{'' if args.no_bot else ' with the bot injected'} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
