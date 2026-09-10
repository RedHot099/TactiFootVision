"""Serve the verified public review packet on loopback, with byte ranges for video seeking."""

import argparse
import functools
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from diagnostic_review import verify_packet


class ReviewHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.remaining = None
        path = Path(self.translate_path(self.path))
        if not path.resolve().is_relative_to(Path(self.directory).resolve()):
            self.send_error(404)
            return None
        if path.is_dir():
            return super().send_head()
        try:
            source = path.open("rb")
        except OSError:
            self.send_error(404)
            return None
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        request_range = self.headers.get("Range")
        if request_range:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", request_range)
            try:
                if not match or not any(match.groups()) or not size:
                    raise ValueError("Invalid byte range")
                first, last = match.groups()
                if first:
                    start = int(first)
                    end = min(int(last), size - 1) if last else size - 1
                else:
                    length = int(last)
                    if length <= 0:
                        raise ValueError("Empty suffix")
                    start = max(0, size - length)
                if start > end or start >= size:
                    raise ValueError("Unsatisfiable byte range")
                status = 206
            except ValueError:
                source.close()
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
        self.send_response(status)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        source.seek(start)
        self.remaining = end - start + 1
        return source

    def copyfile(self, source, outputfile):
        if self.remaining is None:
            return super().copyfile(source, outputfile)
        try:
            while self.remaining > 0:
                chunk = source.read(min(self.remaining, 64 * 1024))
                if not chunk:
                    break
                outputfile.write(chunk)
                self.remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Browsers cancel an in-flight range when seeking to another time.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=Path("cache/v15/human_review"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    verify_packet(args.packet)
    handler = functools.partial(ReviewHandler, directory=str((args.packet / "public").resolve()))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Review: http://127.0.0.1:{args.port} (public packet only)", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
