import functools
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from serve_review import ReviewHandler  # noqa: E402


def test_video_ranges_and_private_file_isolation(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    data = bytes(range(256)) * 4
    (public / "clip.mp4").write_bytes(data)
    (tmp_path / "sample.csv").write_text("private labels")
    (public / "escape.csv").symlink_to(tmp_path / "sample.csv")
    handler = functools.partial(ReviewHandler, directory=str(public))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        for header, expected, content_range in [("bytes=100-199", data[100:200], "bytes 100-199/1024"),
                                                ("bytes=1000-", data[1000:], "bytes 1000-1023/1024"),
                                                ("bytes=-10", data[-10:], "bytes 1014-1023/1024")]:
            with urlopen(Request(base + "/clip.mp4", headers={"Range": header})) as response:
                assert response.status == 206
                assert response.headers["Content-Range"] == content_range
                assert response.read() == expected
        with urlopen(base + "/clip.mp4") as response:
            assert response.read() == data
        for header in ("bytes=9999-", "bytes=-0", "bytes=10-1", "bytes=0-2,8-10"):
            with pytest.raises(HTTPError) as error:
                urlopen(Request(base + "/clip.mp4", headers={"Range": header}))
            assert error.value.code == 416
        for path in ("/sample.csv", "/../sample.csv", "/escape.csv"):
            with pytest.raises(HTTPError) as error:
                urlopen(base + path)
            assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
