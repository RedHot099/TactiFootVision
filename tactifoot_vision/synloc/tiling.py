from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TileWindow:
    x1: int
    y1: int
    x2: int
    y2: int
    index: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def generate_tiles(
    image_width: int,
    image_height: int,
    *,
    tile_size: int,
    overlap: int,
) -> list[TileWindow]:
    stride = max(1, tile_size - overlap)
    xs = _tile_starts(image_width, tile_size, stride)
    ys = _tile_starts(image_height, tile_size, stride)
    tiles: list[TileWindow] = []
    index = 0
    for y in ys:
        for x in xs:
            tiles.append(
                TileWindow(
                    x1=x,
                    y1=y,
                    x2=min(image_width, x + tile_size),
                    y2=min(image_height, y + tile_size),
                    index=index,
                )
            )
            index += 1
    return tiles


def _tile_starts(size: int, tile_size: int, stride: int) -> list[int]:
    if size <= tile_size:
        return [0]
    starts = list(range(0, max(1, size - tile_size + 1), stride))
    final_start = size - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts
