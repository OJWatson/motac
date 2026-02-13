from __future__ import annotations

from dataclasses import dataclass

from pyproj import CRS, Transformer


def utm_crs_for_lonlat(lon: float, lat: float) -> CRS:
    zone = int((lon + 180.0) // 6.0) + 1
    epsg = (32600 + zone) if lat >= 0 else (32700 + zone)
    return CRS.from_epsg(epsg)


@dataclass(frozen=True, slots=True)
class LonLatToXY:
    crs_ll: CRS
    crs_xy: CRS
    to_xy: Transformer
    to_ll: Transformer

    @classmethod
    def for_lonlat(cls, lon0: float, lat0: float) -> LonLatToXY:
        crs_ll = CRS.from_epsg(4326)
        crs_xy = utm_crs_for_lonlat(lon0, lat0)
        to_xy = Transformer.from_crs(crs_ll, crs_xy, always_xy=True)
        to_ll = Transformer.from_crs(crs_xy, crs_ll, always_xy=True)
        return cls(crs_ll=crs_ll, crs_xy=crs_xy, to_xy=to_xy, to_ll=to_ll)
