from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import osmnx as ox
import scipy.sparse as sp

from motac.spatial.grid_builder import LonLatBounds, build_regular_grid


def _utm_crs_from_latlon(lat: float, lon: float) -> str:
    """Return an EPSG code string for the local UTM zone."""

    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


# ------------------------- deterministic IO -------------------------

_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)  # earliest ZIP timestamp, for deterministic archives


def _source_date_epoch_utc() -> str:
    """Return a deterministic UTC timestamp string.

    We deliberately avoid wall-clock time so cache bundles are byte-for-byte
    reproducible. If SOURCE_DATE_EPOCH is set (common in reproducible builds),
    we use it; otherwise we fall back to a fixed epoch string.
    """

    s = os.environ.get("SOURCE_DATE_EPOCH")
    if not s:
        return "1970-01-01T00:00:00Z"
    try:
        ts = int(s)
    except ValueError:
        return "1970-01-01T00:00:00Z"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _zip_write_bytes(zf: zipfile.ZipFile, *, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(filename=name)
    info.date_time = _ZIP_EPOCH
    info.compress_type = zipfile.ZIP_DEFLATED
    # Keep permissions stable: -rw-r--r--
    info.external_attr = (0o100644 & 0xFFFF) << 16
    zf.writestr(info, payload)


def _save_graphml_deterministic(G: nx.MultiDiGraph, path: Path) -> None:
    """Save GraphML with stable node/edge/attribute ordering.

    We rely on NetworkX's GraphML writer but pre-canonicalise ordering.
    This avoids subtle non-determinism from dict/hash iteration order that can
    show up in graph.graphml bytes (and thus bundle_sha256).
    """

    def _sorted_attrs(d: dict[str, Any]) -> dict[str, Any]:
        return {k: d[k] for k in sorted(d.keys())}

    # Rebuild graph with stable insertion order for graph attrs, nodes and edges.
    H = nx.MultiDiGraph()
    H.graph.update(_sorted_attrs(dict(G.graph)))

    for n, data in sorted(G.nodes(data=True), key=lambda t: str(t[0])):
        H.add_node(n, **_sorted_attrs(dict(data)))

    for u, v, k, data in sorted(
        G.edges(keys=True, data=True), key=lambda e: (str(e[0]), str(e[1]), str(e[2]))
    ):
        H.add_edge(u, v, key=k, **_sorted_attrs(dict(data)))

    nx.write_graphml(H, path)


def _npy_bytes(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    # Use allow_pickle=False to avoid non-deterministic pickling for object arrays.
    np.save(buf, a, allow_pickle=False)
    return buf.getvalue()


def _save_npz_deterministic(path: Path, *, arrays: dict[str, np.ndarray]) -> None:
    """Write a deterministic .npz (zip) file.

    np.savez_compressed uses zip timestamps, making output non-deterministic.
    This helper fixes timestamps + entry ordering.
    """

    path = Path(path)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for k in sorted(arrays.keys()):
            _zip_write_bytes(zf, name=f"{k}.npy", payload=_npy_bytes(arrays[k]))


def _save_sparse_npz_deterministic(path: Path, mat: sp.spmatrix) -> None:
    """Write a deterministic SciPy sparse .npz.

    Mirrors scipy.sparse.save_npz, but with fixed zip timestamps.
    """

    mat = mat.tocsr()
    arrays: dict[str, np.ndarray] = {
        "data": np.asarray(mat.data),
        "indices": np.asarray(mat.indices),
        "indptr": np.asarray(mat.indptr),
        "shape": np.asarray(mat.shape),
        "format": np.asarray("csr"),
    }

    # 'format' is stored as a 0-d unicode array.
    with zipfile.ZipFile(Path(path), mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for k in sorted(arrays.keys()):
            _zip_write_bytes(zf, name=f"{k}.npy", payload=_npy_bytes(arrays[k]))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _bundle_sha256(cache_dir: Path, artefacts: Iterable[str]) -> str:
    """Compute a stable bundle hash over a subset of files.

    We hash file names + file bytes in a fixed order.
    """

    h = hashlib.sha256()
    for name in artefacts:
        p = cache_dir / name
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        h.update(b"\x00")
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class SubstrateConfig:
    """Configuration for substrate building.

    Provide one of:
      - bbox (north, south, east, west)
      - place (OSM place query string)
      - graphml_path (local graph for offline/testing)

    POIs are optional. For M1 we support:
      - disable_pois=True
      - poi_geojson_path (local) OR poi_tags + (bbox/place/graph)
    """

    # region spec
    north: float | None = None
    south: float | None = None
    east: float | None = None
    west: float | None = None
    place: str | None = None

    # offline graph
    graphml_path: str | None = None

    # grid
    cell_size_m: float = 250.0

    # neighbourhood
    max_travel_time_s: float = 900.0

    # POIs
    disable_pois: bool = False
    poi_tags: dict[str, Any] | None = None
    poi_geojson_path: str | None = None
    poi_travel_time_features: bool = False

    # caching
    cache_dir: str | None = None

    @staticmethod
    def from_json(path: str | Path) -> SubstrateConfig:
        data = json.loads(Path(path).read_text())
        return SubstrateConfig(**data)

    def bbox(self) -> tuple[float, float, float, float] | None:
        if None in (self.north, self.south, self.east, self.west):
            return None
        assert self.north is not None
        assert self.south is not None
        assert self.east is not None
        assert self.west is not None
        return (self.north, self.south, self.east, self.west)


class SubstrateBuilder:
    """Build a road-constrained substrate and persist cache artefacts.

    Cache format (v1)
    ---------------
    When ``SubstrateConfig.cache_dir`` is set, ``build()`` writes a self-contained
    artefact bundle:

    - ``graph.graphml``: road network (OSMnx GraphML)
    - ``grid.npz``: grid centroid lat/lon + cell_size_m
    - ``neighbours.npz``: CSR travel-time neighbourhood matrix
    - ``meta.json``: provenance + config hash + format version
    - optionally ``poi.npz``: POI feature matrix
    """

    CACHE_FORMAT_VERSION = 1

    def __init__(self, config: SubstrateConfig):
        self.config = config

    def build(self):
        from .types import Substrate

        cache_dir = Path(self.config.cache_dir) if self.config.cache_dir else None
        if cache_dir and self._cache_exists(cache_dir):
            return self._load_cache(cache_dir)

        G, graphml_path = self._load_graph()
        grid = self._build_grid(G)
        neighbours = self._build_neighbours(G, grid)
        poi = None
        if not self.config.disable_pois:
            poi = self._build_pois(grid, neighbours)

        # If caching, always persist a GraphML copy so the cache is self-contained.
        #
        # Determinism note: when building from an offline GraphML file, prefer copying
        # the original bytes rather than re-serialising with OSMnx/NetworkX. This
        # avoids subtle non-determinism from XML writer attribute ordering or float
        # formatting differences.
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            graphml_out = cache_dir / "graph.graphml"
            if self.config.graphml_path and graphml_path:
                shutil.copyfile(graphml_path, graphml_out)
            else:
                _save_graphml_deterministic(G, graphml_out)
            graphml_path = str(graphml_out)

        substrate = Substrate(grid=grid, neighbours=neighbours, poi=poi, graphml_path=graphml_path)
        if cache_dir:
            self._save_cache(cache_dir, substrate)
        return substrate

    # ------------------------- graph -------------------------
    def _load_graph(self):
        if self.config.graphml_path:
            path = Path(self.config.graphml_path)
            G = ox.load_graphml(path)
            return G, str(path)

        if self.config.place:
            G = ox.graph_from_place(self.config.place, network_type="drive")
        else:
            bbox = self.config.bbox()
            if bbox is None:
                raise ValueError("Must provide bbox, place or graphml_path")
            north, south, east, west = bbox
            G = ox.graph_from_bbox(north, south, east, west, network_type="drive")

        # add travel_time (seconds)
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        return G, None

    # ------------------------- grid -------------------------
    def _build_grid(self, G: nx.MultiDiGraph):
        from .types import Grid

        # derive bbox from graph nodes if not provided
        lats = np.array([d["y"] for _, d in G.nodes(data=True)], dtype=float)
        lons = np.array([d["x"] for _, d in G.nodes(data=True)], dtype=float)
        north = float(lats.max())
        south = float(lats.min())
        east = float(lons.max())
        west = float(lons.min())

        # local UTM projection based on bbox centre
        lat0 = (north + south) / 2.0
        lon0 = (east + west) / 2.0
        utm_crs = _utm_crs_from_latlon(lat0, lon0)

        import geopandas as gpd
        from shapely.geometry import Point

        # corners
        corners = gpd.GeoSeries([Point(west, south), Point(east, north)], crs="EPSG:4326").to_crs(
            utm_crs
        )
        minx, miny = corners.iloc[0].x, corners.iloc[0].y
        maxx, maxy = corners.iloc[1].x, corners.iloc[1].y

        cell = float(self.config.cell_size_m)
        nx_cells = max(1, int(math.ceil((maxx - minx) / cell)))
        ny_cells = max(1, int(math.ceil((maxy - miny) / cell)))

        xs = minx + (np.arange(nx_cells) + 0.5) * cell
        ys = miny + (np.arange(ny_cells) + 0.5) * cell
        xx, yy = np.meshgrid(xs, ys)
        centroids_utm = np.stack([xx.ravel(), yy.ravel()], axis=1)

        centroids = gpd.GeoSeries(
            gpd.points_from_xy(centroids_utm[:, 0], centroids_utm[:, 1]), crs=utm_crs
        ).to_crs("EPSG:4326")

        lat = centroids.y.to_numpy(dtype=float)
        lon = centroids.x.to_numpy(dtype=float)
        return Grid(lat=lat, lon=lon, cell_size_m=cell)

    # ---------------------- neighbours ----------------------
    def _build_neighbours(self, G: nx.MultiDiGraph, grid):
        from .types import NeighbourSets

        # ensure travel_time exists if graph loaded from file
        if not any("travel_time" in data for _, _, data in G.edges(data=True)):
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)

        # Project graph + points to avoid scikit-learn dependency for haversine search
        utm_crs = _utm_crs_from_latlon(float(np.mean(grid.lat)), float(np.mean(grid.lon)))
        if not ox.projection.is_projected(G.graph.get("crs")):
            Gp = ox.project_graph(G, to_crs=utm_crs)
        else:
            Gp = G

        import geopandas as gpd

        pts = gpd.GeoSeries(gpd.points_from_xy(grid.lon, grid.lat), crs="EPSG:4326").to_crs(utm_crs)
        xs = pts.x.to_numpy(dtype=float)
        ys = pts.y.to_numpy(dtype=float)

        nodes = ox.distance.nearest_nodes(Gp, X=xs, Y=ys)
        # osmnx returns list/array of node ids
        nodes = list(nodes)

        n = len(nodes)
        max_t = float(self.config.max_travel_time_s)

        indptr = [0]
        indices: list[int] = []
        data: list[float] = []

        # Precompute mapping node id to grid cell indices sharing it
        node_to_cells: dict[Any, list[int]] = {}
        for i, nid in enumerate(nodes):
            node_to_cells.setdefault(nid, []).append(i)

        # Dijkstra from each unique mapped node, with cutoff
        for i in range(n):
            src = nodes[i]
            lengths = nx.single_source_dijkstra_path_length(
                Gp, src, cutoff=max_t, weight="travel_time"
            )
            # convert reachable nodes -> reachable grid cells
            neigh_cells: dict[int, float] = {}
            for nid, t in lengths.items():
                for j in node_to_cells.get(nid, []):
                    # keep minimum if multiple paths/duplicates
                    prev = neigh_cells.get(j)
                    if prev is None or t < prev:
                        neigh_cells[j] = float(t)

            # always include self
            if i not in neigh_cells:
                neigh_cells[i] = 0.0

            # store sorted by index for stable CSR
            for j in sorted(neigh_cells):
                indices.append(j)
                data.append(neigh_cells[j])
            indptr.append(len(indices))

        mat = sp.csr_matrix((np.array(data), np.array(indices), np.array(indptr)), shape=(n, n))
        return NeighbourSets(travel_time_s=mat)

    # -------------------------- POIs ------------------------
    def _build_pois(self, grid, neighbours):
        import geopandas as gpd
        from shapely.geometry import shape

        from .types import POIFeatures

        if self.config.poi_geojson_path:
            try:
                gdf = gpd.read_file(self.config.poi_geojson_path)
            except Exception:  # pragma: no cover - optional IO backend
                # Minimal GeoJSON fallback (geometry only) to avoid heavy IO deps in tests.
                raw = json.loads(Path(self.config.poi_geojson_path).read_text())
                feats = raw.get("features", [])
                geoms = [shape(f["geometry"]) for f in feats]
                gdf = gpd.GeoDataFrame({}, geometry=geoms, crs="EPSG:4326")
        else:
            tags = self.config.poi_tags or {"amenity": True}
            bbox = self.config.bbox()
            if bbox is None:
                raise ValueError("POI download requires bbox/place (or supply poi_geojson_path)")
            north, south, east, west = bbox
            gdf = ox.features_from_bbox(north, south, east, west, tags=tags)

        # grid centroids to gdf
        grid_gdf = gpd.GeoDataFrame(
            {"cell": np.arange(len(grid.lat), dtype=int)},
            geometry=gpd.points_from_xy(grid.lon, grid.lat),
            crs="EPSG:4326",
        )

        # assign each POI to nearest grid centroid in a projected CRS
        utm_crs = _utm_crs_from_latlon(float(np.mean(grid.lat)), float(np.mean(grid.lon)))
        grid_utm = grid_gdf.to_crs(utm_crs)
        poi_raw_utm = gdf.to_crs(utm_crs)

        # normalize POI geometries to points in projected CRS
        geom = poi_raw_utm.geometry
        points = geom.copy()
        non_points = ~geom.geom_type.isin(["Point"])
        points.loc[non_points] = geom.loc[non_points].centroid
        poi_utm = gpd.GeoDataFrame(
            poi_raw_utm.drop(columns=["geometry"], errors="ignore"),
            geometry=points,
            crs=utm_crs,
        )

        from scipy.spatial import cKDTree

        grid_xy = np.column_stack([grid_utm.geometry.x.to_numpy(), grid_utm.geometry.y.to_numpy()])
        poi_xy = np.column_stack([poi_utm.geometry.x.to_numpy(), poi_utm.geometry.y.to_numpy()])
        tree = cKDTree(grid_xy)
        _, idx = tree.query(poi_xy, k=1)
        poi_utm["cell"] = idx.astype(int)

        # Features: total POI count + (optional) counts by selected tag key/value.
        #
        # If `config.poi_tags` is provided, we interpret it as a *selection* of
        # columns/values to break out into separate features. This works for both
        # downloaded OSM features and local GeoJSON (as long as properties are present).
        feature_cols: list[tuple[str, str | None]] = []
        tags = self.config.poi_tags
        if isinstance(tags, dict):
            for k, v in tags.items():
                if v is True:
                    feature_cols.append((str(k), None))
                elif isinstance(v, (list, tuple, set)):
                    for vv in v:
                        feature_cols.append((str(k), str(vv)))

        # Always include total count.
        feature_names = ["poi_count"]
        x_parts: list[np.ndarray] = []

        x_total = np.zeros((len(grid.lat), 1), dtype=float)
        counts_total = poi_utm.groupby("cell").size()
        for cell, c in counts_total.items():
            x_total[int(cell), 0] = float(c)
        x_parts.append(x_total)

        # Optional travel-time features: min travel time to any POI cell, and
        # (when available) to each selected POI breakout.
        if bool(self.config.poi_travel_time_features):
            from .features import min_travel_time_feature_matrix

            default = float(self.config.max_travel_time_s)

            masks: dict[str, np.ndarray] = {"poi": x_total[:, 0] > 0}

            # For breakouts, the last len(feature_cols_filtered) x_parts correspond
            # to the added tag columns; derive masks by re-computing counts on the
            # fly to keep naming aligned.
            for k, vv in feature_cols:
                if k not in poi_utm.columns:
                    continue

                col = poi_utm[k]
                if vv is None:
                    mask_rows = col.notna()
                    prefix = f"poi_{k}"
                else:
                    mask_rows = col.astype(str) == vv
                    prefix = f"poi_{k}={vv}"

                counts = poi_utm.loc[mask_rows].groupby("cell").size()
                has = np.zeros((len(grid.lat),), dtype=bool)
                for cell, c in counts.items():
                    if int(c) > 0:
                        has[int(cell)] = True
                masks[prefix] = has

            x_tt, names_tt = min_travel_time_feature_matrix(
                travel_time_s=neighbours.travel_time_s,
                masks=masks,
                default=default,
                suffix="min_travel_time_s",
            )

            x_parts.append(x_tt)
            feature_names.extend(names_tt)

        # Breakouts by tag.
        for k, vv in feature_cols:
            if k not in poi_utm.columns:
                continue
            col = poi_utm[k]
            if vv is None:
                mask = col.notna()
                name = k
            else:
                mask = col.astype(str) == vv
                name = f"{k}={vv}"

            counts = poi_utm.loc[mask].groupby("cell").size()
            xi = np.zeros((len(grid.lat), 1), dtype=float)
            for cell, c in counts.items():
                xi[int(cell), 0] = float(c)
            x_parts.append(xi)
            feature_names.append(name)

        if x_parts:
            x = np.concatenate(x_parts, axis=1)
        else:
            x = np.zeros((len(grid.lat), 0), dtype=float)
        return POIFeatures(x=x, feature_names=feature_names)

    # -------------------------- cache -----------------------
    def _cache_exists(self, cache_dir: Path) -> bool:
        # Require minimal artefacts + a versioned meta file.
        return (
            (cache_dir / "graph.graphml").exists()
            and (cache_dir / "grid.npz").exists()
            and (cache_dir / "neighbours.npz").exists()
            and (cache_dir / "meta.json").exists()
        )

    def _save_cache(self, cache_dir: Path, substrate) -> None:
        from .types import Substrate

        assert isinstance(substrate, Substrate)

        # Core arrays (deterministic .npz writes)
        _save_npz_deterministic(
            cache_dir / "grid.npz",
            arrays={
                "lat": substrate.grid.lat,
                "lon": substrate.grid.lon,
                "cell_size_m": np.array([substrate.grid.cell_size_m], dtype=float),
            },
        )
        _save_sparse_npz_deterministic(
            cache_dir / "neighbours.npz", substrate.neighbours.travel_time_s
        )

        # Provenance
        # Provenance config: keep it stable across runs.
        # In particular, an absolute graphml_path (often a temp path) would make
        # config_sha256 and meta.json non-deterministic.
        graphml_sha256 = _sha256_file(cache_dir / "graph.graphml")
        cfg_dict = {
            "north": self.config.north,
            "south": self.config.south,
            "east": self.config.east,
            "west": self.config.west,
            "place": self.config.place,
            "graphml_path": "graph.graphml" if self.config.graphml_path else None,
            "graphml_sha256": graphml_sha256,
            "cell_size_m": self.config.cell_size_m,
            "max_travel_time_s": self.config.max_travel_time_s,
            "disable_pois": self.config.disable_pois,
            "poi_tags": self.config.poi_tags,
            "poi_geojson_path": self.config.poi_geojson_path,
            "poi_travel_time_features": self.config.poi_travel_time_features,
        }
        cfg_json = json.dumps(cfg_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
        cfg_hash = hashlib.sha256(cfg_json).hexdigest()

        try:
            from motac._version import __version__
        except Exception:  # pragma: no cover
            __version__ = "unknown"

        # Optional POI features.
        if substrate.poi is not None:
            _save_npz_deterministic(
                cache_dir / "poi.npz",
                arrays={
                    "x": substrate.poi.x,
                    # Store as unicode array to avoid pickle.
                    "feature_names": np.array(substrate.poi.feature_names, dtype=str),
                },
            )

        # Bundle hash over the artefact files (excluding meta.json itself).
        artefacts = ["graph.graphml", "grid.npz", "neighbours.npz"]
        if (cache_dir / "poi.npz").exists():
            artefacts.append("poi.npz")
        # Keep deterministic ordering even if this list changes in future.
        artefacts = sorted(artefacts)
        bundle_sha256 = _bundle_sha256(cache_dir, artefacts)

        # Store graphml_path as a stable, cache-relative path.
        # The returned Substrate.graphml_path may be an absolute path to the cache
        # directory, which would make meta.json non-deterministic across runs.
        graphml_meta_path = "graph.graphml"

        meta = {
            "cache_format_version": self.CACHE_FORMAT_VERSION,
            "built_at_utc": _source_date_epoch_utc(),
            "motac_version": __version__,
            "config": cfg_dict,
            "config_sha256": cfg_hash,
            "graphml_path": graphml_meta_path,
            "has_poi": substrate.poi is not None,
            "bundle_sha256": bundle_sha256,
            "bundle_files": artefacts,
        }
        (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    def _load_cache(self, cache_dir: Path):
        from .types import Grid, NeighbourSets, POIFeatures, Substrate

        grid_npz = np.load(cache_dir / "grid.npz", allow_pickle=True)
        grid = Grid(
            lat=grid_npz["lat"].astype(float),
            lon=grid_npz["lon"].astype(float),
            cell_size_m=float(grid_npz["cell_size_m"][0]),
        )
        neighbours = NeighbourSets(travel_time_s=sp.load_npz(cache_dir / "neighbours.npz").tocsr())
        meta = json.loads((cache_dir / "meta.json").read_text())

        version = meta.get("cache_format_version")
        if version != self.CACHE_FORMAT_VERSION:
            raise ValueError(
                "Unsupported substrate cache format version: "
                f"{version} (expected {self.CACHE_FORMAT_VERSION})"
            )

        poi = None
        if (cache_dir / "poi.npz").exists():
            poi_npz = np.load(cache_dir / "poi.npz", allow_pickle=True)
            poi = POIFeatures(
                x=poi_npz["x"].astype(float),
                feature_names=[str(s) for s in list(poi_npz["feature_names"])],
            )

        graphml_cache = cache_dir / "graph.graphml"
        graphml_path = str(graphml_cache) if graphml_cache.exists() else meta.get("graphml_path")

        return Substrate(
            grid=grid,
            neighbours=neighbours,
            poi=poi,
            graphml_path=graphml_path,
        )


# --- spatial grid integration (M1.2)

def build_grid_from_lonlat_bounds(
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    cell_size_m: float,
):
    """Build a Grid from lon/lat bounds using the spatial grid builder (M1.2).

    Returns motac.substrate.types.Grid.
    """

    b = LonLatBounds(
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
    )
    return build_regular_grid(b, cell_size_m=cell_size_m)
