"""Simulation output containers and serialization helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np


def _kml_color(hex_rgb: str, *, alpha: int = 255) -> str:
    rgb = hex_rgb.lstrip("#")
    if len(rgb) != 6:
        raise ValueError(f"Expected 6-digit RGB hex color, got: {hex_rgb!r}")
    r = int(rgb[0:2], 16)
    g = int(rgb[2:4], 16)
    b = int(rgb[4:6], 16)
    a = int(np.clip(alpha, 0, 255))
    return f"{a:02x}{b:02x}{g:02x}{r:02x}"


def _iso_z(dt: datetime) -> str:
    text = dt.isoformat()
    if text.endswith("+00:00"):
        return text[:-6] + "Z"
    if text.endswith("Z"):
        return text
    return text


def _subsample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=int)
    if n <= max_points:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, num=max_points, dtype=int)


def _kml_linestring_coordinates(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    alt_m: np.ndarray,
    *,
    max_points: int,
) -> str:
    lat = np.asarray(lat_deg, dtype=float).reshape(-1)
    lon = np.asarray(lon_deg, dtype=float).reshape(-1)
    alt = np.asarray(alt_m, dtype=float).reshape(-1)
    if lat.size == 0 or lon.size == 0 or alt.size == 0:
        return ""
    n = min(lat.size, lon.size, alt.size)
    lat = lat[:n]
    lon = lon[:n]
    alt = alt[:n]
    finite_mask = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(alt)
    if not np.any(finite_mask):
        return ""
    lat = lat[finite_mask]
    lon = lon[finite_mask]
    alt = alt[finite_mask]
    idx = _subsample_indices(int(lat.size), max_points)
    lat = lat[idx]
    lon = lon[idx]
    alt = alt[idx]
    return " ".join(f"{lo:.6f},{la:.6f},{al:.1f}" for la, lo, al in zip(lat, lon, alt, strict=False))


def _kml_gx_track_elements(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    alt_m: np.ndarray,
    times: list[datetime] | np.ndarray,
    *,
    max_points: int,
) -> list[str]:
    """Build gx:Track when/coord children for Google Earth time animation."""
    lat = np.asarray(lat_deg, dtype=float).reshape(-1)
    lon = np.asarray(lon_deg, dtype=float).reshape(-1)
    alt = np.asarray(alt_m, dtype=float).reshape(-1)
    if isinstance(times, np.ndarray):
        time_list = list(times)
    else:
        time_list = list(times)
    n = min(lat.size, lon.size, alt.size, len(time_list))
    if n == 0:
        return []
    lat = lat[:n]
    lon = lon[:n]
    alt = alt[:n]
    time_list = time_list[:n]
    finite_mask = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(alt)
    if not np.any(finite_mask):
        return []
    indices = np.nonzero(finite_mask)[0]
    if indices.size > max_points:
        keep = _subsample_indices(int(indices.size), max_points)
        indices = indices[keep]
    lines: list[str] = []
    for i in indices:
        t = time_list[int(i)]
        if not isinstance(t, datetime):
            continue
        lines.append(f"          <when>{escape(_iso_z(t))}</when>")
    for i in indices:
        lines.append(
            f"          <gx:coord>{float(lon[int(i)]):.6f} {float(lat[int(i)]):.6f} {float(alt[int(i)]):.1f}</gx:coord>"
        )
    return lines


@dataclass(slots=True)
class RayResult:
    ray_id: int
    trajectory_local_m: np.ndarray
    trajectory_ecef_m: np.ndarray
    trajectory_geodetic: np.ndarray
    ground_hit: bool
    ground_hit_lat_lon: tuple[float, float] | None
    # Path-length and acoustic travel time along the ray (seconds from emission).
    path_s_m: np.ndarray | None = None
    path_t_rel_s: np.ndarray | None = None
    path_c_eff_mps: np.ndarray | None = None
    launch_dir_ecef: np.ndarray | None = None


@dataclass(slots=True)
class EmissionResult:
    emission_time_utc: datetime
    aircraft_lat_deg: float
    aircraft_lon_deg: float
    aircraft_alt_m: float
    aircraft_position_ecef_m: np.ndarray
    effective_mach: float | None = None
    source_mach_cutoff: bool = False
    mach_cutoff: bool = False
    aircraft_speed_mps: float | None = None
    aircraft_mach: float | None = None
    aircraft_velocity_ecef_mps: np.ndarray | None = None
    rays: list[RayResult] = field(default_factory=list)


@dataclass(slots=True)
class AtmosphericTimeSeries:
    emission_times_utc: list[datetime]
    aircraft_lat_deg: np.ndarray
    aircraft_lon_deg: np.ndarray
    aircraft_alt_m: np.ndarray
    temperature_k: np.ndarray
    relative_humidity_pct: np.ndarray
    pressure_hpa: np.ndarray
    u_wind_mps: np.ndarray
    v_wind_mps: np.ndarray
    sound_speed_mps: np.ndarray
    wind_projection_mps: np.ndarray
    effective_sound_speed_mps: np.ndarray


@dataclass(slots=True)
class AtmosphericVerticalProfile:
    emission_time_utc: datetime
    aircraft_lat_deg: float
    aircraft_lon_deg: float
    aircraft_alt_m: float
    altitude_m: np.ndarray
    temperature_k: np.ndarray
    relative_humidity_pct: np.ndarray
    pressure_hpa: np.ndarray
    u_wind_mps: np.ndarray
    v_wind_mps: np.ndarray
    sound_speed_mps: np.ndarray
    wind_projection_mps: np.ndarray
    effective_sound_speed_mps: np.ndarray


@dataclass(slots=True)
class AtmosphericGrid3D:
    emission_time_utc: datetime
    aircraft_lat_deg: float
    aircraft_lon_deg: float
    aircraft_alt_m: float
    lat_grid_deg: np.ndarray
    lon_grid_deg: np.ndarray
    altitude_m: np.ndarray
    temperature_k: np.ndarray
    relative_humidity_pct: np.ndarray
    pressure_hpa: np.ndarray
    u_wind_mps: np.ndarray
    v_wind_mps: np.ndarray
    sound_speed_mps: np.ndarray
    wind_projection_mps: np.ndarray
    effective_sound_speed_mps: np.ndarray


@dataclass(slots=True)
class PopulationImpactResult:
    dataset_name: str
    dataset_path: str
    heatmap_cell_deg: float
    hit_radius_km: float
    trace_half_width_km: float
    overflight_half_width_km: float
    total_population_in_heatmap: float
    total_exposed_population: float
    total_exposed_area_km2: float
    total_overflight_population: float
    total_overflight_area_km2: float
    heatmap_lat_edges_deg: np.ndarray
    heatmap_lon_edges_deg: np.ndarray
    heatmap_population: np.ndarray
    exposed_cell_mask: np.ndarray
    overflight_cell_mask: np.ndarray
    hit_lat_deg: np.ndarray
    hit_lon_deg: np.ndarray
    hit_emission_index: np.ndarray
    hit_ray_id: np.ndarray
    hit_time_utc: list[str]
    hit_population_within_radius: np.ndarray
    emission_exposed_population: np.ndarray
    emission_exposed_area_km2: np.ndarray
    emission_ground_hit_count: np.ndarray


@dataclass(slots=True)
class SimulationResult:
    emissions: list[EmissionResult]
    config_dict: dict
    terrain_lat_deg: np.ndarray | None = None
    terrain_lon_deg: np.ndarray | None = None
    terrain_elevation_m: np.ndarray | None = None
    atmospheric_time_series: AtmosphericTimeSeries | None = None
    atmospheric_vertical_profile: AtmosphericVerticalProfile | None = None
    atmospheric_grid_3d: AtmosphericGrid3D | None = None
    population_impact: PopulationImpactResult | None = None

    def all_ground_hits(self):
        lats = []
        lons = []
        times = []
        for emission in self.emissions:
            for ray in emission.rays:
                if ray.ground_hit and ray.ground_hit_lat_lon is not None:
                    lat, lon = ray.ground_hit_lat_lon
                    lats.append(lat)
                    lons.append(lon)
                    times.append(emission.emission_time_utc.isoformat())
        return np.asarray(lats), np.asarray(lons), times

    def boom_envelope_geodetic_points(
        self,
        *,
        max_points_per_ray: int = 12,
        max_total_points: int = 8_000,
        include_flight_track: bool = True,
    ) -> np.ndarray:
        """Return sampled geodetic points approximating the full boom propagation envelope.

        Columns are ordered as ``[lat_deg, lon_deg, alt_m]``.
        """
        if max_points_per_ray < 2:
            raise ValueError("max_points_per_ray must be >= 2")
        if max_total_points < 4:
            raise ValueError("max_total_points must be >= 4")

        point_sets: list[np.ndarray] = []

        if include_flight_track and self.emissions:
            flight_points = np.asarray(
                [
                    [e.aircraft_lat_deg, e.aircraft_lon_deg, e.aircraft_alt_m]
                    for e in self.emissions
                ],
                dtype=float,
            )
            flight_points = flight_points[np.all(np.isfinite(flight_points), axis=1)]
            if flight_points.size:
                point_sets.append(flight_points)

        for emission in self.emissions:
            for ray in emission.rays:
                g = np.asarray(ray.trajectory_geodetic, dtype=float)
                if g.ndim != 2 or g.shape[1] < 3 or g.shape[0] == 0:
                    continue
                g = g[:, :3]
                finite_mask = np.all(np.isfinite(g), axis=1)
                if not np.any(finite_mask):
                    continue
                g = g[finite_mask]
                if g.shape[0] > max_points_per_ray:
                    idx = np.linspace(0, g.shape[0] - 1, num=max_points_per_ray, dtype=int)
                    g = g[idx]
                point_sets.append(g)

        if not point_sets:
            return np.empty((0, 3), dtype=float)

        points = np.vstack(point_sets)
        if points.shape[0] <= max_total_points:
            return np.unique(points, axis=0)

        keep_idx: set[int] = set()
        for axis in range(points.shape[1]):
            keep_idx.add(int(np.argmin(points[:, axis])))
            keep_idx.add(int(np.argmax(points[:, axis])))

        if len(keep_idx) >= max_total_points:
            ordered_keep = np.asarray(sorted(keep_idx), dtype=int)[:max_total_points]
            return np.unique(points[ordered_keep], axis=0)

        stride_count = max_total_points - len(keep_idx)
        sample_idx = np.linspace(0, points.shape[0] - 1, num=stride_count, dtype=int)
        keep_idx.update(int(i) for i in sample_idx.tolist())
        ordered_idx = np.asarray(sorted(keep_idx), dtype=int)[:max_total_points]
        return np.unique(points[ordered_idx], axis=0)

    def save_json_summary(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "num_emissions": len(self.emissions),
            "num_rays": int(sum(len(e.rays) for e in self.emissions)),
            "num_ground_hits": int(sum(1 for e in self.emissions for r in e.rays if r.ground_hit)),
            "num_cutoff_emissions": int(sum(1 for e in self.emissions if e.mach_cutoff)),
            "num_source_cutoff_emissions": int(sum(1 for e in self.emissions if e.source_mach_cutoff)),
            "config": self.config_dict,
        }
        summary["cutoff_achieved"] = bool(summary["num_cutoff_emissions"] > 0)
        summary["full_route_cutoff"] = bool(summary["num_cutoff_emissions"] == summary["num_emissions"])
        summary["source_cutoff_achieved"] = bool(summary["num_source_cutoff_emissions"] > 0)

        effective_machs = [float(e.effective_mach) for e in self.emissions if e.effective_mach is not None]
        if effective_machs:
            summary["effective_mach_min"] = float(np.min(effective_machs))
            summary["effective_mach_max"] = float(np.max(effective_machs))
            summary["effective_mach_mean"] = float(np.mean(effective_machs))

        if summary["num_rays"] > 0:
            summary["ground_hit_fraction"] = float(summary["num_ground_hits"] / summary["num_rays"])

        if self.atmospheric_time_series is not None:
            summary["num_atmospheric_samples"] = int(len(self.atmospheric_time_series.emission_times_utc))
        if self.atmospheric_vertical_profile is not None:
            summary["num_atmospheric_profile_levels"] = int(self.atmospheric_vertical_profile.altitude_m.size)
        if self.atmospheric_grid_3d is not None:
            summary["atmospheric_grid_shape"] = [int(v) for v in self.atmospheric_grid_3d.temperature_k.shape]

        if self.population_impact is not None:
            pop = self.population_impact
            pop_summary: dict[str, object] = {
                "dataset_name": pop.dataset_name,
                "dataset_path": pop.dataset_path,
                "heatmap_cell_deg": float(pop.heatmap_cell_deg),
                "hit_radius_km": float(pop.hit_radius_km),
                "trace_half_width_km": float(pop.trace_half_width_km),
                "overflight_half_width_km": float(pop.overflight_half_width_km),
                "total_population_in_heatmap": float(pop.total_population_in_heatmap),
                "total_exposed_population": float(pop.total_exposed_population),
                "total_exposed_area_km2": float(pop.total_exposed_area_km2),
                "total_overflight_population": float(pop.total_overflight_population),
                "total_overflight_area_km2": float(pop.total_overflight_area_km2),
            }

            if pop.hit_population_within_radius.size:
                ranked = np.argsort(pop.hit_population_within_radius)[::-1]
                top_hits = []
                for rank_i in ranked[:10]:
                    idx = int(rank_i)
                    top_hits.append(
                        {
                            "hit_index": idx,
                            "emission_index": int(pop.hit_emission_index[idx]),
                            "ray_id": int(pop.hit_ray_id[idx]),
                            "time_utc": pop.hit_time_utc[idx],
                            "lat_deg": float(pop.hit_lat_deg[idx]),
                            "lon_deg": float(pop.hit_lon_deg[idx]),
                            "population_within_radius": float(pop.hit_population_within_radius[idx]),
                        }
                    )
                pop_summary["top_impacted_ground_hits"] = top_hits
                pop_summary["max_ground_hit_population_within_radius"] = float(
                    np.max(pop.hit_population_within_radius)
                )

            if pop.emission_exposed_population.size:
                best_emission_idx = int(np.argmax(pop.emission_exposed_population))
                best_time = ""
                if 0 <= best_emission_idx < len(self.emissions):
                    best_time = self.emissions[best_emission_idx].emission_time_utc.isoformat()
                pop_summary["max_exposed_emission_index"] = best_emission_idx
                pop_summary["max_exposed_emission_time_utc"] = best_time
                pop_summary["max_exposed_emission_population"] = float(
                    pop.emission_exposed_population[best_emission_idx]
                )
                pop_summary["mean_exposed_population_per_emission"] = float(np.mean(pop.emission_exposed_population))
                pop_summary["num_emissions_with_exposed_population"] = int(
                    np.sum(pop.emission_exposed_population > 0.0)
                )

            summary["population_analysis"] = pop_summary
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def save_npz(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lat_hits, lon_hits, _ = self.all_ground_hits()
        payload = dict(
            hit_lat_deg=lat_hits,
            hit_lon_deg=lon_hits,
            num_emissions=len(self.emissions),
            num_rays=sum(len(e.rays) for e in self.emissions),
        )
        if self.atmospheric_time_series is not None:
            payload.update(
                atmospheric_temperature_k=self.atmospheric_time_series.temperature_k,
                atmospheric_relative_humidity_pct=self.atmospheric_time_series.relative_humidity_pct,
                atmospheric_pressure_hpa=self.atmospheric_time_series.pressure_hpa,
                atmospheric_u_wind_mps=self.atmospheric_time_series.u_wind_mps,
                atmospheric_v_wind_mps=self.atmospheric_time_series.v_wind_mps,
                atmospheric_effective_sound_speed_mps=self.atmospheric_time_series.effective_sound_speed_mps,
            )
        if self.atmospheric_vertical_profile is not None:
            payload.update(
                atmospheric_profile_altitude_m=self.atmospheric_vertical_profile.altitude_m,
                atmospheric_profile_temperature_k=self.atmospheric_vertical_profile.temperature_k,
                atmospheric_profile_relative_humidity_pct=self.atmospheric_vertical_profile.relative_humidity_pct,
                atmospheric_profile_pressure_hpa=self.atmospheric_vertical_profile.pressure_hpa,
                atmospheric_profile_u_wind_mps=self.atmospheric_vertical_profile.u_wind_mps,
                atmospheric_profile_v_wind_mps=self.atmospheric_vertical_profile.v_wind_mps,
                atmospheric_profile_effective_sound_speed_mps=self.atmospheric_vertical_profile.effective_sound_speed_mps,
            )
        if self.population_impact is not None:
            pop = self.population_impact
            payload.update(
                population_dataset_name=np.asarray([pop.dataset_name], dtype="<U256"),
                population_dataset_path=np.asarray([pop.dataset_path], dtype="<U512"),
                population_heatmap_cell_deg=np.asarray([pop.heatmap_cell_deg], dtype=np.float32),
                population_hit_radius_km=np.asarray([pop.hit_radius_km], dtype=np.float32),
                population_trace_half_width_km=np.asarray([pop.trace_half_width_km], dtype=np.float32),
                population_overflight_half_width_km=np.asarray([pop.overflight_half_width_km], dtype=np.float32),
                population_total_in_heatmap=np.asarray([pop.total_population_in_heatmap], dtype=np.float64),
                population_total_exposed=np.asarray([pop.total_exposed_population], dtype=np.float64),
                population_total_exposed_area_km2=np.asarray([pop.total_exposed_area_km2], dtype=np.float64),
                population_total_overflight=np.asarray([pop.total_overflight_population], dtype=np.float64),
                population_total_overflight_area_km2=np.asarray([pop.total_overflight_area_km2], dtype=np.float64),
                population_heatmap_lat_edges_deg=np.asarray(pop.heatmap_lat_edges_deg, dtype=np.float32),
                population_heatmap_lon_edges_deg=np.asarray(pop.heatmap_lon_edges_deg, dtype=np.float32),
                population_heatmap_people=np.asarray(pop.heatmap_population, dtype=np.float32),
                population_exposed_cell_mask=np.asarray(pop.exposed_cell_mask, dtype=np.uint8),
                population_overflight_cell_mask=np.asarray(pop.overflight_cell_mask, dtype=np.uint8),
                population_hit_lat_deg=np.asarray(pop.hit_lat_deg, dtype=np.float32),
                population_hit_lon_deg=np.asarray(pop.hit_lon_deg, dtype=np.float32),
                population_hit_emission_index=np.asarray(pop.hit_emission_index, dtype=np.int32),
                population_hit_ray_id=np.asarray(pop.hit_ray_id, dtype=np.int32),
                population_hit_time_utc=np.asarray(pop.hit_time_utc, dtype="<U64"),
                population_hit_people_within_radius=np.asarray(pop.hit_population_within_radius, dtype=np.float32),
                population_emission_exposed_people=np.asarray(pop.emission_exposed_population, dtype=np.float32),
                population_emission_exposed_area_km2=np.asarray(pop.emission_exposed_area_km2, dtype=np.float32),
                population_emission_ground_hit_count=np.asarray(pop.emission_ground_hit_count, dtype=np.int32),
            )
        np.savez_compressed(path, **payload)

    def save_kml(
        self,
        path: str | Path,
        *,
        max_rays_per_emission: int | None = None,
        max_points_per_ray: int | None = None,
        max_ground_hits: int | None = None,
    ):
        """Write Google Earth-compatible KML overlay for the current simulation.

        By default **all** rays and ground hits are written. Pass positive ints to
        cap rays-per-emission or ground hits for lighter KML files. Point sampling
        defaults to 220 points for timed tracks and all points for static paths.
        """
        if max_rays_per_emission is not None and max_rays_per_emission <= 0:
            raise ValueError("max_rays_per_emission must be positive when set")
        timed_tracks = self.config_dict.get("visualization", {}).get("kml_timed_tracks", True)
        if max_points_per_ray is None and timed_tracks:
            max_points_per_ray = 220
        if max_points_per_ray is not None and max_points_per_ray <= 1:
            raise ValueError("max_points_per_ray must be > 1")
        if max_ground_hits is not None and max_ground_hits <= 0:
            raise ValueError("max_ground_hits must be positive when set")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        gx_namespace = ' xmlns:gx="http://www.google.com/kml/ext/2.2"' if timed_tracks else ""
        lines: list[str] = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<kml xmlns="http://www.opengis.net/kml/2.2"{gx_namespace}>',
            "  <Document>",
            "    <name>Mach Cutoff Simulation Overlay</name>",
            "    <Style id=\"flightStyle\">",
            "      <IconStyle>",
            "        <scale>0</scale>",
            "      </IconStyle>",
            "      <LineStyle>",
            f"        <color>{_kml_color('#111111', alpha=255)}</color>",
            "        <width>3</width>",
            "      </LineStyle>",
            "    </Style>",
            "    <Style id=\"rayStyle\">",
            "      <IconStyle>",
            "        <scale>0</scale>",
            "      </IconStyle>",
            "      <LineStyle>",
            f"        <color>{_kml_color('#1f5dff', alpha=110)}</color>",
            "        <width>1</width>",
            "      </LineStyle>",
            "    </Style>",
            "    <Style id=\"groundHitStyle\">",
            "      <IconStyle>",
            f"        <color>{_kml_color('#d62828', alpha=230)}</color>",
            "        <scale>0.65</scale>",
            "      </IconStyle>",
            "      <LabelStyle>",
            "        <scale>0</scale>",
            "      </LabelStyle>",
            "    </Style>",
        ]

        if self.emissions:
            flight_lat = np.asarray([e.aircraft_lat_deg for e in self.emissions], dtype=float)
            flight_lon = np.asarray([e.aircraft_lon_deg for e in self.emissions], dtype=float)
            flight_alt = np.asarray([e.aircraft_alt_m for e in self.emissions], dtype=float)
            track_elems = []
            if timed_tracks:
                flight_times = [e.emission_time_utc for e in self.emissions]
                track_elems = _kml_gx_track_elements(
                    flight_lat,
                    flight_lon,
                    flight_alt,
                    flight_times,
                    max_points=max(2, int(flight_lat.size)),
                )
            # Static line as fallback for clients without gx:Track scrubbing.
            flight_coords = _kml_linestring_coordinates(
                flight_lat,
                flight_lon,
                flight_alt,
                max_points=max(2, int(flight_lat.size)),
            )
            lines.extend(["    <Folder>", "      <name>Flight</name>"])
            if track_elems:
                lines.extend(
                    [
                        "      <Placemark>",
                        "        <name>Aircraft track (timed)</name>",
                        "        <styleUrl>#flightStyle</styleUrl>",
                        "        <gx:Track>",
                        "          <altitudeMode>absolute</altitudeMode>",
                        *track_elems,
                        "        </gx:Track>",
                        "      </Placemark>",
                    ]
                )
            if flight_coords:
                lines.extend(
                    [
                        "      <Placemark>",
                        "        <name>Aircraft track (path)</name>",
                        "        <styleUrl>#flightStyle</styleUrl>",
                        "        <LineString>",
                        "          <tessellate>1</tessellate>",
                        "          <altitudeMode>absolute</altitudeMode>",
                        f"          <coordinates>{flight_coords}</coordinates>",
                        "        </LineString>",
                        "      </Placemark>",
                    ]
                )
            lines.append("    </Folder>")

        ray_count = 0
        lines.extend(["    <Folder>", "      <name>Shock Rays (sampled)</name>"])
        from datetime import timedelta

        for emission_idx, emission in enumerate(self.emissions):
            rays_iter = (
                emission.rays
                if max_rays_per_emission is None
                else emission.rays[: int(max_rays_per_emission)]
            )
            for ray in rays_iter:
                g = np.asarray(ray.trajectory_geodetic, dtype=float)
                if g.ndim != 2 or g.shape[0] == 0 or g.shape[1] < 3:
                    continue
                n_pts = int(g.shape[0])
                if ray.path_t_rel_s is not None and np.asarray(ray.path_t_rel_s).size == n_pts:
                    t_rel = np.asarray(ray.path_t_rel_s, dtype=float).reshape(-1)
                else:
                    t_rel = np.zeros(n_pts, dtype=float)
                track_elems = []
                if timed_tracks:
                    ray_times = [emission.emission_time_utc + timedelta(seconds=float(t)) for t in t_rel]
                    track_elems = _kml_gx_track_elements(
                        g[:, 0],
                        g[:, 1],
                        g[:, 2],
                        ray_times,
                        max_points=max_points_per_ray or n_pts,
                    )
                coords = _kml_linestring_coordinates(
                    g[:, 0],
                    g[:, 1],
                    g[:, 2],
                    max_points=max_points_per_ray or n_pts,
                )
                if not track_elems and not coords:
                    continue
                ray_count += 1
                ray_name = f"Emission {emission_idx:03d} / Ray {int(ray.ray_id):03d}"
                t_end = float(t_rel[-1]) if t_rel.size else 0.0
                description = (
                    f"ground_hit={bool(ray.ground_hit)}; "
                    f"emission={_iso_z(emission.emission_time_utc)}; "
                    f"path_time_s={t_end:.3f}"
                )
                lines.extend(
                    [
                        "      <Placemark>",
                        f"        <name>{escape(ray_name)}</name>",
                        f"        <description>{escape(description)}</description>",
                        "        <styleUrl>#rayStyle</styleUrl>",
                    ]
                )
                if track_elems:
                    lines.extend(
                        [
                            "        <gx:Track>",
                            "          <altitudeMode>absolute</altitudeMode>",
                            *track_elems,
                            "        </gx:Track>",
                        ]
                    )
                elif coords:
                    lines.extend(
                        [
                            "        <LineString>",
                            "          <tessellate>1</tessellate>",
                            "          <altitudeMode>absolute</altitudeMode>",
                            f"          <coordinates>{coords}</coordinates>",
                            "        </LineString>",
                        ]
                    )
                lines.append("      </Placemark>")
        lines.extend(["    </Folder>"])

        # Ground hits with TimeStamp (emission + ray TOF when available).
        hit_entries: list[tuple[float, float, str]] = []
        for emission in self.emissions:
            for ray in emission.rays:
                if not ray.ground_hit or ray.ground_hit_lat_lon is None:
                    continue
                lat_h, lon_h = ray.ground_hit_lat_lon
                t_rel = 0.0
                if ray.path_t_rel_s is not None and np.asarray(ray.path_t_rel_s).size:
                    t_rel = float(np.asarray(ray.path_t_rel_s).reshape(-1)[-1])
                t_hit = emission.emission_time_utc + timedelta(seconds=t_rel)
                hit_entries.append((float(lat_h), float(lon_h), _iso_z(t_hit)))
        if hit_entries:
            n_hits = len(hit_entries) if max_ground_hits is None else min(len(hit_entries), int(max_ground_hits))
            lines.extend(["    <Folder>", "      <name>Ground Hits</name>"])
            for i in range(n_hits):
                lat_h, lon_h, t_str = hit_entries[i]
                lines.extend(
                    [
                        "      <Placemark>",
                        f"        <name>Hit {i + 1:04d}</name>",
                        f"        <description>{escape(t_str)}</description>",
                        "        <TimeStamp>",
                        f"          <when>{escape(t_str)}</when>",
                        "        </TimeStamp>",
                        "        <styleUrl>#groundHitStyle</styleUrl>",
                        "        <Point>",
                        "          <altitudeMode>clampToGround</altitudeMode>",
                        f"          <coordinates>{lon_h:.6f},{lat_h:.6f},0</coordinates>",
                        "        </Point>",
                        "      </Placemark>",
                    ]
                )
            lines.extend(["    </Folder>"])
        lat_hits = np.asarray([h[0] for h in hit_entries], dtype=float)

        summary = f"emissions={len(self.emissions)}, sampled_rays={ray_count}, ground_hits={int(lat_hits.size)}"
        if timed_tracks:
            summary += "; track times include acoustic travel time"
        lines.insert(4, f"    <description>{escape(summary)}</description>")
        lines.extend(["  </Document>", "</kml>", ""])

        path.write_text("\n".join(lines), encoding="utf-8")
