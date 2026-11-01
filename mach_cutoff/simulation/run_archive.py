"""mach_cutoff_run_v1 multi-table CSV archive for full sim reconstruction."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .outputs import SimulationResult

FORMAT_NAME = "mach_cutoff_run_v1"
SCHEMA_VERSION = 1


def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    text = dt.isoformat().replace("+00:00", "Z")
    if text.endswith("Z"):
        return text
    return text + "Z" if "+" not in text[-6:] else text


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_run_archive(
    result: SimulationResult,
    output_dir: str | Path,
    *,
    run_id: str | None = None,
    config_dict: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write a reconstructable run package (CSV tables + JSON manifest/config).

    Returns a map of logical name → path for written files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = config_dict if config_dict is not None else result.config_dict
    created = datetime.now(timezone.utc)
    rid = run_id or out.name or f"run_{created.strftime('%Y%m%dT%H%M%SZ')}"

    written: dict[str, Path] = {}

    # --- config ---
    config_path = out / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)
    written["config"] = config_path

    # --- flight emissions ---
    emission_rows: list[dict[str, Any]] = []
    for emission_id, emission in enumerate(result.emissions):
        ecef = np.asarray(emission.aircraft_position_ecef_m, dtype=float).reshape(-1)
        vel = emission.aircraft_velocity_ecef_mps
        if vel is None:
            vx = vy = vz = ""
        else:
            v = np.asarray(vel, dtype=float).reshape(-1)
            vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        n_rays = len(emission.rays)
        n_hits = sum(1 for r in emission.rays if r.ground_hit)
        emission_rows.append(
            {
                "emission_id": emission_id,
                "time_utc": _iso_utc(emission.emission_time_utc),
                "lat_deg": float(emission.aircraft_lat_deg),
                "lon_deg": float(emission.aircraft_lon_deg),
                "alt_m": float(emission.aircraft_alt_m),
                "ecef_x_m": float(ecef[0]) if ecef.size >= 3 else "",
                "ecef_y_m": float(ecef[1]) if ecef.size >= 3 else "",
                "ecef_z_m": float(ecef[2]) if ecef.size >= 3 else "",
                "speed_mps": "" if emission.aircraft_speed_mps is None else float(emission.aircraft_speed_mps),
                "mach": "" if emission.aircraft_mach is None else float(emission.aircraft_mach),
                "velocity_ecef_x_mps": vx,
                "velocity_ecef_y_mps": vy,
                "velocity_ecef_z_mps": vz,
                "effective_mach": "" if emission.effective_mach is None else float(emission.effective_mach),
                "source_mach_cutoff": int(bool(emission.source_mach_cutoff)),
                "mach_cutoff": int(bool(emission.mach_cutoff)),
                "num_rays": int(n_rays),
                "num_ground_hits": int(n_hits),
            }
        )
    flight_path = out / "flight_emissions.csv"
    _write_csv(
        flight_path,
        [
            "emission_id",
            "time_utc",
            "lat_deg",
            "lon_deg",
            "alt_m",
            "ecef_x_m",
            "ecef_y_m",
            "ecef_z_m",
            "speed_mps",
            "mach",
            "velocity_ecef_x_mps",
            "velocity_ecef_y_mps",
            "velocity_ecef_z_mps",
            "effective_mach",
            "source_mach_cutoff",
            "mach_cutoff",
            "num_rays",
            "num_ground_hits",
        ],
        emission_rows,
    )
    written["flight_emissions"] = flight_path

    # --- rays + samples ---
    ray_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    num_ray_samples = 0
    for emission_id, emission in enumerate(result.emissions):
        t0 = emission.emission_time_utc
        for ray in emission.rays:
            g = np.asarray(ray.trajectory_geodetic, dtype=float)
            local = np.asarray(ray.trajectory_local_m, dtype=float)
            ecef = np.asarray(ray.trajectory_ecef_m, dtype=float)
            n_samp = int(g.shape[0]) if g.ndim == 2 else 0
            path_s = ray.path_s_m
            path_t = ray.path_t_rel_s
            path_c = ray.path_c_eff_mps
            if path_s is None or np.asarray(path_s).size != n_samp:
                path_s = np.zeros(n_samp, dtype=np.float64)
                if n_samp > 1 and local.ndim == 2 and local.shape[0] == n_samp:
                    diffs = np.linalg.norm(np.diff(local[:, :3], axis=0), axis=1)
                    path_s[1:] = np.cumsum(diffs)
            else:
                path_s = np.asarray(path_s, dtype=np.float64).reshape(-1)
            if path_t is None or np.asarray(path_t).size != n_samp:
                path_t = np.zeros(n_samp, dtype=np.float64)
            else:
                path_t = np.asarray(path_t, dtype=np.float64).reshape(-1)
            if path_c is None or np.asarray(path_c).size != n_samp:
                path_c = np.full(n_samp, np.nan, dtype=np.float64)
            else:
                path_c = np.asarray(path_c, dtype=np.float64).reshape(-1)

            hit_lat = hit_lon = ""
            if ray.ground_hit and ray.ground_hit_lat_lon is not None:
                hit_lat = float(ray.ground_hit_lat_lon[0])
                hit_lon = float(ray.ground_hit_lat_lon[1])
            terminal_alt = float(g[-1, 2]) if n_samp > 0 else ""
            launch = ray.launch_dir_ecef
            if launch is None:
                lx = ly = lz = ""
            else:
                lv = np.asarray(launch, dtype=float).reshape(-1)
                lx, ly, lz = float(lv[0]), float(lv[1]), float(lv[2])

            ray_rows.append(
                {
                    "emission_id": emission_id,
                    "ray_id": int(ray.ray_id),
                    "ground_hit": int(bool(ray.ground_hit)),
                    "hit_lat_deg": hit_lat,
                    "hit_lon_deg": hit_lon,
                    "num_samples": n_samp,
                    "terminal_alt_m": terminal_alt,
                    "launch_dir_ecef_x": lx,
                    "launch_dir_ecef_y": ly,
                    "launch_dir_ecef_z": lz,
                    "path_time_final_s": float(path_t[-1]) if n_samp else "",
                    "path_length_final_m": float(path_s[-1]) if n_samp else "",
                }
            )

            for i in range(n_samp):
                t_rel = float(path_t[i])
                # Absolute sample time = emission + relative TOF
                t_abs = t0 + timedelta(seconds=t_rel)
                enu_e = enu_n = enu_u = ""
                if local.ndim == 2 and local.shape[0] > i and local.shape[1] >= 3:
                    enu_e, enu_n, enu_u = float(local[i, 0]), float(local[i, 1]), float(local[i, 2])
                ex = ey = ez = ""
                if ecef.ndim == 2 and ecef.shape[0] > i and ecef.shape[1] >= 3:
                    ex, ey, ez = float(ecef[i, 0]), float(ecef[i, 1]), float(ecef[i, 2])
                sample_rows.append(
                    {
                        "emission_id": emission_id,
                        "ray_id": int(ray.ray_id),
                        "sample_i": i,
                        "s_m": float(path_s[i]),
                        "t_rel_s": t_rel,
                        "t_utc": _iso_utc(t_abs),
                        "lat_deg": float(g[i, 0]),
                        "lon_deg": float(g[i, 1]),
                        "alt_m": float(g[i, 2]),
                        "ecef_x_m": ex,
                        "ecef_y_m": ey,
                        "ecef_z_m": ez,
                        "enu_e_m": enu_e,
                        "enu_n_m": enu_n,
                        "enu_u_m": enu_u,
                        "c_eff_mps": "" if not np.isfinite(path_c[i]) else float(path_c[i]),
                    }
                )
                num_ray_samples += 1

    rays_path = out / "rays.csv"
    _write_csv(
        rays_path,
        [
            "emission_id",
            "ray_id",
            "ground_hit",
            "hit_lat_deg",
            "hit_lon_deg",
            "num_samples",
            "terminal_alt_m",
            "launch_dir_ecef_x",
            "launch_dir_ecef_y",
            "launch_dir_ecef_z",
            "path_time_final_s",
            "path_length_final_m",
        ],
        ray_rows,
    )
    written["rays"] = rays_path

    samples_path = out / "ray_samples.csv"
    _write_csv(
        samples_path,
        [
            "emission_id",
            "ray_id",
            "sample_i",
            "s_m",
            "t_rel_s",
            "t_utc",
            "lat_deg",
            "lon_deg",
            "alt_m",
            "ecef_x_m",
            "ecef_y_m",
            "ecef_z_m",
            "enu_e_m",
            "enu_n_m",
            "enu_u_m",
            "c_eff_mps",
        ],
        sample_rows,
    )
    written["ray_samples"] = samples_path

    # --- atmosphere along track ---
    atmo_rows: list[dict[str, Any]] = []
    ats = result.atmospheric_time_series
    if ats is not None:
        n = len(ats.emission_times_utc)
        for i in range(n):
            atmo_rows.append(
                {
                    "emission_id": i if i < len(result.emissions) else i,
                    "time_utc": _iso_utc(ats.emission_times_utc[i]),
                    "lat_deg": float(ats.aircraft_lat_deg[i]),
                    "lon_deg": float(ats.aircraft_lon_deg[i]),
                    "alt_m": float(ats.aircraft_alt_m[i]),
                    "temperature_k": float(ats.temperature_k[i]),
                    "relative_humidity_pct": float(ats.relative_humidity_pct[i]),
                    "pressure_hpa": float(ats.pressure_hpa[i]),
                    "u_wind_mps": float(ats.u_wind_mps[i]),
                    "v_wind_mps": float(ats.v_wind_mps[i]),
                    "sound_speed_mps": float(ats.sound_speed_mps[i]),
                    "wind_projection_mps": float(ats.wind_projection_mps[i]),
                    "effective_sound_speed_mps": float(ats.effective_sound_speed_mps[i]),
                }
            )
    atmo_path = out / "atmosphere_along_track.csv"
    _write_csv(
        atmo_path,
        [
            "emission_id",
            "time_utc",
            "lat_deg",
            "lon_deg",
            "alt_m",
            "temperature_k",
            "relative_humidity_pct",
            "pressure_hpa",
            "u_wind_mps",
            "v_wind_mps",
            "sound_speed_mps",
            "wind_projection_mps",
            "effective_sound_speed_mps",
        ],
        atmo_rows,
    )
    written["atmosphere_along_track"] = atmo_path

    # --- vertical profile (long format) ---
    prof = result.atmospheric_vertical_profile
    prof_rows: list[dict[str, Any]] = []
    if prof is not None:
        alts = np.asarray(prof.altitude_m, dtype=float).reshape(-1)
        for level_i, alt in enumerate(alts):
            prof_rows.append(
                {
                    "profile_id": 0,
                    "emission_time_utc": _iso_utc(prof.emission_time_utc),
                    "aircraft_lat_deg": float(prof.aircraft_lat_deg),
                    "aircraft_lon_deg": float(prof.aircraft_lon_deg),
                    "aircraft_alt_m": float(prof.aircraft_alt_m),
                    "level_i": level_i,
                    "alt_m": float(alt),
                    "temperature_k": float(prof.temperature_k[level_i]),
                    "relative_humidity_pct": float(prof.relative_humidity_pct[level_i]),
                    "pressure_hpa": float(prof.pressure_hpa[level_i]),
                    "u_wind_mps": float(prof.u_wind_mps[level_i]),
                    "v_wind_mps": float(prof.v_wind_mps[level_i]),
                    "sound_speed_mps": float(prof.sound_speed_mps[level_i]),
                    "wind_projection_mps": float(prof.wind_projection_mps[level_i]),
                    "effective_sound_speed_mps": float(prof.effective_sound_speed_mps[level_i]),
                }
            )
    prof_path = out / "atmosphere_profile.csv"
    _write_csv(
        prof_path,
        [
            "profile_id",
            "emission_time_utc",
            "aircraft_lat_deg",
            "aircraft_lon_deg",
            "aircraft_alt_m",
            "level_i",
            "alt_m",
            "temperature_k",
            "relative_humidity_pct",
            "pressure_hpa",
            "u_wind_mps",
            "v_wind_mps",
            "sound_speed_mps",
            "wind_projection_mps",
            "effective_sound_speed_mps",
        ],
        prof_rows,
    )
    written["atmosphere_profile"] = prof_path

    # --- atmosphere grid (dense) ---
    grid = result.atmospheric_grid_3d
    if grid is not None:
        grid_path = out / "atmosphere_grid.npz"
        np.savez_compressed(
            grid_path,
            emission_time_utc=np.asarray([_iso_utc(grid.emission_time_utc)], dtype="<U64"),
            aircraft_lat_deg=np.asarray([grid.aircraft_lat_deg], dtype=np.float64),
            aircraft_lon_deg=np.asarray([grid.aircraft_lon_deg], dtype=np.float64),
            aircraft_alt_m=np.asarray([grid.aircraft_alt_m], dtype=np.float64),
            lat_grid_deg=np.asarray(grid.lat_grid_deg, dtype=np.float32),
            lon_grid_deg=np.asarray(grid.lon_grid_deg, dtype=np.float32),
            altitude_m=np.asarray(grid.altitude_m, dtype=np.float32),
            temperature_k=np.asarray(grid.temperature_k, dtype=np.float32),
            relative_humidity_pct=np.asarray(grid.relative_humidity_pct, dtype=np.float32),
            pressure_hpa=np.asarray(grid.pressure_hpa, dtype=np.float32),
            u_wind_mps=np.asarray(grid.u_wind_mps, dtype=np.float32),
            v_wind_mps=np.asarray(grid.v_wind_mps, dtype=np.float32),
            sound_speed_mps=np.asarray(grid.sound_speed_mps, dtype=np.float32),
            wind_projection_mps=np.asarray(grid.wind_projection_mps, dtype=np.float32),
            effective_sound_speed_mps=np.asarray(grid.effective_sound_speed_mps, dtype=np.float32),
        )
        written["atmosphere_grid"] = grid_path

    # --- population ---
    pop = result.population_impact
    if pop is not None:
        pop_summary = {
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
        pop_json = out / "population_summary.json"
        with pop_json.open("w", encoding="utf-8") as f:
            json.dump(pop_summary, f, indent=2)
        written["population_summary"] = pop_json

        pop_npz = out / "population_heatmap.npz"
        np.savez_compressed(
            pop_npz,
            heatmap_lat_edges_deg=np.asarray(pop.heatmap_lat_edges_deg, dtype=np.float32),
            heatmap_lon_edges_deg=np.asarray(pop.heatmap_lon_edges_deg, dtype=np.float32),
            heatmap_population=np.asarray(pop.heatmap_population, dtype=np.float32),
            exposed_cell_mask=np.asarray(pop.exposed_cell_mask, dtype=np.uint8),
            overflight_cell_mask=np.asarray(pop.overflight_cell_mask, dtype=np.uint8),
            hit_lat_deg=np.asarray(pop.hit_lat_deg, dtype=np.float32),
            hit_lon_deg=np.asarray(pop.hit_lon_deg, dtype=np.float32),
            hit_emission_index=np.asarray(pop.hit_emission_index, dtype=np.int32),
            hit_ray_id=np.asarray(pop.hit_ray_id, dtype=np.int32),
            hit_time_utc=np.asarray(pop.hit_time_utc, dtype="<U64"),
            hit_population_within_radius=np.asarray(pop.hit_population_within_radius, dtype=np.float32),
            emission_exposed_population=np.asarray(pop.emission_exposed_population, dtype=np.float32),
            emission_exposed_area_km2=np.asarray(pop.emission_exposed_area_km2, dtype=np.float32),
            emission_ground_hit_count=np.asarray(pop.emission_ground_hit_count, dtype=np.int32),
        )
        written["population_heatmap"] = pop_npz

        hit_rows = []
        for i in range(int(pop.hit_lat_deg.size)):
            hit_rows.append(
                {
                    "hit_index": i,
                    "emission_id": int(pop.hit_emission_index[i]),
                    "ray_id": int(pop.hit_ray_id[i]),
                    "time_utc": pop.hit_time_utc[i],
                    "lat_deg": float(pop.hit_lat_deg[i]),
                    "lon_deg": float(pop.hit_lon_deg[i]),
                    "population_within_radius": float(pop.hit_population_within_radius[i]),
                }
            )
        pop_hits = out / "population_hits.csv"
        _write_csv(
            pop_hits,
            [
                "hit_index",
                "emission_id",
                "ray_id",
                "time_utc",
                "lat_deg",
                "lon_deg",
                "population_within_radius",
            ],
            hit_rows,
        )
        written["population_hits"] = pop_hits

    # Convenience ground hits table
    hit_rows2: list[dict[str, Any]] = []
    for emission_id, emission in enumerate(result.emissions):
        for ray in emission.rays:
            if not ray.ground_hit or ray.ground_hit_lat_lon is None:
                continue
            t_final = 0.0
            if ray.path_t_rel_s is not None and np.asarray(ray.path_t_rel_s).size:
                t_final = float(np.asarray(ray.path_t_rel_s).reshape(-1)[-1])
            t_hit = emission.emission_time_utc + timedelta(seconds=t_final)
            hit_rows2.append(
                {
                    "emission_id": emission_id,
                    "ray_id": int(ray.ray_id),
                    "emission_time_utc": _iso_utc(emission.emission_time_utc),
                    "hit_time_utc": _iso_utc(t_hit),
                    "t_rel_s": t_final,
                    "lat_deg": float(ray.ground_hit_lat_lon[0]),
                    "lon_deg": float(ray.ground_hit_lat_lon[1]),
                }
            )
    gh_path = out / "ground_hits.csv"
    _write_csv(
        gh_path,
        [
            "emission_id",
            "ray_id",
            "emission_time_utc",
            "hit_time_utc",
            "t_rel_s",
            "lat_deg",
            "lon_deg",
        ],
        hit_rows2,
    )
    written["ground_hits"] = gh_path

    manifest = {
        "format": FORMAT_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_id": rid,
        "created_utc": _iso_utc(created),
        "coordinate_frames": {
            "geodetic": "WGS84 lat_deg/lon_deg/alt_m",
            "ecef": "WGS84 ECEF meters",
            "local_enu": "ENU meters relative to emission aircraft origin",
        },
        "time_basis": {
            "emission_time_utc": "Aircraft emission / sample time (UTC ISO-8601)",
            "ray_t_rel_s": "Acoustic travel time along ray from emission (s), dt=ds/c_eff",
            "ray_t_utc": "emission_time_utc + ray_t_rel_s",
        },
        "files": {key: path.name for key, path in written.items()},
        "counts": {
            "num_emissions": len(result.emissions),
            "num_rays": int(sum(len(e.rays) for e in result.emissions)),
            "num_ray_samples": int(num_ray_samples),
            "num_ground_hits": int(len(hit_rows2)),
        },
    }
    manifest_path = out / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    written["manifest"] = manifest_path

    return written


def _parse_time_utc(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def write_kml_from_run_archive(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    max_rays_per_emission: int | None = None,
    max_points_per_ray: int = 220,
    max_ground_hits: int | None = None,
) -> Path:
    """Rebuild Google Earth KML from a mach_cutoff_run_v1 archive (all rays by default).

    Streams ``ray_samples.csv`` so large high-fidelity runs do not need a full
    in-memory ``SimulationResult``.
    """
    from html import escape

    from .outputs import (
        _kml_color,
        _kml_gx_track_elements,
        _kml_linestring_coordinates,
        _iso_z,
    )

    root = Path(run_dir)
    out_path = Path(output_path) if output_path is not None else root / "google_earth_overlay.kml"

    flight_path = root / "flight_emissions.csv"
    samples_path = root / "ray_samples.csv"
    hits_path = root / "ground_hits.csv"
    rays_path = root / "rays.csv"

    if not flight_path.exists():
        raise FileNotFoundError(f"Missing flight_emissions.csv in {root}")

    with flight_path.open("r", encoding="utf-8", newline="") as f:
        flight_rows = list(csv.DictReader(f))

    # ground-hit set for ray descriptions
    hit_keys: set[tuple[int, int]] = set()
    if rays_path.exists():
        with rays_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if int(float(row.get("ground_hit", 0) or 0)):
                    hit_keys.add((int(row["emission_id"]), int(row["ray_id"])))

    with out_path.open("w", encoding="utf-8") as out:
        out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        out.write(
            '<kml xmlns="http://www.opengis.net/kml/2.2" '
            'xmlns:gx="http://www.google.com/kml/ext/2.2">\n'
        )
        out.write("  <Document>\n")
        out.write("    <name>Mach Cutoff Simulation Overlay</name>\n")
        out.write("    <Style id=\"flightStyle\">\n")
        out.write("      <LineStyle>\n")
        out.write(f"        <color>{_kml_color('#111111', alpha=255)}</color>\n")
        out.write("        <width>3</width>\n")
        out.write("      </LineStyle>\n")
        out.write("      <IconStyle><scale>0</scale></IconStyle>\n")
        out.write("    </Style>\n")
        out.write("    <Style id=\"rayStyle\">\n")
        out.write("      <LineStyle>\n")
        out.write(f"        <color>{_kml_color('#1f5dff', alpha=110)}</color>\n")
        out.write("        <width>1</width>\n")
        out.write("      </LineStyle>\n")
        out.write("      <IconStyle><scale>0</scale></IconStyle>\n")
        out.write("    </Style>\n")
        out.write("    <Style id=\"groundHitStyle\">\n")
        out.write("      <IconStyle>\n")
        out.write(f"        <color>{_kml_color('#d62828', alpha=230)}</color>\n")
        out.write("        <scale>0.65</scale>\n")
        out.write("      </IconStyle>\n")
        out.write("      <LabelStyle><scale>0</scale></LabelStyle>\n")
        out.write("    </Style>\n")

        # Flight track
        if flight_rows:
            lat = np.asarray([float(r["lat_deg"]) for r in flight_rows], dtype=float)
            lon = np.asarray([float(r["lon_deg"]) for r in flight_rows], dtype=float)
            alt = np.asarray([float(r["alt_m"]) for r in flight_rows], dtype=float)
            times = [_parse_time_utc(r["time_utc"]) for r in flight_rows]
            track_elems = _kml_gx_track_elements(
                lat, lon, alt, times, max_points=max(2, int(lat.size))
            )
            flight_coords = _kml_linestring_coordinates(
                lat, lon, alt, max_points=max(2, int(lat.size))
            )
            out.write("    <Folder>\n      <name>Flight</name>\n")
            if track_elems:
                out.write("      <Placemark>\n")
                out.write("        <name>Aircraft track (timed)</name>\n")
                out.write("        <styleUrl>#flightStyle</styleUrl>\n")
                out.write("        <gx:Track>\n")
                out.write("          <altitudeMode>absolute</altitudeMode>\n")
                for line in track_elems:
                    out.write(line + "\n")
                out.write("        </gx:Track>\n")
                out.write("      </Placemark>\n")
            if flight_coords:
                out.write("      <Placemark>\n")
                out.write("        <name>Aircraft track (path)</name>\n")
                out.write("        <styleUrl>#flightStyle</styleUrl>\n")
                out.write("        <LineString>\n")
                out.write("          <tessellate>1</tessellate>\n")
                out.write("          <altitudeMode>absolute</altitudeMode>\n")
                out.write(f"          <coordinates>{flight_coords}</coordinates>\n")
                out.write("        </LineString>\n")
                out.write("      </Placemark>\n")
            out.write("    </Folder>\n")

        # Rays from ray_samples.csv (streamed by emission_id, ray_id)
        ray_count = 0
        out.write("    <Folder>\n      <name>Shock Rays (sampled)</name>\n")
        if samples_path.exists():
            with samples_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                current_key: tuple[int, int] | None = None
                lats: list[float] = []
                lons: list[float] = []
                alts: list[float] = []
                t_rels: list[float] = []
                t_utcs: list[datetime] = []
                rays_seen_for_emission: dict[int, int] = {}

                def _flush_ray(em_id: int, ray_id: int) -> None:
                    nonlocal ray_count
                    if not lats:
                        return
                    if max_rays_per_emission is not None:
                        seen = rays_seen_for_emission.get(em_id, 0)
                        if seen >= int(max_rays_per_emission):
                            return
                        rays_seen_for_emission[em_id] = seen + 1
                    lat_a = np.asarray(lats, dtype=float)
                    lon_a = np.asarray(lons, dtype=float)
                    alt_a = np.asarray(alts, dtype=float)
                    times = list(t_utcs)
                    track_elems = _kml_gx_track_elements(
                        lat_a, lon_a, alt_a, times, max_points=max_points_per_ray
                    )
                    coords = _kml_linestring_coordinates(
                        lat_a, lon_a, alt_a, max_points=max_points_per_ray
                    )
                    if not track_elems and not coords:
                        return
                    ray_count += 1
                    ground_hit = (em_id, ray_id) in hit_keys
                    t_end = float(t_rels[-1]) if t_rels else 0.0
                    name = f"Emission {em_id:03d} / Ray {ray_id:03d}"
                    desc = f"ground_hit={ground_hit}; path_time_s={t_end:.3f}"
                    out.write("      <Placemark>\n")
                    out.write(f"        <name>{escape(name)}</name>\n")
                    out.write(f"        <description>{escape(desc)}</description>\n")
                    out.write("        <styleUrl>#rayStyle</styleUrl>\n")
                    if track_elems:
                        out.write("        <gx:Track>\n")
                        out.write("          <altitudeMode>absolute</altitudeMode>\n")
                        for line in track_elems:
                            out.write(line + "\n")
                        out.write("        </gx:Track>\n")
                    elif coords:
                        out.write("        <LineString>\n")
                        out.write("          <tessellate>1</tessellate>\n")
                        out.write("          <altitudeMode>absolute</altitudeMode>\n")
                        out.write(f"          <coordinates>{coords}</coordinates>\n")
                        out.write("        </LineString>\n")
                    out.write("      </Placemark>\n")

                for row in reader:
                    em_id = int(row["emission_id"])
                    ray_id = int(row["ray_id"])
                    key = (em_id, ray_id)
                    if current_key is None:
                        current_key = key
                    if key != current_key:
                        _flush_ray(current_key[0], current_key[1])
                        lats, lons, alts, t_rels, t_utcs = [], [], [], [], []
                        current_key = key
                    lats.append(float(row["lat_deg"]))
                    lons.append(float(row["lon_deg"]))
                    alts.append(float(row["alt_m"]))
                    t_rels.append(float(row.get("t_rel_s") or 0.0))
                    t_utcs.append(_parse_time_utc(row["t_utc"]))
                if current_key is not None:
                    _flush_ray(current_key[0], current_key[1])
        out.write("    </Folder>\n")

        # Ground hits
        hit_count = 0
        if hits_path.exists():
            out.write("    <Folder>\n      <name>Ground Hits</name>\n")
            with hits_path.open("r", encoding="utf-8", newline="") as f:
                for i, row in enumerate(csv.DictReader(f)):
                    if max_ground_hits is not None and i >= int(max_ground_hits):
                        break
                    hit_count += 1
                    t_str = str(row.get("hit_time_utc") or row.get("emission_time_utc") or "")
                    lon_h = float(row["lon_deg"])
                    lat_h = float(row["lat_deg"])
                    out.write("      <Placemark>\n")
                    out.write(f"        <name>Hit {hit_count:04d}</name>\n")
                    out.write(f"        <description>{escape(t_str)}</description>\n")
                    if t_str:
                        out.write("        <TimeStamp>\n")
                        out.write(f"          <when>{escape(t_str)}</when>\n")
                        out.write("        </TimeStamp>\n")
                    out.write("        <styleUrl>#groundHitStyle</styleUrl>\n")
                    out.write("        <Point>\n")
                    out.write("          <altitudeMode>clampToGround</altitudeMode>\n")
                    out.write(f"          <coordinates>{lon_h:.6f},{lat_h:.6f},0</coordinates>\n")
                    out.write("        </Point>\n")
                    out.write("      </Placemark>\n")
            out.write("    </Folder>\n")

        out.write("    <description>")
        out.write(
            escape(
                f"rays_written={ray_count}, ground_hits={hit_count}; "
                "all rays included unless max_rays_per_emission set"
            )
        )
        out.write("</description>\n")
        out.write("  </Document>\n</kml>\n")

    return out_path

