"""Optional extension: OpenStreetMap context features around each cluster.

The competition's ``gee_features.csv`` already holds the Google Earth Engine
imagery features extracted per cluster (night lights, vegetation, land
surface temperature, rainfall, population, and so on), so the main pipeline
does not call this module.

This module shows how to add *new* context features from an OpenStreetMap
point extract (for example health facilities, schools, markets and water
points from a Geofabrik download). The output has one row per DHSID and can
be merged onto the GEE features before screening.

    osm = pd.read_csv("osm_points.csv")            # columns: lat, lon, category
    extra = build_osm_features(clusters, osm)      # clusters: DHSID, LATNUM, LONGNUM
    features = gee.merge(extra, on="DHSID", how="left")

Note: DHS displaces cluster GPS points by up to 2 km (urban) or 5 km (rural,
1% up to 10 km) to protect privacy, so buffers smaller than ~5 km are noisy.
"""
import numpy as np
import pandas as pd

from . import config

BUFFER_RADII_KM = (2, 5, 10)
OSM_CATEGORIES = ["health_facility", "school", "market", "water_point",
                  "road_primary", "populated_place"]
EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; works element-wise on numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def build_osm_features(clusters: pd.DataFrame, osm_points: pd.DataFrame,
                       radii_km=BUFFER_RADII_KM, categories=OSM_CATEGORIES,
                       max_search_km=50.0) -> pd.DataFrame:
    """Counts of OSM points per category within each radius, plus nearest distance.

    Nearest distance is capped at ``max_search_km`` (NaN if nothing is that close),
    and a bounding-box pre-filter keeps each cluster's search cheap.
    """
    pts = {c: osm_points.loc[osm_points["category"] == c, ["lat", "lon"]].to_numpy()
           for c in categories}
    records = []
    for cid, lat, lon in clusters[[config.ID_COLUMN, config.LAT_COLUMN,
                                   config.LON_COLUMN]].itertuples(index=False):
        row = {config.ID_COLUMN: cid}
        dlat = max_search_km / 111.0
        dlon = dlat / max(np.cos(np.radians(lat)), 0.05)
        for c, arr in pts.items():
            box = arr[(np.abs(arr[:, 0] - lat) <= dlat) & (np.abs(arr[:, 1] - lon) <= dlon)]
            d = haversine_km(lat, lon, box[:, 0], box[:, 1]) if len(box) else np.array([])
            d = d[d <= max_search_km]
            for r in radii_km:
                row[f"osm_{c}_count_{r}km"] = int((d <= r).sum())
            row[f"osm_{c}_nearest_km"] = float(d.min()) if len(d) else np.nan
        records.append(row)
    return pd.DataFrame(records)
