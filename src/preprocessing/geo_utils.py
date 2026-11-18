import numpy as np

EARTH_RADIUS_M = 6371000.0

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon1, lat1, lon2, lat2]
    )
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def bearing_deg(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon1, lat1, lon2, lat2]
    )
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = (np.cos(lat1) * np.sin(lat2)
         - np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

def angle_difference_deg(a, b):
    return np.abs((a - b + 180) % 360 - 180)

def local_xy(lon, lat, ref_lon, ref_lat):
    """Local equirectangular coordinates in metres; reference is in degrees."""
    ref_lat_rad = np.radians(ref_lat)
    x = np.radians(np.asarray(lon) - ref_lon) * EARTH_RADIUS_M * np.cos(ref_lat_rad)
    y = np.radians(np.asarray(lat) - ref_lat) * EARTH_RADIUS_M
    return x, y

def point_to_segment_distance(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom == 0:
        return float(np.hypot(px - ax, py - ay))
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / denom, 0.0, 1.0)
    qx, qy = ax + t * dx, ay + t * dy
    return float(np.hypot(px - qx, py - qy))
