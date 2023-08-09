import os
import warnings
import random
import pickle
import numpy as np
import pandas as pd
import folium
import shapely
import geopandas as gpd
import networkx as nx
import osmnx as ox
from tqdm import tqdm
from pyproj import Geod
from math import atan2, degrees
from shapely.geometry import LineString
from typing import Optional, Tuple, List

DATA_FOLDER = "../data/"
GEOG_CRS = "EPSG:4326"
PROJ_CRS = "EPSG:3857"

# length_km: weight for path length in km

interv_dist_weights = dict(
    #traffic_light=2.6767,
    #turn_left=14.7726,
    #turn_right=13.8288,
    #curve=0.0,
    #four_way=6.1864,
    
    four_way = 18.563574999999997,
    curve = 14.190058333333333,
    road_immission = 7.442591666666666,
    traffic_light = 17.32914166666667,
    turn_left = 16.0386,
    turn_right = 28.914708333333333,

    length_km=0.0
)

co2_weights = dict(
    #traffic_light=0.0060,
    #turn_left=0.0,
    #turn_right=0.0088,
    #curve=0.0049,
    #four_way=0.0088,
    
    four_way = 0.0036333333333333335,
    curve = 0.0028083333333333333,
    road_immission = 0.004358333333333333,
    traffic_light = 0.0038250000000000003,
    turn_left = 0.0029749999999999998,
    turn_right = 0.004791666666666666,
    
    length_km=0.1
)


def load_obs(filepath) -> gpd.GeoDataFrame:
    """
    Load observations from a csv file.
    """
    df = pd.read_csv(filepath, sep=r"\+ACI-,\+ACI-", names=["lat", "lon"], engine="python")
    df = df.drop(df.index[0])

    # remove prefix from lat column
    df["lat"] = df["lat"].str.replace("+ACI-", "", regex=False).str.replace(",", ".").astype(float)
    # remove suffix from lng column
    df["lon"] = df["lon"].str.replace("+ACI-", "", regex=False).str.replace(",", ".").astype(float)

    points = gpd.points_from_xy(df.lon, df.lat)
    return gpd.GeoDataFrame(df, geometry=points, crs=GEOG_CRS)


def graph_containing_obs(obs: gpd.GeoDataFrame, distance: float = 1000) -> Tuple[
    nx.MultiDiGraph, shapely.geometry.Polygon]:
    """
    Create a graph of the area containing the observations.
    """

    assert (obs.crs == GEOG_CRS)

    # project the gdf to PROJ_CRS
    obs_proj = obs.to_crs(PROJ_CRS)

    # get the bounding box of the projected gdf
    bbox_proj = obs_proj.total_bounds

    # enlarge bbox_proj by <distance> meters
    bbox_proj = [bbox_proj[0] - distance, bbox_proj[1] - distance, bbox_proj[2] + distance, bbox_proj[3] + distance]

    # bbox_proj to polygon
    bbox_proj_geom = gpd.GeoSeries([shapely.geometry.box(*bbox_proj)], crs=PROJ_CRS).geometry
    bbox_poly = bbox_proj_geom.to_crs(GEOG_CRS).geometry.unary_union

    # create a graph from the bounding box
    return ox.graph_from_polygon(bbox_poly, network_type="drive"), bbox_poly


def impute_missing_geometries(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Impute missing geometries in the graph
    """

    # only simplified edges have the geometry attribute
    # see https://stackoverflow.com/questions/64333794/osmnx-graph-from-point-and-geometry-information

    x_lookup = nx.get_node_attributes(G, "x")
    y_lookup = nx.get_node_attributes(G, "y")

    for u, v, data in G.edges(data=True):
        if "geometry" not in data:
            data["geometry"] = LineString([(x_lookup[u], y_lookup[u]), (x_lookup[v], y_lookup[v])])

    return G


def add_edge_linearity(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Add edge linearity to the graph
    """

    for u, v, k, data in G.edges(keys=True, data=True):
        if "geometry" not in data:
            print("geometry not in data", u, v)

        assert "geometry" in data

        start_x, start_y = data["geometry"].coords[0]
        end_x, end_y = data["geometry"].coords[-1]

        assert abs(start_x - G.nodes[u]["x"]) < 0.001
        assert abs(start_y - G.nodes[u]["y"]) < 0.001
        assert abs(end_x - G.nodes[v]["x"]) < 0.001
        assert abs(end_y - G.nodes[v]["y"]) < 0.001

        greatc_old = ox.distance.great_circle_vec(G.nodes[u]["y"], G.nodes[u]["x"], G.nodes[v]["y"], G.nodes[v]["x"])
        greatc_dist = ox.distance.great_circle_vec(start_y, start_x, end_y, end_x)

        if abs(greatc_dist - greatc_old) > 1:
            print("len", data["length"])
            print("greatc", greatc_dist)
            print("greatc wrong", greatc_old)
            print("u:", G.nodes[u], "v:", G.nodes[v])
            print("start", (start_x, start_y), "end", (end_x, end_y))
            print()

        length = data["length"]

        data["linearity"] = greatc_dist / length

    return G


def marker_from_row(row: pd.Series, m: folium.Map, icon: str = "info-sign", color: str = "blue"):
    """
    row: Series representing the POI to plot
    m: folium map object
    icon: icon name
    color: marker color
    """
    long, lat = row.geometry.x, row.geometry.y
    folium.Marker(
        location=[lat, long],
        popup=row.to_string(),
        icon=folium.Icon(icon=icon),
        color=color
    ).add_to(m)


def path_to_uvk(G, path):
    node_pairs = zip(path[:-1], path[1:])
    uvk = ((u, v, min(G[u][v].items(), key=lambda k: k[1]["length"])[0]) for u, v in node_pairs)
    return uvk


def perpendicular_line(ab: LineString, cd_length: Optional[float] = None) -> LineString:
    if cd_length is None:
        cd_length = ab.length / 2

    left = list(ab.parallel_offset(cd_length / 2, 'left').coords)
    right = list(ab.parallel_offset(cd_length / 2, 'right').coords)
    c = [
        (left[0][0] + left[1][0]) / 2,
        (left[0][1] + left[1][1]) / 2
    ]
    d = [
        (right[0][0] + right[1][0]) / 2,
        (right[0][1] + right[1][1]) / 2
    ]
    cd = LineString([c, d])

    return cd


def generate_midpoints(orig_node: int, dest_node: int, n_midpoints: int, nodes: gpd.GeoDataFrame,
                       G_proj: nx.MultiDiGraph) -> List[int]:
    """
    orig_node: origin node osmid
    dest_node: destination node osmid
    n_midpoints: number of midpoints to generate
    nodes: nodes GeoDataFrame
    G_proj: projected graph
    """
    assert nodes.crs == GEOG_CRS
    assert G_proj.graph['crs'] == PROJ_CRS

    orig_point, _ = ox.projection.project_geometry(nodes.loc[orig_node, 'geometry'], crs=GEOG_CRS, to_crs=PROJ_CRS)
    orig_coords = orig_point.coords[0]
    dest_point, _ = ox.projection.project_geometry(nodes.loc[dest_node, 'geometry'], crs=GEOG_CRS, to_crs=PROJ_CRS)
    dest_coords = dest_point.coords[0]

    ab = LineString([orig_coords, dest_coords])
    cd = perpendicular_line(ab)

    ab_geog, _ = ox.projection.project_geometry(ab, crs=PROJ_CRS, to_crs=GEOG_CRS)
    cd_geog, _ = ox.projection.project_geometry(cd, crs=PROJ_CRS, to_crs=GEOG_CRS)

    spacing = cd.length / (n_midpoints - 1)

    X, Y = zip(*ox.utils_geo.interpolate_points(geom=cd, dist=spacing))

    midpoints = ox.distance.nearest_nodes(G_proj, X, Y)

    return midpoints


def multiple_paths(G, orig_node, dest_node, midpoints, weight='length'):
    paths = []
    for midpoint in midpoints:
        gen_1 = ox.shortest_path(G, orig_node, midpoint, weight)
        if gen_1 is None:  # no path found
            continue

        temp = list(gen_1)
        temp.pop()

        gen_2 = ox.shortest_path(G, midpoint, dest_node, weight)
        if gen_2 is None:  # no path found
            continue

        temp += list(gen_2)
        paths.append(temp)

    return paths


## Original implementation
# def path_metrics(nodes, path_edges):
#     events = []
#     left_turns = 0
#     right_turns = 0
#     iterator = path_edges.itertuples()
#     prev = next(iterator)
#     prev_end = LineString(prev.geometry.coords[-2:])

#     for curr in iterator:
#         u, v, _ = curr.Index

#         if nodes.loc[u, 'street_count'] < 3:
#             continue

#         curr_start = LineString(curr.geometry.coords[:2])

#         # folium.PolyLine(locations=[coord[::-1] for coord in prev_end.coords], color="red").add_to(m)
#         # folium.PolyLine(locations=[coord[::-1] for coord in curr_start.coords], color="green").add_to(m)

#         x1 = prev_end.coords[1][0] - prev_end.coords[0][0]
#         y1 = prev_end.coords[1][1] - prev_end.coords[0][1]
#         x2 = curr_start.coords[1][0] - curr_start.coords[0][0]
#         y2 = curr_start.coords[1][1] - curr_start.coords[0][1]
#         theta = degrees(atan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2))

#         print(f"Node {u} --> Angle between edge {prev.osmid} and {curr.osmid}: {theta} degrees")

#         # svolta a sinistra: angolo tra prev e curr compreso tra 25 e 155
#         if 25 <= theta <= 155:
#             left_turns += 1
#             events.append((nodes.loc[u].geometry.x, nodes.loc[u].geometry.y, "left_turn"))
#         elif -155 <= theta <= -25:
#             right_turns += 1
#             events.append((nodes.loc[u].geometry.x, nodes.loc[u].geometry.y, "right_turn"))

#         prev = curr
#         prev_end = LineString(prev.geometry.coords[-2:])

#     return events


def detect_turns(nodes: gpd.GeoDataFrame, path_edges: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    left_turns = []
    right_turns = []

    iterator = path_edges.itertuples()
    prev = next(iterator)
    prev_end = LineString(prev.geometry.coords[-2:])

    for curr in iterator:
        u, v, _ = curr.Index

        if nodes.loc[u, 'street_count'] < 3:
            continue

        curr_start = LineString(curr.geometry.coords[:2])

        # folium.PolyLine(locations=[coord[::-1] for coord in prev_end.coords], color="red").add_to(m)
        # folium.PolyLine(locations=[coord[::-1] for coord in curr_start.coords], color="green").add_to(m)

        x1 = prev_end.coords[1][0] - prev_end.coords[0][0]
        y1 = prev_end.coords[1][1] - prev_end.coords[0][1]
        x2 = curr_start.coords[1][0] - curr_start.coords[0][0]
        y2 = curr_start.coords[1][1] - curr_start.coords[0][1]
        theta = degrees(atan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2))

        # print(f"Node {u} --> Angle between edge {prev.Index} and {curr.Index}: {theta} degrees")

        # svolta a sinistra: angolo tra prev e curr compreso tra 25 e 155
        if 25 <= theta <= 155:
            item = {}
            item['node'] = u
            item['prev'] = prev.Index
            item['curr'] = curr.Index
            item['angle'] = theta
            item['geometry'] = nodes.loc[u].geometry
            left_turns.append(item)
        elif -155 <= theta <= -25:
            item = {}
            item['node'] = u
            item['prev'] = prev.Index
            item['curr'] = curr.Index
            item['angle'] = theta
            item['geometry'] = nodes.loc[u].geometry
            right_turns.append(item)

        prev = curr
        prev_end = LineString(prev.geometry.coords[-2:])

    if left_turns:
        left_turns = gpd.GeoDataFrame(left_turns, geometry='geometry', crs=GEOG_CRS)
    else:
        left_turns = None

    if right_turns:
        right_turns = gpd.GeoDataFrame(right_turns, geometry='geometry', crs=GEOG_CRS)
    else:
        right_turns = None

    return left_turns, right_turns


## Alternative implementation for buffers
# def path_buffer(path_data: gpd.GeoDataFrame, meters: int)-> gpd.GeoDataFrame:
#     if path_data.crs != PROJ_CRS:
#         path_data = ox.project_gdf(path_data, to_crs=PROJ_CRS)

#     gs = path_data.buffer(meters)
#     geom = gs.unary_union

#     buffer_gdf = gpd.GeoDataFrame(geometry=[geom], crs=gs.crs)

#     return buffer_gdf


# def count_traffic_lights(traffic_lights: gpd.GeoDataFrame, buffer: gpd.GeoDataFrame) -> int:
#     if traffic_lights.crs != PROJ_CRS:
#         traffic_lights = traffic_lights.to_crs(PROJ_CRS)

#     if buffer.crs != PROJ_CRS:
#         buffer = buffer.to_crs(PROJ_CRS)

#     return traffic_lights.sjoin(buffer)


def traffic_lights_along_path(traffic_lights: gpd.GeoDataFrame, path_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if traffic_lights.crs != PROJ_CRS:
        traffic_lights = traffic_lights.to_crs(PROJ_CRS)

    if path_data.crs != PROJ_CRS:
        path_data = path_data.to_crs(PROJ_CRS)

    result = []
    edges_iterator = path_data.iterrows()
    _, prev_edge = next(edges_iterator)
    i = 1
    for edge_index, edge in edges_iterator:
        for light_index, light in traffic_lights.iterrows():
            if (edge['geometry'].buffer(1).difference(prev_edge['geometry'].buffer(1))).intersects(light['geometry']) \
                    and \
                    (edge['reversed'] == False and light['traffic_signals:direction'] != 'backward'
                     or edge['reversed'] == True and light['traffic_signals:direction'] != 'forward'):
                item = {}
                item['osmid'] = light['osmid']
                item['direction'] = light['traffic_signals:direction']
                item['reversed'] = edge['reversed']
                item['edge'] = edge_index
                item['geometry'] = light['geometry']
                result.append(item)

        prev_edge = edge
        i += 1

    if result:
        return gpd.GeoDataFrame(result, crs=PROJ_CRS).to_crs(GEOG_CRS)

    return None


def detect_curves(path_data: gpd.GeoDataFrame, lin_th: float = 0.95) -> gpd.GeoDataFrame:
    result = path_data[path_data['linearity'] < lin_th]

    if result.shape[0]:
        result.loc[:, 'geometry'] = result['geometry'].apply(lambda x: x.representative_point())
        return result[['name', 'linearity', 'geometry']]

    return None


def detect_fwi(nodes: gpd.GeoDataFrame, path_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """ Detect 4-way instersections """

    # extract path_osmids from path_data
    path_osmids, _, _ = zip(*list(path_data.index))
    path_osmids = list(path_osmids)

    path_nodes = nodes.loc[path_osmids]
    result = path_nodes[path_nodes['street_count'] == 4]

    if result.shape[0]:
        return result[['street_count', 'geometry']]

    return None


def plot_pois(m: folium.Map, pois: gpd.GeoDataFrame, icon: str = "info-sign", color: str = "blue") -> folium.Map:
    """
    m: folium map object
    pois: POIs GeoDataFrame
    icon: icon name for markers
    color: marker color
    """
    pois.apply(lambda row: marker_from_row(row, m, icon, color), axis=1)
    return m


def path_score(n_left_turns, n_right_turns, n_curves, n_lights, n_four_ways, length_km, mode):
    if mode == "co2":
        weights = co2_weights
    elif mode == "interv_dist":
        weights = interv_dist_weights
    else:
        raise ValueError("Possible values for 'mode' are 'co2' and 'interv_dist'")

    return weights['turn_left'] * n_left_turns + \
           weights['turn_right'] * n_right_turns + \
           weights['curve'] * n_curves + \
           weights['traffic_light'] * n_lights + \
           weights['four_way'] * n_four_ways + \
           weights['length_km'] * length_km


def cast_attributes_to_float(G: nx.MultiDiGraph, columns: List) -> nx.MultiGraph:
    for u, v, k, data in G.edges(keys=True, data=True):
        for column in columns:
            data[column] = float(data[column])

    return G
