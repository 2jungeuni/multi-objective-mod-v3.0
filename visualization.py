import folium
import numpy as np
import pandas as pd

colors = [
    'red',
    'blue',
    'green',
    'darkred',
    'lightred',
    'orange',
    'beige',
    'gray',
    'darkgreen',
    'lightgreen',
    'darkblue',
    'lightblue',
    'purple',
    'darkpurple',
    'pink',
    'cadetblue',
    'lightgray',
    'black'
]

def visualization(func, route, num_veh):
    # central coordinate of New York City
    central_lat, central_lng = 40.78554, -73.95956

    # visualization
    m = folium.Map(location=(central_lat, central_lng), zoom_start=10)
    m_ = folium.Map(location=(central_lat, central_lng), zoom_start=10)

    for i in range(num_veh):
        locs = []
        r = route[i]
        for idx in range(len(r)-1):
            func.init()
            func.astar_path(r[idx], r[idx+1])
            path_detail = pd.read_csv("./result/optimal_path.csv")
            lat_path = list(path_detail["latitude"])
            lng_path = list(path_detail["longitude"])
            locs.append([lat_path[0], lng_path[0]])

            folium.PolyLine(locations=np.array([lat_path, lng_path]).T, smooth_factor=1.0, weight=2.0,
                            color=colors[i]).add_to(m)
            folium.CircleMarker(location=[lat_path[0], lng_path[0]], radius=2.0, color=colors[i], fill=True).add_to(m)
            folium.CircleMarker(location=[lat_path[0], lng_path[0]], radius=2.0, color=colors[i], fill=True).add_to(m_)

        locs.append([lat_path[-1], lng_path[-1]])
        folium.CircleMarker(location=[lat_path[-1], lng_path[-1]], radius=2.0, color=colors[i], fill=True).add_to(m)
        folium.CircleMarker(location=[lat_path[-1], lng_path[-1]], radius=2.0, color=colors[i], fill=True).add_to(m_)
        folium.PolyLine(locations=locs, smooth_factor=1.0, weight=2.0, color=colors[i]).add_to(m_)

    m.save("./result/paths.html")
    m_.save("./result/sequence.html")