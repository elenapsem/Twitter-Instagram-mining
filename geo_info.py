import geocoder
import folium
import pandas


def geocoding(location):
    g = geocoder.osm(location)
    if g.json != None:
        Long = g.osm['x']
        Lat = g.osm['y']
        country = g.country
        return [Long, Lat], country
    else:
        return None, None

def find_country(coordinates):
    # latitude and longitude
    g = geocoder.osm([coordinates['coordinates'][1], coordinates['coordinates'][0]], method='reverse')
    return g.country

def draw_map(country):
    df = pandas.DataFrame(pandas.Series.value_counts(country))
    countries = list(df.index)
    my_map = folium.Map()
    i = 0
    for x in countries:
        cor, country = geocoding(x)
        folium.Marker([cor[1], cor[0]], popup='Number of tweets in ' + x + ': ' + str(df.iloc[i,0])).add_to(my_map)
        i = i + 1
    my_map.save("country_map.html")