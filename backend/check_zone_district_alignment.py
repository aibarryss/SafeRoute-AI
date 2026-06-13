"""
Скрипт для проверки соответствия между зонами и районами.
"""
import json
import math
import sys
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def haversine(lat1, lon1, lat2, lon2):
    """Расстояние между двумя точками в метрах."""
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c

def point_in_polygon(lat, lng, polygon):
    """Проверяет, находится ли точка внутри полигона (ray casting)."""
    n = len(polygon)
    inside = False
    j = n - 1

    for i in range(n):
        yi = polygon[i]['lat']
        xi = polygon[i]['lng']
        yj = polygon[j]['lat']
        xj = polygon[j]['lng']

        if ((yi > lat) != (yj > lat)) and (lng < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside

def get_polygon_center(polygon):
    """Вычисляет центр полигона."""
    lat_sum = sum(p['lat'] for p in polygon)
    lng_sum = sum(p['lng'] for p in polygon)
    return lat_sum / len(polygon), lng_sum / len(polygon)

def main():
    
    data_dir = Path(__file__).parent / 'data'

    with open(data_dir / 'semey_zones.json', 'r', encoding='utf-8') as f:
        zones = json.load(f)

    with open(data_dir / 'districts.json', 'r', encoding='utf-8') as f:
        districts_data = json.load(f)
        districts = districts_data['districts']

    print("=" * 80)
    print("АНАЛИЗ СООТВЕТСТВИЯ ЗОН И РАЙОНОВ")
    print("=" * 80)
    print()

    
    for zone in zones:
        print(f"\nЗОНА: {zone['name']} (ID: {zone['id']})")
        print(f"  Центр: ({zone['lat']}, {zone['lng']})")
        print(f"  Радиус: {zone['radius']} м")
        print(f"  Danger Level: {zone['danger_level']}")
        print()
   
        affected_districts = []

        for district in districts:
            center_lat, center_lng = get_polygon_center(district['polygon'])
            distance = haversine(zone['lat'], zone['lng'], center_lat, center_lng)

            
            if distance <= zone['radius']:
                affected_districts.append({
                    'district': district,
                    'distance': distance
                })

        if affected_districts:
            print("  Районы внутри зоны:")
            for item in affected_districts:
                d = item['district']
                dist = item['distance']
                match = "✓" if d['danger_level'] == zone['danger_level'] else "✗"
                print(f"    {match} {d['name']} (ID: {d['id']})")
                print(f"       Distance: {dist:.0f} m")
                print(f"       District DL: {d['danger_level']}, Zone DL: {zone['danger_level']}")
                if d['danger_level'] != zone['danger_level']:
                    print(f"       ⚠️  НЕСООТВЕТСТВИЕ: разница {abs(d['danger_level'] - zone['danger_level'])}")
        else:
            print("  ⚠️  Нет районов внутри зоны")

    print()
    print("=" * 80)
    print("СВОДКА ПО НЕСООТВЕТСТВИЯМ")
    print("=" * 80)


    mismatches = []

    for zone in zones:
        for district in districts:
            center_lat, center_lng = get_polygon_center(district['polygon'])
            distance = haversine(zone['lat'], zone['lng'], center_lat, center_lng)

            if distance <= zone['radius'] and district['danger_level'] != zone['danger_level']:
                mismatches.append({
                    'zone': zone,
                    'district': district,
                    'distance': distance
                })

    if mismatches:
        print(f"\nНайдено {len(mismatches)} несоответствий:\n")
        for m in mismatches:
            print(f"  {m['zone']['name']} (DL:{m['zone']['danger_level']}) <-> {m['district']['name']} (DL:{m['district']['danger_level']})")
            print(f"    Расстояние: {m['distance']:.0f} м")
    else:
        print("\n✓ Все уровни опасности соответствуют!")

if __name__ == '__main__':
    main()
