// Backend URL (бэкенд на порту 8000)
const API_BASE = 'http://localhost:8000';

// 1. Создаём карту, центр — Семей
const map = L.map('map').setView([50.4111, 80.2275], 13);

// 2. Подключаем тайлы OpenStreetMap
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap'
}).addTo(map);

// 3. Функция: уровень опасности → цвет
function getColor(dangerLevel) {
    if (dangerLevel <= 3) return 'green';
    if (dangerLevel <= 6) return 'orange';
    return 'red';
}

// 4. Загружаем зоны с бэкенда
async function loadZones() {
    try {
        const response = await fetch(`${API_BASE}/api/zones`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const zones = await response.json();

        // 5. Рисуем зоны на карте
        zones.forEach(zone => {
            L.circle([zone.lat, zone.lng], {
                color: getColor(zone.danger_level),
                fillColor: getColor(zone.danger_level),
                fillOpacity: 0.4,
                radius: zone.radius
            })
            .bindPopup(`
                <b>${zone.name}</b><br>
                Опасность: ${zone.danger_level}/10<br>
                <small>${zone.description || ''}</small>
            `)
            .addTo(map);
        });

        console.log(`[OK] Загружено ${zones.length} зон`);

    } catch (error) {
        console.error('[ERROR] Не удалось загрузить зоны:', error);
        alert('Не удалось загрузить зоны опасности. Убедитесь что бэкенд запущен на порту 8000');
    }
}

// Загружаем зоны при старте
loadZones();
