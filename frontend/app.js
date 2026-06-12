// 1. Создаём карту, центр — Семей
const map = L.map('map').setView([50.4111, 80.2275], 13);

// 2. Подключаем тайлы OpenStreetMap (сами карты)
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap'
}).addTo(map);

// 3. Хардкод данные по районам Семея
const zones = [
    { name: "Центр",        lat: 50.4111, lng: 80.2275, danger: 3 },
    { name: "Промзона",     lat: 50.4300, lng: 80.2500, danger: 8 },
    { name: "Набережная",   lat: 50.4050, lng: 80.2100, danger: 2 },
    { name: "Спальный р-н", lat: 50.3950, lng: 80.2400, danger: 6 },
    { name: "Рынок Шыгыс", lat: 50.4200, lng: 80.2600, danger: 5 },
];

// 4. Функция: уровень опасности → цвет
function getColor(danger) {
    if (danger <= 3) return 'green';
    if (danger <= 6) return 'orange';
    return 'red';
}

// 5. Рисуем круги на карте
zones.forEach(zone => {
    L.circle([zone.lat, zone.lng], {
        color: getColor(zone.danger),
        fillColor: getColor(zone.danger),
        fillOpacity: 0.4,
        radius: 600  // метры
    })
    .bindPopup(`<b>${zone.name}</b><br>Опасность: ${zone.danger}/10`)
    .addTo(map);
});