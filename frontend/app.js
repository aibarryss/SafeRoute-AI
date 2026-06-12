const API_BASE = 'http://localhost:8000';
 
// ─── КАРТА ───────────────────────────────────────────────
const map = L.map('map').setView([50.4111, 80.2275], 13);
 
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap'
}).addTo(map);
 
function getColor(dangerLevel) {
    if (dangerLevel <= 3) return 'green';
    if (dangerLevel <= 6) return 'orange';
    return 'red';
}
 
async function loadZones() {
    try {
        const response = await fetch(`${API_BASE}/api/zones`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const zones = await response.json();
 
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
        console.error('[ERROR] Зоны не загружены:', error);
    }
}
 
loadZones();
 
 
// ─── РЕЖИМЫ ──────────────────────────────────────────────
let selectedMode = 'car'; // режим по умолчанию
 
document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        // убираем active у всех
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        // ставим active на нажатую
        btn.classList.add('active');
        selectedMode = btn.dataset.mode;
        console.log('[MODE]', selectedMode);
    });
});
 
 
// ─── МАРШРУТ НА КАРТЕ ────────────────────────────────────
let currentRouteLine = null; // храним линию чтобы удалять старую
 
function drawRoute(coordinates) {
    // удаляем предыдущий маршрут если был
    if (currentRouteLine) {
        map.removeLayer(currentRouteLine);
    }
 
    // рисуем новый маршрут
    currentRouteLine = L.polyline(coordinates, {
        color: '#56CCF2',
        weight: 4,
        opacity: 0.9,
        dashArray: '8, 4' // пунктирная линия
    }).addTo(map);
 
    // центрируем карту на маршруте
    map.fitBounds(currentRouteLine.getBounds(), { padding: [30, 30] });
}
 
 
// ─── КНОПКА ПОСТРОИТЬ МАРШРУТ ────────────────────────────
document.getElementById('build-route-btn').addEventListener('click', async () => {
    const from = document.getElementById('input-from').value.trim();
    const to = document.getElementById('input-to').value.trim();
 
    if (!from || !to) {
        alert('Введите откуда и куда');
        return;
    }
 
    const btn = document.getElementById('build-route-btn');
    btn.textContent = 'Строим маршрут...';
    btn.disabled = true;
 
    try {
        const response = await fetch(`${API_BASE}/api/route`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                from: from,
                to: to,
                mode: selectedMode
            })
        });
 
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
 
        // рисуем маршрут на карте
        if (data.route && data.route.length > 0) {
            drawRoute(data.route);
        }
 
        // показываем объяснение от ИИ
        const aiBox = document.getElementById('ai-explanation');
        const aiText = document.getElementById('ai-text');
        aiText.textContent = data.ai_explanation || 'Маршрут построен.';
        aiBox.classList.remove('hidden');
 
    } catch (error) {
        console.error('[ERROR] Маршрут не построен:', error);
        alert('Не удалось построить маршрут. Убедитесь что бэкенд запущен.');
    } finally {
        btn.textContent = 'Построить маршрут';
        btn.disabled = false;
    }
});