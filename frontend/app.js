// Backend URL
const API_BASE = "http://localhost:8000";

// Текущий режим маршрута
let currentMode = "car";

// Инициализация карты
const map = L.map("map").setView([50.4111, 80.2275], 13);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

// Слой районов (всегда виден)
let districtsLayer = null;

// Загрузка районов города
async function loadDistricts() {
  try {
    const response = await fetch(`${API_BASE}/api/districts`);
    const districts = await response.json();

    // Удаляем старый слой
    if (districtsLayer) {
      map.removeLayer(districtsLayer);
    }

    // Создаём слой с районами
    districtsLayer = L.layerGroup();

    districts.forEach((district) => {
      const color = getDangerColor(district.danger_level);
      const polygon = L.polygon(
        district.polygon.map(p => [p.lat, p.lng]),
        {
          color: "#333",
          weight: 2,
          fillColor: color,
          fillOpacity: 0.4
        }
      ).addTo(districtsLayer);

      polygon.bindPopup(`
        <b>${district.name}</b><br>
        Уровень опасности: ${district.danger_level}/10<br>
        <small>${district.description || ""}</small>
      `);

      // Добавляем название района в центр
      const center = polygon.getBounds().getCenter();
      L.marker(center, {
        icon: L.divIcon({
          className: "district-label",
          html: `<div style="background: ${color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px; white-space: nowrap;">${district.name}</div>`,
          iconSize: null,
          iconAnchor: [50, 10]
        })
      }).addTo(districtsLayer);
    });

    districtsLayer.addTo(map);
    console.log(`✓ Районы загружены: ${districts.length}`);

  } catch (error) {
    console.error("Ошибка загрузки районов:", error);
  }
}

// Загрузка информации о ML модели
async function loadMLInfo() {
  try {
    const response = await fetch(`${API_BASE}/api/ml/info`);
    const info = await response.json();

    displayModelInfo(info);
    console.log("✓ ML модель загружена:", info.model_info.model_type);
  } catch (error) {
    console.error("Ошибка загрузки ML info:", error);
  }
}

// Отображение информации о модели
function displayModelInfo(info) {
  const panel = document.getElementById("ml-info-panel");
  if (!panel) return;

  const metrics = info.model_info.metrics;
  const importance = info.feature_importance;

  // Создание HTML для feature importance
  let importanceHTML = '<div class="feature-importance">';
  Object.entries(importance).forEach(([feature, value]) => {
    const desc = info.feature_descriptions[feature];
    const percentage = (value * 100).toFixed(1);
    importanceHTML += `
            <div class="feature-bar">
                <div class="feature-name" title="${desc}">${feature}</div>
                <div class="feature-value">
                    <div class="bar" style="width: ${percentage}%"></div>
                    <span>${percentage}%</span>
                </div>
            </div>
        `;
  });
  importanceHTML += "</div>";

  panel.innerHTML = `
        <h3>ML Модель</h3>
        <div class="model-stats">
            <div class="stat">
                <span class="label">Тип:</span>
                <span class="value">${info.model_info.model_type}</span>
            </div>
            <div class="stat">
                <span class="label">Accuracy:</span>
                <span class="value">${(metrics.accuracy * 100).toFixed(1)}%</span>
            </div>
            <div class="stat">
                <span class="label">F1-score:</span>
                <span class="value">${(metrics.f1_macro * 100).toFixed(1)}%</span>
            </div>
            <div class="stat">
                <span class="label">Предсказаний:</span>
                <span class="value">${info.statistics.total_predictions}</span>
            </div>
        </div>
        <h4>Важность признаков</h4>
        ${importanceHTML}
    `;
}

// Обработчик клика по карте для ML предсказания
map.on("click", async (e) => {
  const { lat, lng } = e.latlng;

  // Показываем индикатор загрузки
  const marker = L.marker([lat, lng]).addTo(map);
  marker.bindPopup("<b>Анализ...</b><br>Получение ML предсказания").openPopup();

  try {
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        lat: lat,
        lng: lng,
        hour: new Date().getHours(),
        day: new Date().getDay(),
      }),
    });

    const prediction = await response.json();

    // Обновляем маркер с результатами
    const color = getDangerColor(prediction.danger_level);
    marker.setIcon(
      L.divIcon({
        className: "prediction-marker",
        html: `<div style="background: ${color}; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">${prediction.danger_level}</div>`,
        iconSize: [30, 30],
        iconAnchor: [15, 15],
      }),
    );

    marker.setPopupContent(`
            <b>ML Предсказание</b><br>
            <div class="prediction-result">
                <div class="danger-level">
                    <span class="label">Уровень опасности:</span>
                    <span class="value" style="color: ${color}; font-size: 24px; font-weight: bold;">
                        ${prediction.danger_level}/10
                    </span>
                </div>
                <div class="confidence">
                    <span class="label">Уверенность:</span>
                    <span class="value">${(prediction.confidence * 100).toFixed(1)}%</span>
                </div>
                <div class="risk-category">
                    <span class="label">Категория риска:</span>
                    <span class="value">${translateRiskCategory(prediction.risk_category)}</span>
                </div>
                <div class="latency">
                    <span class="label">Время ответа:</span>
                    <span class="value">${prediction.latency_ms.toFixed(2)} мс</span>
                </div>
            </div>
        `);

    console.log("✓ Предсказание:", prediction);
  } catch (error) {
    console.error("Ошибка предсказания:", error);
    marker.setPopupContent("<b>Ошибка</b><br>Не удалось получить предсказание");
  }
});

// Переключение режимов маршрута
document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    // Убираем active со всех кнопок
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    // Добавляем active на текущую
    btn.classList.add('active');
    // Сохраняем режим
    currentMode = btn.dataset.mode;
    console.log(`✓ Режим изменен: ${currentMode}`);
  });
});

// Цвет по уровню опасности
function getDangerColor(level) {
  if (level <= 3) return "#4CAF50"; // Зеленый
  if (level <= 6) return "#FF9800"; // Оранжевый
  if (level <= 8) return "#FF5722"; // Красный
  return "#D32F2F"; // Темно-красный
}

// Перевод категорий риска
function translateRiskCategory(category) {
  const translations = {
    low: "Низкий",
    medium: "Средний",
    high: "Высокий",
    critical: "Критический",
  };
  return translations[category] || category;
}

// Построение маршрута
document.getElementById('build-route-btn').addEventListener('click', async () => {
  const startCoords = getFromCoords();
  const endCoords = getToCoords();

  if (!startCoords || !endCoords) {
    alert('Выберите адреса из списка подсказок');
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/api/route`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        start: startCoords,
        end: endCoords,
        mode: currentMode
      })
    });

    const data = await response.json();

    // Отображение маршрута
    if (window.routeLayer) {
      map.removeLayer(window.routeLayer);
    }

    // Цвет линии в зависимости от уровня опасности
    const routeColor = data.danger_score <= 3 ? '#4CAF50' :  // Зеленый - безопасно
                       data.danger_score <= 6 ? '#FF9800' :  // Оранжевый - средне
                       '#F44336';  // Красный - опасно

    window.routeLayer = L.polyline(
      data.route.map(p => [p.lat, p.lng]),
      { color: routeColor, weight: 5, opacity: 0.8 }
    ).addTo(map);

    // Показать информацию о безопасности маршрута
    const safetyLevel = data.danger_score <= 3 ? 'БЕЗОПАСНЫЙ' :
                        data.danger_score <= 6 ? 'СРЕДНЕЙ ОПАСНОСТИ' : 'ОПАСНЫЙ';

    const safetyIcon = data.danger_score <= 3 ? '✅' :
                       data.danger_score <= 6 ? '⚠️' : '🚨';

    const explanationDiv = document.getElementById('ai-explanation');
    explanationDiv.classList.remove('hidden');
    explanationDiv.innerHTML = `
      <div style="margin-bottom: 12px;">
        <b style="font-size: 16px;">${safetyIcon} Маршрут: ${safetyLevel}</b>
      </div>
      <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 6px; margin-bottom: 12px;">
        <div style="font-size: 13px; margin-bottom: 4px;">
          <b>Уровень опасности:</b> ${data.danger_score.toFixed(1)}/10
        </div>
        <div style="font-size: 12px; color: #aaa;">
          Режим: ${currentMode === 'car' ? '🚗 Автомобиль' :
                  currentMode === 'child' ? '👶 С ребёнком' : '🧳 Турист'}
        </div>
      </div>
      <div style="font-size: 13px; line-height: 1.6;">
        ${data.ai_explanation}
      </div>
    `;

    // Автоматически масштабировать карту на весь маршрут
    map.fitBounds(window.routeLayer.getBounds(), { padding: [50, 50] });

    console.log('✓ Маршрут построен:', data);
  } catch (error) {
    console.error('Ошибка построения маршрута:', error);
    alert('Не удалось построить маршрут');
  }
});

// ГЕОКОДИНГ ЧЕРЕЗ NOMINATIM (OpenStreetMap)
const NOMINATIM_URL = 'https://nominatim.openstreetmap.org/search';
const SEARCH_DEBOUNCE_MS = 200; // Ускорено с 400 до 200ms
let searchTimeout = null;

// Популярные места Семея (мгновенно показываются при фокусе)
const POPULAR_PLACES = [
  { name: 'Площадь Абая', address: 'Центральная площадь, Семей', lat: 50.4111, lng: 80.2275 },
  { name: 'Центральный парк', address: 'Парк культуры и отдыха', lat: 50.4089, lng: 80.2312 },
  { name: 'Ж/д вокзал Семей', address: 'Привокзальная площадь', lat: 50.4156, lng: 80.2189 },
  { name: 'Автостанция', address: 'ул. Абая, 100', lat: 50.4123, lng: 80.2234 },
  { name: 'СГУ им. Шакарима', address: 'ул. Глинки, 20', lat: 50.4145, lng: 80.2256 },
  { name: 'Медицинский университет', address: 'ул. Абая, 103', lat: 50.4098, lng: 80.2298 },
  { name: 'ТЦ "Family"', address: 'ул. Абая, 95', lat: 50.4134, lng: 80.2267 },
  { name: 'ТЦ "Евразия"', address: 'ул. Ленина, 45', lat: 50.4167, lng: 80.2245 },
  { name: 'Центральный рынок', address: 'ул. Рыскулова, 12', lat: 50.4078, lng: 80.2321 },
  { name: 'Драматический театр', address: 'ул. Абая, 88', lat: 50.4101, lng: 80.2287 },
  { name: 'Музей Достоевского', address: 'ул. Достоевского, 135', lat: 50.4067, lng: 80.2345 },
  { name: 'Больница №1', address: 'ул. Кайым Мухамедханова, 15', lat: 50.4189, lng: 80.2212 },
];

// Кэш результатов поиска
const searchCache = new Map();

// Поиск адреса через Nominatim с кэшированием
async function searchAddress(query) {
  if (!query || query.length < 3) return [];

  // Проверка кэша
  const cacheKey = query.toLowerCase();
  if (searchCache.has(cacheKey)) {
    return searchCache.get(cacheKey);
  }

  try {
    const url = `${NOMINATIM_URL}?q=${encodeURIComponent(query + ', Semey, Kazakhstan')}&format=json&addressdetails=1&limit=8&accept-language=ru`;
    const response = await fetch(url);
    const results = await response.json();

    const mapped = results.map(r => ({
      name: r.display_name.split(',')[0],
      address: r.display_name,
      lat: parseFloat(r.lat),
      lng: parseFloat(r.lon)
    }));

    // Сохранение в кэш
    searchCache.set(cacheKey, mapped);
    return mapped;
  } catch (error) {
    console.error('Ошибка поиска адреса:', error);
    return [];
  }
}

// Фильтрация популярных мест по запросу
function filterPopularPlaces(query) {
  if (!query || query.length < 2) return POPULAR_PLACES;

  const q = query.toLowerCase();
  return POPULAR_PLACES.filter(place =>
    place.name.toLowerCase().includes(q) ||
    place.address.toLowerCase().includes(q)
  );
}

// Показ результатов в dropdown
function showSearchResults(dropdownId, results, onSelect) {
  const dropdown = document.getElementById(dropdownId);
  if (!dropdown) return;

  if (results.length === 0) {
    dropdown.innerHTML = '<div class="search-empty">Ничего не найдено</div>';
    dropdown.classList.add('active');
    return;
  }

  dropdown.innerHTML = results.map((r, i) => `
    <div class="search-item" data-index="${i}">
      <div class="search-item-name">${r.name}</div>
      <div class="search-item-address">${r.address}</div>
    </div>
  `).join('');

  dropdown.classList.add('active');

  // Обработчики клика
  dropdown.querySelectorAll('.search-item').forEach((item, i) => {
    item.addEventListener('click', () => {
      onSelect(results[i]);
      dropdown.classList.remove('active');
    });
  });
}

// Настройка автокомплита для поля
function setupAutocomplete(inputId, dropdownId, markerColor) {
  const input = document.getElementById(inputId);
  const dropdown = document.getElementById(dropdownId);
  if (!input || !dropdown) return;

  let selectedCoords = null;
  let marker = null;

  // Мгновенное появление при фокусе - показываем популярные места
  input.addEventListener('focus', () => {
    const query = input.value.trim();
    const popular = filterPopularPlaces(query);

    showSearchResults(dropdownId, popular, (result) => {
      input.value = result.name;
      selectedCoords = { lat: result.lat, lng: result.lng };

      // Показать маркер на карте
      if (marker) map.removeLayer(marker);
      marker = L.marker([result.lat, result.lng], {
        icon: L.divIcon({
          className: 'custom-marker',
          html: `<div style="background:${markerColor};width:24px;height:24px;border-radius:50%;border:3px solid white;box-shadow:0 2px 6px rgba(0,0,0,0.3);"></div>`,
          iconSize: [24, 24],
          iconAnchor: [12, 12]
        })
      }).addTo(map).bindPopup(result.name);

      map.setView([result.lat, result.lng], 14);
    });
  });

  // Обработчик ввода с debounce - фильтрация + поиск в Nominatim
  input.addEventListener('input', (e) => {
    const query = e.target.value.trim();
    selectedCoords = null;

    if (searchTimeout) clearTimeout(searchTimeout);

    if (query.length < 2) {
      // Показываем все популярные места
      const popular = filterPopularPlaces('');
      showSearchResults(dropdownId, popular, (result) => {
        input.value = result.name;
        selectedCoords = { lat: result.lat, lng: result.lng };

        if (marker) map.removeLayer(marker);
        marker = L.marker([result.lat, result.lng], {
          icon: L.divIcon({
            className: 'custom-marker',
            html: `<div style="background:${markerColor};width:24px;height:24px;border-radius:50%;border:3px solid white;box-shadow:0 2px 6px rgba(0,0,0,0.3);"></div>`,
            iconSize: [24, 24],
            iconAnchor: [12, 12]
          })
        }).addTo(map).bindPopup(result.name);

        map.setView([result.lat, result.lng], 14);
      });
      return;
    }

    searchTimeout = setTimeout(async () => {
      dropdown.innerHTML = '<div class="search-loading">Поиск...</div>';
      dropdown.classList.add('active');

      // Сначала фильтруем популярные места
      const popular = filterPopularPlaces(query);

      // Затем ищем в Nominatim
      const nominatimResults = await searchAddress(query);

      // Объединяем результаты (популярные первыми)
      const allResults = [...popular, ...nominatimResults];

      // Убираем дубликаты по координатам
      const uniqueResults = allResults.filter((result, index, self) =>
        index === self.findIndex(r =>
          Math.abs(r.lat - result.lat) < 0.0001 &&
          Math.abs(r.lng - result.lng) < 0.0001
        )
      );

      showSearchResults(dropdownId, uniqueResults, (result) => {
        input.value = result.name;
        selectedCoords = { lat: result.lat, lng: result.lng };

        // Показать маркер на карте
        if (marker) map.removeLayer(marker);
        marker = L.marker([result.lat, result.lng], {
          icon: L.divIcon({
            className: 'custom-marker',
            html: `<div style="background:${markerColor};width:24px;height:24px;border-radius:50%;border:3px solid white;box-shadow:0 2px 6px rgba(0,0,0,0.3);"></div>`,
            iconSize: [24, 24],
            iconAnchor: [12, 12]
          })
        }).addTo(map).bindPopup(result.name);

        map.setView([result.lat, result.lng], 14);
      });
    }, SEARCH_DEBOUNCE_MS);
  });

  // Закрытие dropdown при клике вне
  document.addEventListener('click', (e) => {
    if (!input.contains(e.target) && !dropdown.contains(e.target)) {
      dropdown.classList.remove('active');
    }
  });

  // Возвращаем функцию для получения координат
  return () => selectedCoords;
}

// Настройка автокомплита для полей "Откуда" и "Куда"
const getFromCoords = setupAutocomplete('input-from', 'dropdown-from', '#4CAF50');
const getToCoords = setupAutocomplete('input-to', 'dropdown-to', '#F44336');

// Инициализация
loadDistricts();
loadMLInfo();
