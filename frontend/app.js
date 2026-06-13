// ===== КОНФИГУРАЦИЯ =====
const API_BASE = "http://localhost:8000";
const SEARCH_DEBOUNCE_MS = 250;

// Текущий режим маршрута
let currentMode = "car";
let TWOGIS_API_KEY = null;
let map = null; // Глобальная переменная карты

// Получаем 2GIS ключ с бэкенда (с fallback для разработки)
async function loadConfig() {
  try {
    const response = await fetch(`${API_BASE}/api/config`);
    const config = await response.json();
    TWOGIS_API_KEY = config.twogis_api_key;

    // Если бэкенд вернул пустой ключ — используем fallback
    if (!TWOGIS_API_KEY) {
      console.warn("⚠️ 2GIS API ключ пуст. Используйте .env файл для настройки.");
      TWOGIS_API_KEY = "2db79fdb-a7bc-43c4-8e90-5ca828ef5449"; // Только для разработки!
    }
    return TWOGIS_API_KEY;
  } catch (error) {
    console.warn("⚠️ Бэкенд недоступен, используется fallback ключ (только для разработки)");
    TWOGIS_API_KEY = "2db79fdb-a7bc-43c4-8e90-5ca828ef5449"; // Только для разработки!
    return TWOGIS_API_KEY;
  }
}

// ===== ИНИЦИАЛИЗАЦИЯ КАРТЫ 2GIS =====
async function initMap() {
  console.log("🗺️ Инициализация карты 2GIS...");

  if (!TWOGIS_API_KEY) {
    await loadConfig();
  }

  if (!TWOGIS_API_KEY) {
    console.error("❌ 2GIS API ключ не получен");
    return null;
  }

  console.log("✓ 2GIS ключ получен, создаю карту...");

  try {
    map = new mapgl.Map("map", {
      center: [80.2275, 50.4111], // [lng, lat] — порядок 2GIS!
      zoom: 13,
      key: TWOGIS_API_KEY,
      style: "c080bb6a-8134-4623-9aa9-446e2f3866c6", // тёмный стиль (опционально)
    });

    console.log("✓ Объект карты создан");

    // Ждём загрузки карты
    map.on("idle", () => {
      console.log("✓ Карта 2GIS загружена");
    });

    // ===== КЛИК ПО КАРТЕ — ML ПРЕДСКАЗАНИЕ =====
    map.on("click", async (e) => {
      const [lng, lat] = e.lngLat; // 2GIS возвращает [lng, lat]

      // Показываем индикатор загрузки
      if (predictMarker) {
        predictMarker.destroy();
        predictMarker = null;
      }

      predictMarker = new mapgl.HtmlMarker(map, {
        coordinates: [lng, lat],
        html: `<div style="background:#555;width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-size:11px;">⏳</div>`,
      });

      showMapToast("Анализ... Получение ML предсказания");

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
        const color = getDangerColor(prediction.danger_level);

        // Заменяем маркер загрузки на результат
        if (predictMarker) {
          predictMarker.destroy();
          predictMarker = null;
        }

        predictMarker = new mapgl.HtmlMarker(map, {
          coordinates: [lng, lat],
          html: `<div style="background:${color};width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;">${prediction.danger_level}</div>`,
        });

        const riskText = translateRiskCategory(prediction.risk_category);
        showMapToast(
          `ML Предсказание: Уровень опасности ${prediction.danger_level}/10 (${riskText}). Уверенность: ${(prediction.confidence * 100).toFixed(1)}%`
        );

        console.log("✓ Предсказание:", prediction);
      } catch (error) {
        console.error("Ошибка предсказания:", error);
        if (predictMarker) {
          predictMarker.destroy();
          predictMarker = null;
        }
        showMapToast("Ошибка: Не удалось получить предсказание");
      }
    });

    return map;
  } catch (error) {
    console.error("❌ Ошибка создания карты:", error);
    return null;
  }
}

// Массивы для хранения объектов карты
let districtObjects = []; // полигоны + маркеры районов
let markerObjects = [];   // маркеры выбранных адресов
let routeObject = null;   // полилиния маршрута
let predictMarker = null; // маркер ML предсказания

// ===== ЗАГРУЗКА РАЙОНОВ ГОРОДА =====
async function loadDistricts() {
  try {
    const response = await fetch(`${API_BASE}/api/districts`);
    const districts = await response.json();

    // Удаляем старые объекты
    districtObjects.forEach((obj) => obj.destroy());
    districtObjects = [];

    districts.forEach((district) => {
      const color = getDangerColor(district.danger_level);
      const fillColor = hexToRgba(color, 0.4);

      // Полигон района
      const polygon = new mapgl.Polygon(map, {
        coordinates: [district.polygon.map((p) => [p.lng, p.lat])], // [lng, lat]!
        strokeWidth: 2,
        strokeColor: "#333",
        fillColor: fillColor,
      });

      polygon.on("click", () => {
        showDistrictPopup(district);
      });

      districtObjects.push(polygon);

      // Вычисление центра полигона для подписи
      const center = getPolygonCenter(district.polygon);
      const label = new mapgl.HtmlMarker(map, {
        coordinates: [center.lng, center.lat], // [lng, lat]!
        html: `<div style="background: ${color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px; white-space: nowrap; pointer-events: none;">${district.name}</div>`,
      });
      districtObjects.push(label);
    });

    console.log(`✓ Районы загружены: ${districts.length}`);
  } catch (error) {
    console.error("Ошибка загрузки районов:", error);
  }
}

// Всплывающее окно района (используем простой alert/popup)
function showDistrictPopup(district) {
  const info = `
    ${district.name}
    Уровень опасности: ${district.danger_level}/10
    ${district.description || ""}
  `;
  // Показываем в виде временного сообщения на карте
  showMapToast(info);
}

// Показывает временное сообщение поверх карты
function showMapToast(text) {
  let toast = document.getElementById("map-toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "map-toast";
    toast.style.cssText =
      "position:absolute;top:10px;left:50%;transform:translateX(-50%);background:#1e1e2e;color:#fff;padding:10px 16px;border-radius:8px;font-size:13px;z-index:2000;box-shadow:0 4px 12px rgba(0,0,0,0.5);max-width:280px;pointer-events:none;text-align:center;line-height:1.5;";
    document.getElementById("map").appendChild(toast);
  }
  toast.textContent = text;
  toast.style.display = "block";
  clearTimeout(toast._timeout);
  toast._timeout = setTimeout(() => {
    toast.style.display = "none";
  }, 4000);
}

// ===== ЗАГРУЗКА ИНФОРМАЦИИ О ML МОДЕЛИ =====
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


// ===== ПЕРЕКЛЮЧЕНИЕ РЕЖИМОВ МАРШРУТА =====
document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentMode = btn.dataset.mode;
    console.log(`✓ Режим изменен: ${currentMode}`);
  });
});

// ===== ЦВЕТА И УТИЛИТЫ =====

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

// Конвертирует HEX в RGBA (для заливки полигонов)
function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// Центр полигона (среднее по координатам)
function getPolygonCenter(polygon) {
  let sumLat = 0,
    sumLng = 0;
  polygon.forEach((p) => {
    sumLat += p.lat;
    sumLng += p.lng;
  });
  return {
    lat: sumLat / polygon.length,
    lng: sumLng / polygon.length,
  };
}

// ===== ПОСТРОЕНИЕ МАРШРУТА =====
document.getElementById("build-route-btn").addEventListener("click", async () => {
  const startCoords = getFromCoords.get();
  const endCoords = getToCoords.get();

  if (!startCoords || !endCoords) {
    alert("Выберите адреса из списка подсказок");
    return;
  }

  try {
    // Передаём текущее время для ML-прогноза
    const now = new Date();
    const hour = now.getHours();
    const day = now.getDay();  // 0=Sunday, нужно преобразовать в 0=Monday
    const dayOfWeek = day === 0 ? 6 : day - 1;  // Преобразуем: 0=Пн, 6=Вс

    const response = await fetch(`${API_BASE}/api/route`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        start: startCoords,
        end: endCoords,
        mode: currentMode,
        hour: hour,
        day: dayOfWeek,
      }),
    });

    const data = await response.json();

    // Удаляем старый маршрут
    if (routeObject) {
      routeObject.destroy();
      routeObject = null;
    }

    // Если маршрут невозможно построить - показываем предупреждение и не рисуем линию
    if (!data.route_buildable) {
      showMapToast("Невозможно построить безопасный маршрут. Измените точки старта/финиша.");
      const explanationDiv = document.getElementById("ai-explanation");
      explanationDiv.classList.remove("hidden");
      explanationDiv.innerHTML = `
        <div style="margin-bottom: 12px;">
          <b style="font-size: 16px;">🚨 Маршрут невозможно построить</b>
        </div>
        <div style="background: rgba(255,0,0,0.2); padding: 10px; border-radius: 6px; margin-bottom: 12px; border: 2px solid #F44336;">
          <div style="font-size: 13px; line-height: 1.6;">
            ${data.warnings.join('<br>')}
          </div>
        </div>
        <div style="font-size: 12px; color: #aaa;">
          Попробуйте выбрать другие точки старта и финиша, избегая опасных районов.
        </div>
      `;
      return;
    }

    // Цвет линии в зависимости от уровня опасности
    const routeColor =
      data.danger_score <= 3
        ? "#4CAF50" // Зеленый - безопасно
        : data.danger_score <= 6
          ? "#FF9800" // Оранжевый - средне
          : "#F44336"; // Красный - опасно

    // Рисуем полилинию маршрута
    routeObject = new mapgl.Polyline(map, {
      coordinates: data.route.map((p) => [p.lng, p.lat]), // [lng, lat]!
      width: 5,
      color: routeColor,
    });

    // Показать информацию о безопасности маршрута
    const safetyLevel =
      data.danger_score <= 3
        ? "БЕЗОПАСНЫЙ"
        : data.danger_score <= 6
          ? "СРЕДНЕЙ ОПАСНОСТИ"
          : "ОПАСНЫЙ";

    const safetyIcon =
      data.danger_score <= 3 ? "✅" : data.danger_score <= 6 ? "⚠️" : "🚨";

    // Формируем HTML для предупреждений
    const warningsHTML = data.warnings.length > 0
      ? `<div style="background: rgba(255,152,0,0.2); padding: 10px; border-radius: 6px; margin-bottom: 12px; border: 2px solid #FF9800;">
           <div style="font-size: 13px; line-height: 1.6; color: #FFA726;">
             ${data.warnings.join('<br>')}
           </div>
         </div>`
      : '';

    const explanationDiv = document.getElementById("ai-explanation");
    explanationDiv.classList.remove("hidden");
    explanationDiv.innerHTML = `
      <div style="margin-bottom: 12px;">
        <b style="font-size: 16px;">${safetyIcon} Маршрут: ${safetyLevel}</b>
      </div>
      <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 6px; margin-bottom: 12px;">
        <div style="font-size: 13px; margin-bottom: 4px;">
          <b>Уровень опасности:</b> ${data.danger_score.toFixed(1)}/10
        </div>
        <div style="font-size: 12px; color: #aaa;">
          Режим: ${
            currentMode === "car"
              ? "🚗 Автомобиль"
              : currentMode === "child"
                ? "👶 С ребёнком"
                : "🧳 Турист"
          }
        </div>
      </div>
      ${warningsHTML}
      <div style="font-size: 13px; line-height: 1.6;">
        ${data.ai_explanation}
      </div>
    `;

    // Автоматически масштабируем карту на весь маршрут
    const lats = data.route.map((p) => p.lat);
    const lngs = data.route.map((p) => p.lng);
    const minLat = Math.min(...lats),
      maxLat = Math.max(...lats);
    const minLng = Math.min(...lngs),
      maxLng = Math.max(...lngs);

    map.fit([
      [minLng - 0.005, minLat - 0.005], // [lng, lat] — юго-запад
      [maxLng + 0.005, maxLat + 0.005], // [lng, lat] — северо-восток
    ]);

    console.log("✓ Маршрут построен:", data);
  } catch (error) {
    console.error("Ошибка построения маршрута:", error);
    alert("Не удалось построить маршрут");
  }
});

// ===== ГЕОКОДИНГ ЧЕРЕЗ 2GIS API =====

// Популярные места Семея (точные координаты из 2GIS API)
const POPULAR_PLACES = [
  { name: "Площадь Абая", address: "Площадь Абая, Семей", lat: 50.401401, lng: 80.257177 },
  { name: "Центральный парк", address: "Центральный парк, Семей", lat: 50.4092, lng: 80.251695 },
  { name: "Ж/д вокзал Семей", address: "Привокзальная площадь, 1", lat: 50.431783, lng: 80.262986 },
  { name: "Автовокзал", address: "ул. Чокана Валиханова, 167", lat: 50.417208, lng: 80.248 },
  { name: "Университет Шакарима", address: "ул. Глинки, 20а", lat: 50.399279, lng: 80.213045 },
  { name: "Медицинский университет", address: "ул. Абая Кунанбаева, 103", lat: 50.405884, lng: 80.24392 },
  { name: "Family мини-маркет", address: "проспект Шакарима, 150", lat: 50.425707, lng: 80.266045 },
  { name: "Евразия страховая", address: "ул. Миржакипа Дулатова, 135", lat: 50.408093, lng: 80.255193 },
  { name: "Рынок Акшын", address: "Рынок Акшын, Семей", lat: 50.418192, lng: 80.24843 },
  { name: "Драматический театр им. Абая", address: "Муз.-драм. театр им. Абая, Семей", lat: 50.405668, lng: 80.250347 },
  { name: "Музей Достоевского", address: "Музей Достоевского, Семей", lat: 50.404296, lng: 80.251854 },
  { name: "Больница №1", address: "ул. Богенбай батыра, 134", lat: 50.40693, lng: 80.255409 },
];

// ===== ПОИСК ЧЕРЕЗ NOMINATIM (OpenStreetMap) =====

// Поиск адреса через Nominatim (бесплатный, быстрый, точный)
async function searchNominatim(query) {
  if (!query || query.length < 2) return [];

  try {
    const url = `${API_BASE}/api/geocode?q=${encodeURIComponent(query)}`;
    const response = await fetch(url);

    if (!response.ok) {
      console.warn(`Nominatim search error: ${response.status}`);
      return [];
    }

    const data = await response.json();
    return (data.results || []).map(item => ({
      name: item.name,
      address: item.address,
      lat: item.lat,
      lng: item.lng,
    }));

  } catch (error) {
    console.error("Ошибка Nominatim поиска:", error);
    return [];
  }
}

// Обратное геокодирование (координаты → адрес)
async function reverseGeocode(lat, lng) {
  try {
    const url = `${API_BASE}/api/geocode/reverse?lat=${lat}&lng=${lng}`;
    const response = await fetch(url);

    if (!response.ok) {
      return { name: "Моё местоположение", address: "", lat, lng };
    }

    return await response.json();

  } catch (error) {
    console.error("Ошибка обратного геокодирования:", error);
    return { name: "Моё местоположение", address: "", lat, lng };
  }
}

// Кэш результатов поиска
const searchCache = new Map();
let searchTimeout = null;

// Границы города Семей для фильтрации результатов
const SEMEY_BOUNDS = {
  minLat: 50.35,
  maxLat: 50.48,
  minLng: 80.15,
  maxLng: 80.35,
};

// Поиск адреса через Nominatim + 2GIS (объединённые результаты)
async function searchAddress(query) {
  if (!query || query.length < 2) return [];

  // Проверка кэша
  const cacheKey = query.toLowerCase().trim();
  if (searchCache.has(cacheKey)) {
    return searchCache.get(cacheKey);
  }

  try {
    // Параллельный поиск через Nominatim и 2GIS
    const [nominatimResults, twogisResults] = await Promise.allSettled([
      searchNominatim(query),
      search2GIS(query),
    ]);

    const nomResults = nominatimResults.status === "fulfilled" ? nominatimResults.value : [];
    const twoResults = twogisResults.status === "fulfilled" ? twogisResults.value : [];

    // Объединяем: Nominatim приоритетнее (быстрее и точнее для адресов)
    const allResults = [...nomResults, ...twoResults];

    // Убираем дубликаты по близости координат
    const uniqueResults = allResults.filter(
      (result, index, self) =>
        index ===
        self.findIndex(
          (r) =>
            Math.abs(r.lat - result.lat) < 0.0001 &&
            Math.abs(r.lng - result.lng) < 0.0001
        )
    );

    const results = uniqueResults.slice(0, 10);

    // Сохранение в кэш
    searchCache.set(cacheKey, results);
    return results;

  } catch (error) {
    console.error("Ошибка поиска адреса:", error);
    return [];
  }
}

// Поиск через 2GIS (как fallback, Nominatim — основной)
async function search2GIS(query) {
  if (!query || query.length < 2) return [];

  try {
    const searchQuery = query.toLowerCase().includes("семей") || query.toLowerCase().includes("semey")
      ? query
      : `${query}, Семей, Казахстан`;

    const url = `${API_BASE}/api/search?q=${encodeURIComponent(searchQuery)}`;
    const response = await fetch(url);

    if (!response.ok) return [];

    const data = await response.json();
    if (!data.result || !data.result.items) return [];

    return data.result.items
      .filter((item) => item.point)
      .map((item) => {
        const fullText = (item.full_name || item.address_name || "").toLowerCase();
        const admDivText = (item.adm_div || []).map(d => d.name).join(" ").toLowerCase();
        const isSemey = fullText.includes("семей") || admDivText.includes("семей") || fullText.includes("semey");
        const inSemeyBounds =
          item.point.lat >= SEMEY_BOUNDS.minLat &&
          item.point.lat <= SEMEY_BOUNDS.maxLat &&
          item.point.lon >= SEMEY_BOUNDS.minLng &&
          item.point.lon <= SEMEY_BOUNDS.maxLng;

        const address = item.address_name || item.street || item.full_name || "";

        return {
          name: item.name || item.address_name || "Неизвестное место",
          address: address,
          lat: item.point.lat,
          lng: item.point.lon,
          isSemey: isSemey,
          inSemeyBounds: inSemeyBounds,
          score: (isSemey ? 100 : 0) + (inSemeyBounds ? 50 : 0),
        };
      })
      .sort((a, b) => b.score - a.score)
      .filter((item) => item.isSemey || item.inSemeyBounds)
      .map(({ name, address, lat, lng }) => ({ name, address, lat, lng }));

  } catch (error) {
    console.error("Ошибка 2GIS поиска:", error);
    return [];
  }
}

// Фильтрация популярных мест по запросу
function filterPopularPlaces(query) {
  if (!query || query.length < 2) return POPULAR_PLACES;

  const q = query.toLowerCase();
  return POPULAR_PLACES.filter(
    (place) =>
      place.name.toLowerCase().includes(q) ||
      place.address.toLowerCase().includes(q)
  );
}

// Показ результатов в dropdown с улучшенным форматированием
function showSearchResults(dropdownId, results, onSelect) {
  const dropdown = document.getElementById(dropdownId);
  if (!dropdown) return;

  if (results.length === 0) {
    dropdown.innerHTML = '<div class="search-empty">Ничего не найдено</div>';
    dropdown.classList.add("active");
    return;
  }

  dropdown.innerHTML = results
    .map((r, i) => {
      // Форматируем адрес - убираем дублирование
      let displayAddress = r.address || "";
      if (displayAddress === r.name) displayAddress = "";

      return `
        <div class="search-item" data-index="${i}">
          <div class="search-item-name">${r.name}</div>
          ${displayAddress ? `<div class="search-item-address">${displayAddress}</div>` : ""}
        </div>
      `;
    })
    .join("");

  dropdown.classList.add("active");

  // Обработчики клика
  dropdown.querySelectorAll(".search-item").forEach((item, i) => {
    item.addEventListener("click", () => {
      onSelect(results[i]);
      dropdown.classList.remove("active");
    });
  });
}

// Настройка автокомплита для поля с улучшенной навигацией + геолокация
function setupAutocomplete(inputId, dropdownId, markerColor) {
  const input = document.getElementById(inputId);
  const dropdown = document.getElementById(dropdownId);
  if (!input || !dropdown) {
    const noop = { get: () => null, set: () => {} };
    return noop;
  }

  let selectedCoords = null;
  let marker = null;
  let selectedIndex = -1;
  let currentResults = [];

  // Функция выделения элемента в списке
  function highlightItem(index) {
    const items = dropdown.querySelectorAll(".search-item");
    items.forEach((item, i) => {
      item.classList.toggle("search-item-highlight", i === index);
    });
    selectedIndex = index;

    // Прокрутка к выделенному элементу
    if (items[index]) {
      items[index].scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }

  // Функция выбора элемента
  function selectItem(result) {
    input.value = result.name;
    selectedCoords = { lat: result.lat, lng: result.lng };
    dropdown.classList.remove("active");
    selectedIndex = -1;

    // Показать маркер на карте
    if (marker) {
      marker.destroy();
      marker = null;
    }
    marker = new mapgl.HtmlMarker(map, {
      coordinates: [result.lng, result.lat],
      html: `<div style="background:${markerColor};width:24px;height:24px;border-radius:50%;border:3px solid white;box-shadow:0 2px 6px rgba(0,0,0,0.3);"></div>`,
    });

    map.setCenter([result.lng, result.lat], 14);
  }

  // Обработка клавиш клавиатуры
  input.addEventListener("keydown", (e) => {
    const items = dropdown.querySelectorAll(".search-item");
    if (!dropdown.classList.contains("active") || items.length === 0) return;

    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        highlightItem(selectedIndex < items.length - 1 ? selectedIndex + 1 : 0);
        break;
      case "ArrowUp":
        e.preventDefault();
        highlightItem(selectedIndex > 0 ? selectedIndex - 1 : items.length - 1);
        break;
      case "Enter":
        e.preventDefault();
        if (selectedIndex >= 0 && currentResults[selectedIndex]) {
          selectItem(currentResults[selectedIndex]);
        }
        break;
      case "Escape":
        dropdown.classList.remove("active");
        input.blur();
        break;
    }
  });

  // Мгновенное появление при фокусе - показываем популярные места
  input.addEventListener("focus", () => {
    const query = input.value.trim();
    const popular = filterPopularPlaces(query);
    currentResults = popular;

    showSearchResults(dropdownId, popular, (result) => selectItem(result));
  });

  // Обработчик ввода с debounce
  input.addEventListener("input", (e) => {
    const query = e.target.value.trim();
    selectedCoords = null;
    selectedIndex = -1;

    if (searchTimeout) clearTimeout(searchTimeout);

    if (query.length < 2) {
      // Показываем все популярные места
      const popular = filterPopularPlaces("");
      currentResults = popular;
      showSearchResults(dropdownId, popular, (result) => selectItem(result));
      return;
    }

    searchTimeout = setTimeout(async () => {
      dropdown.innerHTML = '<div class="search-loading">Поиск...</div>';
      dropdown.classList.add("active");

      // Сначала фильтруем популярные места
      const popular = filterPopularPlaces(query);

      // Затем ищем в 2GIS
      const twogisResults = await searchAddress(query);

      // Объединяем результаты (популярные первыми)
      const allResults = [...popular, ...twogisResults];

      // Убираем дубликаты по координатам
      const uniqueResults = allResults.filter(
        (result, index, self) =>
          index ===
          self.findIndex(
            (r) =>
              Math.abs(r.lat - result.lat) < 0.0001 &&
              Math.abs(r.lng - result.lng) < 0.0001
          )
      );

      currentResults = uniqueResults;
      showSearchResults(dropdownId, uniqueResults, (result) => selectItem(result));
    }, SEARCH_DEBOUNCE_MS);
  });

  // Закрытие dropdown при клике вне
  document.addEventListener("click", (e) => {
    if (!input.contains(e.target) && !dropdown.contains(e.target)) {
      dropdown.classList.remove("active");
      selectedIndex = -1;
    }
  });

  // Возвращаем объект с методами get/set для координат
  return {
    get: () => selectedCoords,
    set: (coords, name = "Моё местоположение") => {
      selectedCoords = coords;
      input.value = name;

      // Создаём маркер
      if (marker) {
        marker.destroy();
        marker = null;
      }

      marker = new mapgl.HtmlMarker(map, {
        coordinates: [coords.lng, coords.lat],
        html: `<div style="background:${markerColor};width:24px;height:24px;border-radius:50%;border:3px solid white;box-shadow:0 2px 6px rgba(0,0,0,0.3);display:flex;align-items:center;justify-content:center;font-size:12px;">📍</div>`,
      });

      // Центрируем карту
      map.setCenter([coords.lng, coords.lat], 15);

      // Закрываем dropdown
      dropdown.classList.remove("active");
    }
  };
}

// Настройка автокомплита для полей "Откуда" и "Куда"
const getFromCoords = setupAutocomplete("input-from", "dropdown-from", "#4CAF50");
const getToCoords = setupAutocomplete("input-to", "dropdown-to", "#F44336");

// ===== ГЕОЛОКАЦИЯ (Моё местоположение) =====

// Обработчики кнопок геолокации
function setupGeolocation(inputId, buttonId, dropdownId, markerColor) {
  const input = document.getElementById(inputId);
  const button = document.getElementById(buttonId);
  const dropdown = document.getElementById(dropdownId);

  if (!input || !button || !dropdown) return;

  button.addEventListener("click", async (e) => {
    e.stopPropagation();

    if (!navigator.geolocation) {
      showMapToast("Геолокация не поддерживается вашим браузером");
      return;
    }

    // Визуальная индикация загрузки
    button.classList.add("loading");
    button.disabled = true;
    showMapToast("Определение местоположения...");

    try {
      const position = await new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(resolve, reject, {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 60000,
        });
      });

      const lat = position.coords.latitude;
      const lng = position.coords.longitude;

      // Проверяем что пользователь в Семею
      if (lat < SEMEY_BOUNDS.minLat || lat > SEMEY_BOUNDS.maxLat ||
          lng < SEMEY_BOUNDS.minLng || lng > SEMEY_BOUNDS.maxLng) {
        showMapToast("Вы находитесь за пределами Семея");
        button.classList.remove("loading");
        button.disabled = false;
        return;
      }

      // Обратное геокодирование
      const address = await reverseGeocode(lat, lng);
      const displayName = address.name || address.address || "Моё местоположение";

      // Используем новый метод set() для установки координат и обновления UI
      const coordSetter = inputId === "input-from" ? getFromCoords.set : getToCoords.set;
      coordSetter({ lat, lng }, displayName);

      showMapToast(`Местоположение определено: ${address.address || "Семей"}`);

    } catch (error) {
      console.error("Ошибка геолокации:", error);
      let message = "Не удалось определить местоположение";
      if (error.code === error.PERMISSION_DENIED) {
        message = "Доступ к геолокации запрещён";
      } else if (error.code === error.TIMEOUT) {
        message = "Превышено время ожидания";
      }
      showMapToast(message);
    } finally {
      button.classList.remove("loading");
      button.disabled = false;
    }
  });
}

setupGeolocation("input-from", "geolocate-from-btn", "dropdown-from", "#4CAF50");
setupGeolocation("input-to", "geolocate-to-btn", "dropdown-to", "#F44336");

// ===== УПРАВЛЕНИЕ РАЙОНАМИ =====
const districtsModal = document.getElementById('districts-modal');
const manageBtn = document.getElementById('manage-districts-btn');
const closeBtn = document.querySelector('.close-modal-btn');
const saveBtn = document.getElementById('save-districts-btn');
const resetBtn = document.getElementById('reset-districts-btn');
const districtsList = document.getElementById('districts-list');

let originalDistricts = [];

// Открытие модального окна
manageBtn.addEventListener('click', async () => {
  await loadDistrictsForEdit();
  districtsModal.classList.remove('hidden');
});

// Закрытие модального окна
closeBtn.addEventListener('click', () => {
  districtsModal.classList.add('hidden');
});

// Закрытие при клике вне модального окна
districtsModal.addEventListener('click', (e) => {
  if (e.target === districtsModal) {
    districtsModal.classList.add('hidden');
  }
});

// Загрузка районов для редактирования
async function loadDistrictsForEdit() {
  try {
    const response = await fetch('http://localhost:8000/api/districts');
    const districts = await response.json();

    originalDistricts = JSON.parse(JSON.stringify(districts)); // Глубокое копирование

    districtsList.innerHTML = '';

    districts.forEach(district => {
      const item = document.createElement('div');
      item.className = 'district-item';
      item.dataset.id = district.id;

      item.innerHTML = `
        <div class="district-item-header">
          <div class="district-item-name">${district.name}</div>
        </div>
        <div class="district-item-description">${district.description || ''}</div>
        <div class="danger-level-control">
          <span class="danger-level-label">Уровень опасности:</span>
          <input
            type="number"
            class="danger-level-input"
            min="1"
            max="10"
            value="${district.danger_level}"
            data-district-id="${district.id}"
          >
          <div class="danger-level-preview" style="background-color: ${getDangerColor(district.danger_level)}">
            ${district.danger_level}
          </div>
        </div>
      `;

      const input = item.querySelector('.danger-level-input');
      const preview = item.querySelector('.danger-level-preview');

      input.addEventListener('input', (e) => {
        const value = parseInt(e.target.value) || 1;
        const clampedValue = Math.max(1, Math.min(10, value));
        preview.textContent = clampedValue;
        preview.style.backgroundColor = getDangerColor(clampedValue);
      });

      districtsList.appendChild(item);
    });
  } catch (error) {
    console.error('Ошибка загрузки районов:', error);
    alert('Не удалось загрузить районы');
  }
}

// Сохранение изменений
saveBtn.addEventListener('click', async () => {
  const inputs = districtsList.querySelectorAll('.danger-level-input');
  const updates = {};

  inputs.forEach(input => {
    const districtId = input.dataset.districtId;
    const value = parseInt(input.value) || 1;
    const clampedValue = Math.max(1, Math.min(10, value));

    // Проверяем, изменилось ли значение
    const original = originalDistricts.find(d => d.id === districtId);
    if (original && original.danger_level !== clampedValue) {
      updates[districtId] = clampedValue;
    }
  });

  if (Object.keys(updates).length === 0) {
    alert('Нет изменений для сохранения');
    districtsModal.classList.add('hidden');
    return;
  }

  try {
    const response = await fetch('http://localhost:8000/api/districts/batch-update', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(updates)
    });

    if (response.ok) {
      const result = await response.json();
      alert(`Успешно обновлено районов: ${result.updated}`);
      districtsModal.classList.add('hidden');

      // Перезагружаем районы на карте
      await loadDistricts();
    } else {
      throw new Error('Ошибка сохранения');
    }
  } catch (error) {
    console.error('Ошибка сохранения:', error);
    alert('Не удалось сохранить изменения');
  }
});

// Сброс изменений
resetBtn.addEventListener('click', () => {
  originalDistricts.forEach(original => {
    const input = districtsList.querySelector(`[data-district-id="${original.id}"]`);
    const preview = input.parentElement.querySelector('.danger-level-preview');

    if (input) {
      input.value = original.danger_level;
      preview.textContent = original.danger_level;
      preview.style.backgroundColor = getDangerColor(original.danger_level);
    }
  });
});

// ===== ИНИЦИАЛИЗАЦИЯ =====
// Ждем загрузки DOM, затем инициализируем карту и загружаем данные
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', startApp);
} else {
  // DOM уже загружен
  startApp();
}

async function startApp() {
  console.log("🚀 Запуск приложения SafeRoute AI...");

  await initMap();

  if (map) {
    console.log("✓ Карта инициализирована, загружаю данные...");
    loadDistricts();
    loadMLInfo();
  } else {
    console.error("❌ Не удалось инициализировать карту");
  }
}
