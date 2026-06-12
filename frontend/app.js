// Backend URL
const API_BASE = "http://localhost:8000";

// Инициализация карты
const map = L.map("map").setView([50.4111, 80.2275], 13);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

// Загрузка зон опасности
async function loadZones() {
  try {
    const response = await fetch(`${API_BASE}/api/zones`);
    const zones = await response.json();

    zones.forEach((zone) => {
      const color = getDangerColor(zone.danger_level);
      L.circle([zone.lat, zone.lng], {
        color: color,
        fillColor: color,
        fillOpacity: 0.3,
        radius: zone.radius,
      }).addTo(map).bindPopup(`
                <b>${zone.name}</b><br>
                Уровень опасности: ${zone.danger_level}/10<br>
                ${zone.description}
            `);
    });

    console.log(`✓ Загружено ${zones.length} зон`);
  } catch (error) {
    console.error("Ошибка загрузки зон:", error);
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

// Инициализация
loadZones();
loadMLInfo();
