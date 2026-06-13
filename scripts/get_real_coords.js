// Скрипт для получения реальных координат мест Семея через 2GIS API
const TWOGIS_API_KEY = "2db79fdb-a7bc-43c4-8e90-5ca828ef5449";

const PLACES_TO_FIND = [
  "Площадь Абая, Семей",
  "Центральный парк, Семей",
  "Железнодорожный вокзал, Семей",
  "Автостанция Семей",
  "Семейский государственный университет Шакарима",
  "Медицинский университет, Семей",
  "ТЦ Family, Семей",
  "ТЦ Евразия, Семей",
  "Центральный рынок, Семей",
  "Драматический театр, Семей",
  "Музей Достоевского, Семей",
  "Больница №1, Семей",
];

async function searchPlace(query) {
  try {
    const url = `https://catalog.api.2gis.com/3.0/items?q=${encodeURIComponent(query)}&region_id=77&fields=items.point,items.adm_div,items.address_name,items.name&key=${TWOGIS_API_KEY}&page_size=1`;
    const response = await fetch(url);
    const data = await response.json();

    if (!data.result || !data.result.items || data.result.items.length === 0) {
      console.log(`❌ Не найдено: ${query}`);
      return null;
    }

    const item = data.result.items[0];
    if (!item.point) {
      console.log(`❌ Нет координат: ${query}`);
      return null;
    }

    return {
      name: item.name || item.address_name || query.split(",")[0],
      address: item.address_name || item.full_name || "",
      lat: item.point.lat,
      lng: item.point.lon,
    };
  } catch (error) {
    console.error(`❌ Ошибка поиска ${query}:`, error);
    return null;
  }
}

async function main() {
  console.log("🔍 Поиск реальных координат мест Семея через 2GIS...\n");

  const results = [];

  for (const place of PLACES_TO_FIND) {
    const result = await searchPlace(place);
    if (result) {
      results.push(result);
      console.log(`✓ ${result.name}`);
      console.log(`  Адрес: ${result.address}`);
      console.log(`  Координаты: lat: ${result.lat}, lng: ${result.lng}\n`);
    }
    await new Promise((r) => setTimeout(r, 200)); // Задержка между запросами
  }

  console.log("\n📋 Готовый массив POPULAR_PLACES:\n");
  console.log("const POPULAR_PLACES = [");
  results.forEach((r, i) => {
    const comma = i < results.length - 1 ? "," : "";
    console.log(
      `  { name: "${r.name}", address: "${r.address}", lat: ${r.lat}, lng: ${r.lng} }${comma}`
    );
  });
  console.log("];");
}

main().catch(console.error);
