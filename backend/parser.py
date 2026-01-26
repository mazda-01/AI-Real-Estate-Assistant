import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
import random
import re

# ────────────────────────────────────────────────
# Настройки
# ────────────────────────────────────────────────
TOTAL_PAGES = 50             # начни с малого, чтобы не словить бан
MIN_PRICE = 5_000_000
MAX_PRICE = 30_000_000
ROOMS = [1, 2, 3, 4]

output_file = "data.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

print(f"Файл будет сохранён как: {output_file}")
print(f"Парсим {TOTAL_PAGES} страниц, цена {MIN_PRICE:,} – {MAX_PRICE:,} ₽\n")

# Параметры поиска
params = {
    "deal_type": "sale",
    "engine_version": "2",
    "offer_type": "flat",
    "region": "1",                  # Москва
    "minprice": str(MIN_PRICE),
    "maxprice": str(MAX_PRICE),
}
for r in ROOMS:
    params[f"room{r}"] = "1"

# ────────────────────────────────────────────────
# Основной цикл
# ────────────────────────────────────────────────
session = requests.Session()
all_flats = []
failed = []

for page in tqdm(range(1, TOTAL_PAGES + 1), desc="страницы"):
    params["p"] = page

    try:
        r = session.get("https://www.cian.ru/cat.php", params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        # Пытаемся найти карточки любыми способами
        cards = (
            soup.select('article[data-name="CardComponent"]') or
            soup.select('div[data-testid="offer-card"]') or
            soup.select('div[data-marker*="Card"]') or
            soup.select('article') or
            soup.select('div[class*="card"], div[class*="offer"]')
        )

        if not cards:
            print(f"  стр {page:2d}  → карточки не найдены")
            failed.append(page)
            time.sleep(random.uniform(10, 18))
            continue

        print(f"  стр {page:2d}  → найдено карточек: {len(cards)}")

        for card in cards:
            flat = {}

            # Цена
            price_el = card.select_one('[data-mark="MainPrice"], [data-testid="price"], span[class*="price"], div[class*="price"]')
            flat["price"] = price_el.get_text(strip=True).replace("\xa0", " ").strip() if price_el else "—"

            # Адрес
            addr_el = card.select_one('[data-name="GeoLabel"], [data-testid="address"], a[href*="/map/"], div[class*="address"]')
            flat["address"] = addr_el.get_text(strip=True).replace("\xa0", " ").strip() if addr_el else "—"

            # Получаем ВЕСЬ видимый текст карточки
            text_parts = [t.strip() for t in card.stripped_strings if t.strip() and len(t.strip()) > 1]
            full_text = " ".join(text_parts).replace("\xa0", " ")

            # Заголовок — обычно содержит комнаты + площадь
            title = "—"
            for part in text_parts[:5]:  # первые 5 строк чаще всего содержат главное
                if "м²" in part or "комнат" in part.lower() or "студия" in part.lower():
                    title = part
                    break
            if title == "—":
                title = text_parts[0] if text_parts else "—"
            flat["title"] = title

            # Регулярки — основная надежда
            rooms = "—"
            area_m2 = "—"
            floor = "—"

            # Комнаты
            m_rooms = re.search(r"(?:(\d+)[-\s]?комн(?:атн(?:ая|ых?))?|студия)", full_text, re.I)
            if m_rooms:
                rooms = m_rooms.group(1) if m_rooms.group(1) else m_rooms.group(0)

            # Площадь
            m_area = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:м²|м2|m²|кв\.?м)", full_text, re.I)
            if m_area:
                area_m2 = m_area.group(1).replace(",", ".") + " м²"

            # Этаж (если повезёт)
            m_floor = re.search(r"(\d+(?:/\d+)?)\s*(?:этаж|эт\.?|/|из)", full_text, re.I)
            if m_floor:
                floor = m_floor.group(1)

            flat["rooms"] = rooms
            flat["total_area_m2"] = area_m2
            flat["floor"] = floor

            # Ссылка
            link = card.select_one('a[href^="/sale/flat/"], a[data-name="Link"], a[data-marker*="title"]')
            if link and "href" in link.attrs:
                href = link["href"]
                flat["url"] = "https://www.cian.ru" + href if href.startswith("/") else href
            else:
                flat["url"] = "—"

            if flat["price"] != "—" and flat["url"] != "—":
                all_flats.append(flat)

        time.sleep(random.uniform(5.5, 12.5))

    except Exception as e:
        print(f"  стр {page:2d}  → ошибка: {str(e)[:80]}")
        failed.append(page)
        time.sleep(15)

# ────────────────────────────────────────────────
# Сохранение
# ────────────────────────────────────────────────
print("\n" + "═" * 60)

if all_flats:
    df = pd.DataFrame(all_flats)
    cols = ["price", "rooms", "total_area_m2", "floor", "address", "title", "url"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Сохранено → {output_file}")
    print(f"Записей: {len(df):,}")
    if failed:
        print(f"Не удалось страницы: {failed}")
else:
    print("Ничего не собрано → скорее всего блокировка или сильное изменение структуры")

print("═" * 60)