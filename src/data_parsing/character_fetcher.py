import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd

def get_swgoh_characters():
    scraper = cloudscraper.create_scraper(
        browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
    )

    url = "https://swgoh.gg/characters/"

    try:
        response = scraper.get(url)
        if response.status_code != 200:
            print(f"Blocked or error: {response.status_code}")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        char_cells = soup.find_all('div', class_='unit-card-grid__cell')
        
        characters = []
        for cell in char_cells:
            # 1. Basic Info
            name_div = cell.find('div', class_='unit-card__name')
            name = name_div.get_text(strip=True) if name_div else "Unknown"
            
            link_tag = cell.find('a', href=True)
            link = "https://swgoh.gg" + link_tag['href'] if link_tag else None

            # 2. Extract specific "Tags" from the text
            cats_div = cell.find('div', class_='unit-card__cats')
            tags = []
            if cats_div:
                # Split by bullet points and clean up
                tags = [t.strip() for t in cats_div.get_text().split('•') if t.strip()]

            # 3. Handle Alignment and Galactic Legend via CSS Classes
            # We look at the div with the class "unit-card"
            card_div = cell.find('div', class_='unit-card')
            if card_div:
                classes = card_div.get('class', [])
                
                # Alignment check
                if 'unit-card--alignment-2' in classes:
                    tags.append("Light Side")
                elif 'unit-card--alignment-3' in classes:
                    tags.append("Dark Side")
                elif 'unit-card--alignment-1' in classes:
                    tags.append("Neutral")
                
                # Galactic Legend check
                if 'unit-card--is-galactic-legend' in classes:
                    tags.append("Galactic Legend")

            characters.append({
                'name': name,
                'character_url': link,
                'tags': tags
            })
        
        return characters

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    import os
    from parquet_utils import save_to_parquet

    data_path = os.path.join(os.path.dirname(__file__), "../../data")
    # 1. Fetch the data using your cloudscraper method
    data = get_swgoh_characters()
    
    # 2. Save it
    if data:
        save_to_parquet(data, os.path.join(data_path, "swgoh_units.parquet"))