import cloudscraper
from bs4 import BeautifulSoup
import re
from parquet_utils import save_to_parquet

def parse_character_details(character_url):
    scraper = cloudscraper.create_scraper(
        browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
    )
    
    base_url = character_url.rstrip('/')
    
    try:
        response = scraper.get(base_url + "/")
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. Predictable URL Construction
        mods_data_url = f"{base_url}/data/mods/?filter_type=guilds_100_gp"
        player_data_url = f"{base_url}/data/stats/?filter_type=guilds_100_gp"

        # 2. Extract Stats
        # In eg_char.html, stats are in 'unit-stat-row' within the summary or sidebars
        base_stats = {}
        for row in soup.select('.stat-table-data__entry'):
            label = row.select_one('.stat-table-data__entry-primary-label')
            value = row.select_one('.stat-table-data__entry-primary-value')
            if label and value:
                base_stats[label.get_text(strip=True)] = value.get_text(strip=True)

        # 3. Extract Ability Classes (the global list for the character)
        ability_classes = []
        ac_header = soup.find('h4', string=re.compile(r'Ability Classes', re.I))
        if ac_header:
            # Classes are in the immediate sibling 'div' container
            ac_container = ac_header.find_next_sibling('div')
            if ac_container:
                ability_classes = [a.get_text(strip=True) for a in ac_container.find_all('a')]

        # 4. Extract Individual Abilities
        abilities = []
        # Abilities are wrapped in 'list-group-item-wrapper' or 'ability-card'
        for idx, card in enumerate(soup.select('.unit-ability__header')):
            is_zeta = False
            is_omicron = False
            is_ultimate = False

            name_tag = card.select_one('.unit-ability__name')
            if not name_tag:
                continue
            ability_name = name_tag.get_text(strip=True)

            href = name_tag.find('a')['href'] if name_tag.find('a') else None
            if href:
                if 'basicability' in href:
                    ability_type = 'Basic'
                elif 'specialability' in href:
                    ability_type = 'Special'
                elif 'uniqueability' in href:
                    ability_type = 'Unique'
                elif 'leaderability' in href:
                    ability_type = 'Leader'
                elif 'ultimateability' in href:
                    ability_type = 'Ultimate'
            
            # Ability Description
            desc_tag = soup.select('.unit-ability__description')[idx]
            
            # Badge Detection (Zeta/Omicron)
            # These are usually span elements with specific classes
            ability_material = card.find('div', class_='unit-ability__header-aside') if card.find('div', class_='unit-ability__header-aside') else None
            if ability_material:
                # Check span for Zeta/Omicron
                for span in ability_material.find_all('span'):
                    if 'Zeta' in span.get('title', ''):
                        is_zeta = True
                    elif 'Omicron' in span.get('title', ''):
                        is_omicron = True

                # Check inner div for Ultimate
                generic_item = ability_material.find('div', class_='generic-item')
                is_ultimate = generic_item and 'Ultimate' in generic_item.get('title', '')

            abilities.append({
                'ability_name': ability_name,
                'ability_type': ability_type,
                'description': desc_tag.get_text(strip=True) if desc_tag else "",
                'breakdown_link': "https://swgoh.gg" + href,
                'is_zeta': is_zeta,
                'is_omicron': is_omicron,
                'is_ultimate': is_ultimate
            })

        return {
            'character_url': character_url,
            'base_stats': base_stats,
            'ability_classes': ability_classes,
            'abilities': abilities,
            'mods_data_url': mods_data_url,
            'player_data_url': player_data_url
        }
    except Exception as e:
        print(f"Error parsing {character_url}: {e}")
        return None


if __name__ == "__main__":
    import tqdm
    import pandas as pd
    import os
    from parquet_utils import save_to_parquet
    import time

    # url = "https://swgoh.gg/units/jedi-master-mace-windu/"
    # data = parse_character_details(url)
    # if data:
    #     print(f"Base Stats: {data['base_stats']}")
    #     print(f"Abilities: {data['abilities']}")

    data_path = os.path.join(os.path.dirname(__file__), "../../data")
    df_all = pd.read_parquet(os.path.join(data_path, "swgoh_units.parquet"))

    character_details = []
    for idx, row in tqdm.tqdm(df_all.iterrows(), total=df_all.shape[0], desc="Parsing characters",
                              unit="character", unit_scale=True, unit_divisor=1, leave=True):
        char_data = parse_character_details(row['character_url'])
        if char_data:
            character_details.append(char_data)
        time.sleep(2)
    save_to_parquet(character_details, os.path.join(data_path, "character_details.parquet"))