import pandas as pd

def save_to_parquet(character_data, filename="swgoh_characters.parquet"):
    """
    Takes the list of dictionaries from get_swgoh_characters 
    and saves it to a Parquet file.
    """
    if not character_data:
        print("No data provided to save.")
        return

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(character_data)

    try:
        # Save to parquet using the pyarrow engine
        df.to_parquet(filename, engine='pyarrow', index=False)
        print(f"Successfully saved {len(df)} characters to {filename}")
    except Exception as e:
        print(f"Failed to save parquet: {e}")