import sqlite3
import pandas as pd
import re
from market_intelligence_platform import MarketIntelligencePlatform

DB_PATH = 'supplement_market.db'

# Load platform for categorization logic
tool = MarketIntelligencePlatform(DB_PATH)

def fix_price(price):
    if price is None:
        return None
    if isinstance(price, (int, float)):
        return float(price)
    # Convert string price with comma or dot
    price_str = str(price).strip()
    # If price is like '89,90', convert to '89.90'
    if re.match(r'^\d{1,3}(?:[.,]\d{3})*[.,]\d+$', price_str):
        price_str = price_str.replace('.', '').replace(',', '.') if ',' in price_str else price_str
    try:
        return float(price_str)
    except Exception:
        return None

def main():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM products", conn)
    print(f"Loaded {len(df)} products from database.")

    # Clean prices and categories
    updated = 0
    for idx, row in df.iterrows():
        orig_price = row['price']
        fixed_price = fix_price(orig_price)
        # Re-categorize
        name = row['name']
        desc = row.get('description', '')
        ingredients = row.get('ingredients', '')
        new_category = tool.categorize_product(name, desc, ingredients)
        # Only update if changed
        if fixed_price != orig_price or new_category != row['category']:
            updated += 1
            df.at[idx, 'price'] = fixed_price
            df.at[idx, 'category'] = new_category

    print(f"Products needing update: {updated}")
    # Write back to DB
    cursor = conn.cursor()
    for idx, row in df.iterrows():
        cursor.execute('''
            UPDATE products SET price=?, category=? WHERE id=?
        ''', (row['price'], row['category'], row['id']))
    conn.commit()
    conn.close()
    print("✅ Database cleaned and updated.")

if __name__ == "__main__":
    main() 