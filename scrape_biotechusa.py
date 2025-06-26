import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from urllib.parse import urljoin

class BioTechUSAEnhancedScraper:
    def __init__(self):
        self.base_url = 'https://shop.biotechusa.com'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Category-wise data storage
        self.category_data = {
            'Protein': [],
            'Nutraceuticals': [],
            'Bars': [],
            'Apparel': []
        }

    def categorize_product(self, product_name, product_description="", ingredients=""):
        """Enhanced categorization with ingredient-based detection"""
        name_lower = product_name.lower()
        desc_lower = product_description.lower()
        ingredients_lower = ingredients.lower()
        full_text = f"{name_lower} {desc_lower} {ingredients_lower}"
        
        # Enhanced apparel keywords
        apparel_keywords = [
            'shirt', 't-shirt', 'tank top', 'tank', 'hoodie', 'sweatshirt', 'sweater',
            'jacket', 'shorts', 'pants', 'leggings', 'joggers', 'tracksuit',
            'cap', 'hat', 'beanie', 'bag', 'backpack', 'bottle', 'shaker', 'towel',
            'apparel', 'clothing', 'wear', 'gloves', 'socks', 'underwear', 'vest',
            'polo', 'zip', 'pullover', 'crew neck', 'v-neck', 'tee', 'jersey'
        ]
        
        # Enhanced supplement keywords with ingredient-based detection
        supplement_keywords = [
            'capsule', 'tablet', 'softgel', 'pill', 'vitamin', 'mineral', 'supplement',
            'mg', 'mcg', 'iu', 'daily', 'complex', 'formula', 'extract', 'acid',
            'collagen', 'omega', 'fish oil', 'probiotic', 'enzyme', 'antioxidant'
        ]
        
        # Ingredient-based protein detection
        protein_ingredients = [
            'whey protein', 'casein protein', 'whey isolate', 'whey concentrate',
            'soy protein', 'pea protein', 'hemp protein', 'rice protein',
            'milk protein', 'egg protein', 'beef protein'
        ]
        
        # Ingredient-based nutraceutical detection
        nutraceutical_ingredients = [
            'vitamin c', 'vitamin d', 'vitamin b', 'vitamin a', 'vitamin e',
            'calcium', 'magnesium', 'zinc', 'iron', 'potassium',
            'collagen peptides', 'hyaluronic acid', 'coenzyme q10',
            'omega-3', 'fish oil', 'evening primrose oil',
            'probiotics', 'lactobacillus', 'bifidobacterium',
            'green tea extract', 'grape seed extract', 'turmeric extract'
        ]
        
        # Bar-specific ingredients
        bar_ingredients = [
            'protein crisp', 'chocolate coating', 'caramel layer',
            'nuts', 'almonds', 'peanuts', 'cashews', 'dates',
            'oats', 'rice crisps', 'cocoa powder'
        ]
        
        # Check for apparel first (most specific)
        if any(keyword in name_lower for keyword in apparel_keywords):
            return 'Apparel'
        
        # Check for bars (nutrition bars)
        bar_keywords = ['bar', 'bite', 'snack bar']
        if (any(keyword in name_lower for keyword in bar_keywords) or 
            any(ingredient in ingredients_lower for ingredient in bar_ingredients)):
            return 'Bars'
        
        # Check for protein supplements using both name and ingredients
        protein_keywords = [
            'whey', 'casein', 'protein powder', 'protein blend', 'isolate', 'concentrate',
            'mass gainer', 'weight gainer', 'amino acid', 'bcaa', 'creatine', 
            'pre-workout', 'post-workout', 'pre workout', 'post workout', 'pump',
            'glutamine', 'arginine', 'citrulline', 'beta-alanine'
        ]
        
        if (any(keyword in full_text for keyword in protein_keywords) or
            any(ingredient in ingredients_lower for ingredient in protein_ingredients)):
            return 'Protein'
        
        # Enhanced nutraceutical detection using ingredients
        if (any(keyword in full_text for keyword in supplement_keywords) or
            any(ingredient in ingredients_lower for ingredient in nutraceutical_ingredients)):
            # Make sure it's not apparel with supplement words in description
            if not any(keyword in name_lower for keyword in apparel_keywords):
                return 'Nutraceuticals'
        
        # Check if it's clearly a physical product (accessories)
        physical_product_keywords = ['bottle', 'shaker', 'bag', 'towel', 'gloves', 'equipment']
        if any(keyword in name_lower for keyword in physical_product_keywords):
            return 'Apparel'
        
        # Default to Nutraceuticals for unclear cases
        return 'Nutraceuticals'

    def scrape_all_products(self):
        """Scrape all products from collections/all"""
        collection_path = '/collections/all'
        
        print(f"📦 Scraping products from {collection_path}...")
        
        try:
            product_urls = self.scrape_collection_pages(collection_path)
            
            # Scrape each product
            for i, url in enumerate(product_urls, 1):
                print(f"Processing product {i}/{len(product_urls)}: {url.split('/')[-1]}")
                self.scrape_and_categorize_product(url)
                time.sleep(0.5)  # Rate limiting
            
            print(f"✅ Processed {len(product_urls)} products from {collection_path}")
            
        except Exception as e:
            print(f"❌ Error processing collection {collection_path}: {str(e)}")

    def scrape_collection_pages(self, collection_path):
        """Scrape all pages of a collection and return product URLs"""
        product_urls = set()
        page = 1
        consecutive_empty_pages = 0
        max_empty_pages = 3
        
        while consecutive_empty_pages < max_empty_pages:
            url = f"{self.base_url}{collection_path}?page={page}"
            print(f"  Scraping page {page}...")
            
            try:
                response = self.session.get(url, timeout=15)
                if response.status_code != 200:
                    consecutive_empty_pages += 1
                    if consecutive_empty_pages >= max_empty_pages:
                        break
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find product links
                product_links = soup.select('a[href*="/products/"]')
                
                if not product_links:
                    consecutive_empty_pages += 1
                else:
                    consecutive_empty_pages = 0
                
                # Extract product URLs
                page_urls = set()
                for link in product_links:
                    href = link.get('href')
                    if href and '/products/' in href:
                        clean_href = href.split('?')[0].split('#')[0]
                        full_url = urljoin(self.base_url, clean_href)
                        page_urls.add(full_url)
                
                if page_urls:
                    product_urls.update(page_urls)
                    print(f"    Found {len(page_urls)} products on page {page}")
                
                page += 1
                time.sleep(0.5)
                
                # Safety check
                if page > 50:
                    break
                    
            except Exception as e:
                print(f"    Error on page {page}: {str(e)}")
                consecutive_empty_pages += 1
                time.sleep(1)
        
        return product_urls

    def scrape_and_categorize_product(self, url):
        """Scrape product details and categorize appropriately"""
        try:
            response = self.session.get(url, timeout=15)
            
            if response.status_code != 200:
                return
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract product data
            product_name = self.extract_name(soup)
            product_description = self.extract_description(soup)
            ingredients = self.extract_ingredients(soup)
            
            # Enhanced categorization using ingredients
            category = self.categorize_product(product_name, product_description, ingredients)
            
            product_info = {
                'Sr No': len(self.category_data[category]) + 1,
                'Product Type': self.determine_detailed_product_type(soup, category, product_name, ingredients),
                'Name of Product': product_name,
                'Price(INR)': self.extract_price(soup),
                'Gms': self.extract_weight(soup),
                'Ingredients': ingredients
            }
            
            # Add to appropriate category
            self.category_data[category].append(product_info)
            print(f"    ✅ {category}: {product_name}")
            
            # Debug: Print ingredients status
            if ingredients and ingredients != 'N/A':
                print(f"    📝 Ingredients found: {ingredients[:50]}...")
            else:
                print(f"    ⚠️  No ingredients found")
            
        except Exception as e:
            print(f"    ❌ Error scraping {url}: {str(e)}")

    def extract_description(self, soup):
        """Extract product description with multiple fallbacks"""
        selectors = [
            '.product__description',
            '.product-single__description',
            '.product-description',
            '.description',
            '[class*="description"]',
            '.rte',
            '.product-content',
            '.product-details'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                desc = element.get_text(strip=True)
                if desc and len(desc) > 10:
                    return desc[:500]  # Increased to 500 chars for better context
        return ""

    def determine_detailed_product_type(self, soup, category, product_name, ingredients=""):
        """Enhanced product type determination using ingredients"""
        name_lower = product_name.lower()
        ingredients_lower = ingredients.lower() if ingredients else ""
        
        if category == 'Protein':
            # Use ingredients to determine protein type more accurately
            if any(keyword in name_lower or keyword in ingredients_lower for keyword in ['whey isolate', 'isolate']):
                return 'Whey Isolate'
            elif any(keyword in name_lower or keyword in ingredients_lower for keyword in ['whey protein', 'whey concentrate', 'whey']):
                return 'Whey Protein'
            elif any(keyword in name_lower or keyword in ingredients_lower for keyword in ['casein protein', 'casein']):
                return 'Casein Protein'
            elif any(keyword in name_lower or keyword in ingredients_lower for keyword in ['soy protein', 'pea protein', 'hemp protein', 'plant protein', 'vegan']):
                return 'Plant Protein'
            elif 'mass' in name_lower or 'gainer' in name_lower:
                return 'Mass Gainer'
            elif any(keyword in name_lower for keyword in ['amino', 'bcaa', 'glutamine', 'arginine']):
                return 'Amino Acids'
            elif 'creatine' in name_lower or 'creatine' in ingredients_lower:
                return 'Creatine'
            elif 'pre-workout' in name_lower or 'pre workout' in name_lower:
                return 'Pre-Workout'
            elif 'post-workout' in name_lower or 'post workout' in name_lower:
                return 'Post-Workout'
            else:
                return 'Protein Supplement'
        
        elif category == 'Bars':
            if 'protein' in name_lower or 'protein' in ingredients_lower:
                return 'Protein Bar'
            elif 'energy' in name_lower:
                return 'Energy Bar'
            else:
                return 'Nutrition Bar'
        
        elif category == 'Apparel':
            if any(word in name_lower for word in ['shirt', 't-shirt', 'tank', 'hoodie', 'sweater', 'sweatshirt', 'jersey', 'tee']):
                return 'Tops'
            elif any(word in name_lower for word in ['shorts', 'pants', 'leggings', 'joggers']):
                return 'Bottoms'
            elif any(word in name_lower for word in ['cap', 'hat', 'beanie']):
                return 'Headwear'
            elif any(word in name_lower for word in ['bag', 'backpack', 'bottle', 'shaker', 'towel', 'gloves']):
                return 'Accessories'
            else:
                return 'Apparel'
        
        elif category == 'Nutraceuticals':
            # Enhanced nutraceutical categorization using ingredients
            if any(keyword in name_lower or keyword in ingredients_lower for keyword in ['vitamin', 'multi']):
                return 'Vitamin/Mineral'
            elif any(keyword in name_lower or keyword in ingredients_lower for keyword in ['collagen', 'hyaluronic']):
                return 'Collagen/Beauty'
            elif 'joint' in name_lower or any(keyword in ingredients_lower for keyword in ['glucosamine', 'chondroitin']):
                return 'Joint Support'
            elif any(keyword in name_lower or keyword in ingredients_lower for keyword in ['omega', 'fish oil', 'dha', 'epa']):
                return 'Omega/Fish Oil'
            elif any(keyword in name_lower or keyword in ingredients_lower for keyword in ['probiotic', 'lactobacillus', 'bifidobacterium']):
                return 'Probiotic'
            elif 'fat burn' in name_lower or 'weight loss' in name_lower:
                return 'Fat Burner'
            elif any(keyword in ingredients_lower for keyword in ['green tea', 'extract', 'antioxidant']):
                return 'Herbal/Extract'
            else:
                return 'General Supplement'
        
        return 'Supplement'

    def extract_name(self, soup):
        """Extract product name with enhanced selectors"""
        selectors = [
            'h1.product__title',
            'h1[class*="product"]',
            '.product__title',
            '.product-title',
            '.product-name',
            'h1.title',
            'h1',
            '.page-title'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                name = element.get_text(strip=True)
                if name and len(name) > 1:
                    return name
        return 'N/A'

    def extract_price(self, soup):
        """Extract and convert price to INR"""
        price_selectors = [
            '.price .money',
            '.price-item .money',
            '.product__price .money',
            '.price',
            '.money',
            '[class*="price"] .money',
            '.price-item',
            '.current-price',
            '.product-price .money',
            '.sale-price .money'
        ]
        
        for selector in price_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                
                # Extract numeric value and currency
                price_match = re.search(r'[€$₹£]?\s*(\d+(?:[.,]\d+)?)', text)
                
                if price_match:
                    price_str = price_match.group(1).replace(',', '.')
                    try:
                        price = float(price_str)
                        
                        # Convert to INR based on currency symbol
                        if '€' in text:
                            return f"₹{int(price * 90)}"
                        elif '$' in text:
                            return f"₹{int(price * 83)}"
                        elif '£' in text:
                            return f"₹{int(price * 100)}"
                        elif '₹' in text:
                            return f"₹{int(price)}"
                        else:
                            # Assume EUR if no currency symbol
                            return f"₹{int(price * 90)}"
                    except ValueError:
                        continue
        
        return 'N/A'

    def extract_weight(self, soup):
        """Extract product weight/size with enhanced patterns"""
        elements_to_check = []
        
        # Add title
        title = soup.select_one('h1')
        if title:
            elements_to_check.append(title)
        
        # Add description
        desc = soup.select_one('.product__description, .product-single__description, .description')
        if desc:
            elements_to_check.append(desc)
        
        # Add variant selectors and product info
        variants = soup.select('select option, .variant-option, [class*="size"], .product-info, .product-meta')
        elements_to_check.extend(variants)
        
        for element in elements_to_check:
            text = element.get_text(strip=True)
            
            # Enhanced weight patterns
            patterns = [
                r'(\d+(?:[.,]\d+)?)\s*kg\b',           # kg
                r'(\d+(?:[.,]\d+)?)\s*g\b',            # g
                r'(\d+(?:[.,]\d+)?)\s*lbs?\b',         # lbs
                r'(\d+(?:[.,]\d+)?)\s*oz\b',           # oz
                r'(\d+)\s*x\s*(\d+(?:[.,]\d+)?)\s*g', # format like "30 x 25g"
                r'(\d+)\s*x\s*(\d+(?:[.,]\d+)?)\s*ml', # format like "12 x 330ml"
                r'(\d+(?:[.,]\d+)?)\s*ml\b',           # ml
                r'(\d+(?:[.,]\d+)?)\s*l\b',            # liters
                r'(\d+)\s*capsules?\b',                # capsules
                r'(\d+)\s*tablets?\b',                 # tablets
                r'(\d+)\s*servings?\b',                # servings
                r'(\d+)\s*pcs?\b',                     # pieces
                r'(\d+)\s*pack\b'                      # pack
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        if isinstance(match, tuple) and len(match) == 2:
                            # Handle "30 x 25g" format
                            count = float(match[0])
                            weight = float(match[1].replace(',', '.'))
                            
                            if 'ml' in text.lower():
                                total_volume = count * weight
                                return f"{int(total_volume)}ml"
                            else:
                                total_weight = count * weight
                                return f"{int(total_weight)}g"
                        else:
                            weight_str = match.replace(',', '.')
                            weight = float(weight_str)
                            
                            # Convert to appropriate units
                            if 'kg' in text.lower():
                                return f"{int(weight * 1000)}g"
                            elif 'lbs' in text.lower():
                                return f"{int(weight * 453.592)}g"
                            elif 'oz' in text.lower():
                                return f"{int(weight * 28.3495)}g"
                            elif 'ml' in text.lower():
                                return f"{int(weight)}ml"
                            elif 'l' in text.lower():
                                return f"{int(weight * 1000)}ml"
                            elif any(unit in text.lower() for unit in ['capsule', 'tablet', 'serving', 'pcs', 'pack']):
                                return f"{int(weight)} units"
                            elif 'g' in text.lower():
                                return f"{int(weight)}g"
                            
                            if weight > 0:
                                return f"{int(weight)}g"
                    except (ValueError, IndexError):
                        continue
        
        return 'N/A'

    def extract_ingredients(self, soup):
        """Enhanced ingredient extraction with multiple strategies"""
        
        # Strategy 1: Look for dedicated ingredients sections with more selectors
        dedicated_selectors = [
            '#ingredients', 
            '.ingredients',
            '.product-ingredients',
            '.ingredients-content',
            '.tab-ingredients',
            '.product-info-ingredients',
            '.ingredient-list',
            '.ingredients-section',
            '[data-tab="ingredients"]',
            '.product-composition',
            '.nutritional-info .ingredients'
        ]
        
        for selector in dedicated_selectors:
            element = soup.select_one(selector)
            if element:
                ingredients = element.get_text(separator=' ', strip=True)
                if len(ingredients) > 10:
                    cleaned = self.clean_ingredients_text(ingredients)
                    if cleaned and cleaned != 'N/A':
                        return cleaned
        
        # Strategy 2: Look for ingredients in tabs or expandable sections
        tab_selectors = [
            '.tab-content',
            '.product-tabs',
            '.accordion-content',
            '.collapsible-content'
        ]
        
        for selector in tab_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(separator=' ', strip=True)
                if 'ingredients' in text.lower():
                    ingredients_match = re.search(
                        r'ingredients?[:\s-]+(.*?)(?:\n\n|allergen|nutrition|directions|supplement|$)',
                        text, 
                        re.IGNORECASE | re.DOTALL
                    )
                    if ingredients_match:
                        cleaned = self.clean_ingredients_text(ingredients_match.group(1))
                        if cleaned and cleaned != 'N/A':
                            return cleaned
        
        # Strategy 3: Look for ingredients in description with enhanced patterns
        description = self.extract_description(soup)
        if description:
            # Multiple ingredient patterns
            ingredient_patterns = [
                r'ingredients?[:\s-]+(.*?)(?:\n\n|allergen|nutrition|directions|supplement|contains|warning|note|$)',
                r'composition[:\s-]+(.*?)(?:\n\n|allergen|nutrition|directions|supplement|$)',
                r'formula[:\s-]+(.*?)(?:\n\n|allergen|nutrition|directions|supplement|$)',
                r'active ingredients?[:\s-]+(.*?)(?:\n\n|allergen|nutrition|directions|supplement|$)'
            ]
            
            for pattern in ingredient_patterns:
                ingredients_match = re.search(pattern, description, re.IGNORECASE | re.DOTALL)
                if ingredients_match:
                    cleaned = self.clean_ingredients_text(ingredients_match.group(1))
                    if cleaned and cleaned != 'N/A':
                        return cleaned
        
        # Strategy 4: Search entire page content
        body_text = soup.get_text(separator=' ', strip=True)
        
        # Look for ingredient sections in multiple languages
        multilingual_patterns = [
            r'(?:ingredients?|összetevők|inhaltsstoffe|ingrédients|ingredienti)[:\s-]+(.*?)(?:\n\n|allergen|nutrition|directions|supplement|warning|note|$)',
            r'(?:composition|zusammensetzung|composizione)[:\s-]+(.*?)(?:\n\n|allergen|nutrition|directions|supplement|$)'
        ]
        
        for pattern in multilingual_patterns:
            ingredients_match = re.search(pattern, body_text, re.IGNORECASE | re.DOTALL)
            if ingredients_match:
                cleaned = self.clean_ingredients_text(ingredients_match.group(1))
                if cleaned and cleaned != 'N/A':
                    return cleaned
        
        # Strategy 5: Look for structured data or JSON-LD
        json_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_scripts:
            try:
                import json
                data = json.loads(script.string)
                if isinstance(data, dict) and 'ingredients' in str(data).lower():
                    # This would need more specific parsing based on the actual JSON structure
                    pass
            except:
                pass
        
        return 'N/A'

    def clean_ingredients_text(self, raw_text):
        """Enhanced ingredient text cleaning"""
        if not raw_text:
            return 'N/A'
        
        # Basic cleanup
        text = raw_text.strip()
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        
        # Remove common non-ingredient content
        text = re.sub(r'(?i)\b(?:allergen|warning|note:|contains|free from|may contain|produced in|manufactured in|nutrition|directions|supplement facts|daily value|serving size|servings per container).*', '', text)
        
        # Remove leading numbers/bullets
        text = re.sub(r'^\s*[\d.•-]+\s*', '', text)
        
        # Remove trailing punctuation
        text = re.sub(r'[.,;:]+$', '', text)
        
        # Remove empty parentheses and brackets
        text = re.sub(r'\(\s*\)|\[\s*\]', '', text)
        
        # Clean up multiple punctuation
        text = re.sub(r'[,;]{2,}', ',', text)
        
        # Remove very short or clearly non-ingredient text
        if len(text.strip()) < 5:
            return 'N/A'
        
        # Check if it's actually ingredients (should contain typical ingredient indicators)
        ingredient_indicators = [
            ',', ';', '(', ')', 'protein', 'acid', 'extract', 'powder', 'concentrate',
            'isolate', 'vitamin', 'mineral', 'natural', 'artificial', 'flavor',
            'sweetener', 'preservative', 'stabilizer', 'emulsifier'
        ]
        
        if not any(indicator in text.lower() for indicator in ingredient_indicators):
            return 'N/A'
        
        # Trim to reasonable length
        if len(text) > 400:
            text = text[:400] + '...'
        
        return text.strip()

    def save_to_excel(self, filename='biotechusa_enhanced_catalogue.xlsx'):
        """Save all categorized data to Excel with ingredient information"""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                total_products = 0
                ingredients_found = 0
                
                for category, data in self.category_data.items():
                    if data:
                        # Re-number Sr No for each category
                        for i, product in enumerate(data, 1):
                            product['Sr No'] = i
                            # Count products with ingredients
                            if product.get('Ingredients') and product['Ingredients'] != 'N/A':
                                ingredients_found += 1
                        
                        df = pd.DataFrame(data)
                        df.to_excel(writer, sheet_name=category, index=False)
                        total_products += len(data)
                        print(f"  📊 Sheet '{category}': {len(data)} products")
                    else:
                        print(f"  ⚠️  No products found for: {category}")
                
                print(f"  💾 Total products in Excel: {total_products}")
                print(f"  📝 Products with ingredients: {ingredients_found}")
                print(f"  📁 File saved as: {filename}")
                
            return True
            
        except Exception as e:
            print(f"  ❌ Error saving Excel: {str(e)}")
            return False

    def run_scraper(self):
        """Run the enhanced scraper"""
        print("🚀 Starting Enhanced BioTechUSA Scraper")
        print("=" * 50)
        print("📋 Target Structure:")
        print("  🥤 Protein Sheet")
        print("  💊 Nutraceuticals Sheet")
        print("  🍫 Bars Sheet")
        print("  👕 Apparel Sheet")
        print("🔍 Enhanced Features:")
        print("  📝 Improved ingredient extraction")
        print("  🎯 Ingredient-based categorization")
        print("  🔄 Multiple extraction strategies")
        print("=" * 50)
        
        start_time = time.time()
        
        # Scrape all products from collections/all
        self.scrape_all_products()
        
        # Calculate totals
        total_products = sum(len(data) for data in self.category_data.values())
        
        print(f"\n" + "=" * 50)
        print(f"🎯 SCRAPING SUMMARY")
        print(f"=" * 50)
        print(f"⏱️  Time taken: {int(time.time() - start_time)} seconds")
        print(f"📦 Total products found: {total_products}")
        print(f"📊 Breakdown:")
        for category, data in self.category_data.items():
            print(f"   • {category}: {len(data)} products")
        
        # Save to Excel
        if total_products > 0:
            print(f"\n💾 Saving catalogue to Excel...")
            success = self.save_to_excel()
            
            if success:
                print(f"\n🎉 SUCCESS! Enhanced BioTechUSA catalogue saved!")
                print(f"📁 Check file: biotechusa_enhanced_catalogue.xlsx")
                print(f"📝 Ingredient extraction enabled for all products")
            else:
                print(f"\n❌ Failed to save Excel file")
        else:
            print(f"\n❌ No products found - please check website accessibility")

# Run the scraper
if __name__ == "__main__":
    scraper = BioTechUSAEnhancedScraper()
    scraper.run_scraper()