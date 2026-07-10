import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import numpy as np

def analyze_datasets():
    # Setup directories
    output_dir = 'dataset_analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load datasets
    products_path = 'data/products.csv'
    outfits_path = 'data/outfits.csv'
    
    try:
        products_df = pd.read_csv(products_path)
        outfits_df = pd.read_csv(outfits_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Basic stats
    total_products = len(products_df)
    total_outfits = len(outfits_df)
    
    # Products analysis
    categories = products_df['category'].unique() if 'category' in products_df.columns else []
    num_categories = len(categories)
    
    occasions = products_df['occasion'].unique() if 'occasion' in products_df.columns else []
    # split occasions if comma separated
    all_occasions = []
    if 'occasion' in products_df.columns:
        for occ in products_df['occasion'].dropna():
            all_occasions.extend([o.strip() for o in str(occ).split(',')])
    unique_occasions = list(set(all_occasions))
    num_occasions = len(unique_occasions)
    
    genders = products_df['gender'].unique() if 'gender' in products_df.columns else []
    num_genders = len(genders)
    
    brands = products_df['brand'].unique() if 'brand' in products_df.columns else []
    num_brands = len(brands)
    
    # Category Analysis
    cat_counts = {}
    if 'category' in products_df.columns:
        cat_counts = products_df['category'].value_counts().to_dict()
    
    # Subcategory distribution (approximation based on typical values)
    topwear = products_df[products_df['category'].str.contains('top|shirt|t-shirt|jacket|sweater', case=False, na=False)].shape[0] if 'category' in products_df.columns else 0
    bottomwear = products_df[products_df['category'].str.contains('pant|jean|short|skirt|trouser', case=False, na=False)].shape[0] if 'category' in products_df.columns else 0
    footwear = products_df[products_df['category'].str.contains('shoe|sneaker|boot|sandal', case=False, na=False)].shape[0] if 'category' in products_df.columns else 0
    accessories = products_df[products_df['category'].str.contains('watch|belt|hat|cap|bag|sunglasses', case=False, na=False)].shape[0] if 'category' in products_df.columns else 0

    # Occasion Analysis
    occ_counts = pd.Series(all_occasions).value_counts().to_dict()
    
    # Gender Analysis
    men_count = products_df[products_df['gender'].str.contains('men|male', case=False, na=False)].shape[0] if 'gender' in products_df.columns else 0
    women_count = products_df[products_df['gender'].str.contains('women|female', case=False, na=False)].shape[0] if 'gender' in products_df.columns else 0
    unisex_count = products_df[products_df['gender'].str.contains('unisex|both', case=False, na=False)].shape[0] if 'gender' in products_df.columns else 0
    
    # Price Analysis
    min_price, max_price, avg_price, median_price = 0, 0, 0, 0
    if 'price_inr' in products_df.columns:
        prices = pd.to_numeric(products_df['price_inr'].replace(r'[\$,]', '', regex=True), errors='coerce').dropna()
        if not prices.empty:
            min_price = prices.min()
            max_price = prices.max()
            avg_price = prices.mean()
            median_price = prices.median()

    # Data Quality Analysis
    missing_values = products_df.isnull().sum().to_dict()
    duplicate_products = products_df.duplicated(subset=['id'] if 'id' in products_df.columns else None).sum()
    missing_images = products_df['image'].isnull().sum() if 'image' in products_df.columns else 0
    missing_metadata = products_df[['category', 'price_inr', 'gender']].isnull().sum().sum() if set(['category', 'price_inr', 'gender']).issubset(products_df.columns) else 0
    
    # Generate Visualizations
    sns.set_theme(style="whitegrid")
    
    # 1. Category distribution chart
    if 'category' in products_df.columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(data=products_df, y='category', order=products_df['category'].value_counts().index)
        plt.title('Category Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'category_distribution.png'))
        plt.close()

    # 2. Occasion distribution chart
    if occ_counts:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(occ_counts.values())[:10], y=list(occ_counts.keys())[:10])
        plt.title('Top 10 Occasions')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'occasion_distribution.png'))
        plt.close()

    # 3. Gender distribution chart
    if 'gender' in products_df.columns:
        plt.figure(figsize=(8, 8))
        products_df['gender'].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title('Gender Distribution')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gender_distribution.png'))
        plt.close()

    # 4. Price distribution histogram
    if 'price_inr' in products_df.columns and not prices.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(prices, bins=30, kde=True)
        plt.title('Price Distribution')
        plt.xlabel('Price (INR)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'price_distribution.png'))
        plt.close()

    # Generate JSON Summary
    summary_data = {
        "dataset_overview": {
            "total_products": total_products,
            "total_outfits": total_outfits,
            "unique_categories": num_categories,
            "unique_occasions": num_occasions,
            "unique_genders": num_genders,
            "unique_brands": num_brands
        },
        "category_analysis": {
            "top_categories": {k: v for k, v in list(cat_counts.items())[:5]},
            "distribution": {
                "topwear": topwear,
                "bottomwear": bottomwear,
                "footwear": footwear,
                "accessories": accessories
            }
        },
        "occasion_analysis": {
            "top_occasions": {k: v for k, v in list(occ_counts.items())[:5]},
            "total_unique": len(unique_occasions)
        },
        "gender_analysis": {
            "men": men_count,
            "women": women_count,
            "unisex": unisex_count
        },
        "price_analysis": {
            "min": float(min_price),
            "max": float(max_price),
            "average": float(avg_price),
            "median": float(median_price)
        },
        "data_quality": {
            "missing_values_by_column": missing_values,
            "duplicates": int(duplicate_products),
            "missing_images": int(missing_images),
            "missing_metadata": int(missing_metadata)
        }
    }

    with open(os.path.join(output_dir, 'dataset_summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=4)

    # Generate Markdown Report
    md_content = f"""# Dataset Analysis Report

## Dataset Overview
* Total number of products: {total_products}
* Total number of outfits: {total_outfits}
* Number of unique categories: {num_categories}
* Number of unique occasions: {num_occasions}
* Number of unique genders: {num_genders}
* Number of unique brands: {num_brands}

## Category Analysis
* Top categories by count:
"""
    for k, v in list(cat_counts.items())[:5]:
        md_content += f"  * {k}: {v}\n"
    
    md_content += f"""
* Distribution of major types:
  * Topwear: {topwear}
  * Bottomwear: {bottomwear}
  * Footwear: {footwear}
  * Accessories: {accessories}

![Category Distribution](category_distribution.png)

## Occasion Analysis
* Count per occasion (Top 10):
"""
    for k, v in list(occ_counts.items())[:10]:
        md_content += f"  * {k}: {v}\n"

    md_content += """
![Occasion Distribution](occasion_distribution.png)

## Gender Analysis
* Men products count: {men_count}
* Women products count: {women_count}
* Unisex products count: {unisex_count}

![Gender Distribution](gender_distribution.png)

## Price Analysis
* Minimum price: ${min_price:.2f}
* Maximum price: ${max_price:.2f}
* Average price: ${avg_price:.2f}
* Median price: ${median_price:.2f}

![Price Distribution](price_distribution.png)

## Data Quality Analysis
* Missing values:
"""
    for k, v in missing_values.items():
        if v > 0:
            md_content += f"  * {k}: {v}\n"
            
    md_content += f"""
* Duplicate products: {duplicate_products}
* Missing images: {missing_images}
* Missing metadata fields: {missing_metadata}
"""

    with open(os.path.join(output_dir, 'DATASET_ANALYSIS.md'), 'w') as f:
        f.write(md_content)

    print(f"Analysis complete. Results saved in '{output_dir}' directory.")

if __name__ == '__main__':
    analyze_datasets()
