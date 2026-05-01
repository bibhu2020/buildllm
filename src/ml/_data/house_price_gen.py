import pandas as pd
import numpy as np
import random
import uuid
from datetime import datetime, timedelta

def generate_dirty_house_data(n=6000):
    neighborhoods = ["North_Heights", "South_Side", "Downtown_Lofts", "Green_Valley", "West_End", "East_Gate", "Suburban_Oasis"]
    styles = ["Ranch", "Colonial", "Modern", "Victorian", "Bungalow", "Split-Level"]
    materials = ["Brick", "Vinyl_Siding", "Wood", "Stucco", "Stone"]
    utilities = ["All_Pub", "No_Sewr", "No_Seer", "ELO"] # ELO = Electricity only
    
    data = []
    
    for i in range(n):
        # 1. House_ID
        house_id = str(uuid.uuid4())
        
        # 2. Date_Sold (Mixed formats)
        base_date = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1500))
        date_fmt = random.choice(['iso', 'unix', 'mdy', 'dmy'])
        if date_fmt == 'iso':
            date_sold = base_date.isoformat()
        elif date_fmt == 'unix':
            date_sold = str(int(base_date.timestamp()))
        elif date_fmt == 'mdy':
            date_sold = base_date.strftime('%m/%d/%Y')
        else:
            date_sold = base_date.strftime('%d-%m-%Y')
            
        # 3. Year_Built (Noise)
        year_built_val = random.randint(1900, 2023)
        if random.random() < 0.1:
            year_built = f"'{str(year_built_val)[-2:]}"
        else:
            year_built = str(year_built_val)
            
        # 4. Bedrooms (Mixed types)
        beds = random.randint(1, 6)
        if random.random() < 0.05:
            bedrooms = f"{beds}+1"
        else:
            bedrooms = str(beds)
            
        # 5. Bathrooms
        bathrooms = round(random.uniform(1, 4) * 2) / 2 # 1, 1.5, 2, etc.
        
        # 6. Living_Area (Outliers)
        living_area = random.randint(800, 5000)
        if random.random() < 0.02:
            living_area *= 10 # Extreme outlier
            
        # 7. Lot_Size (Mixed units: sqft vs acres)
        lot_val = random.randint(2000, 40000)
        if random.random() < 0.1:
            lot_size = f"{round(lot_val/43560, 2)} ac"
        else:
            lot_size = str(lot_val)
            
        # 8. Neighborhood (Typos)
        nb = random.choice(neighborhoods)
        if random.random() < 0.05:
            nb = nb.replace("_", "").lower()
            
        # 9. House_Style
        house_style = random.choice(styles)
        
        # 10. Overall_Condition (Mixed scale/strings)
        cond = random.randint(1, 10)
        if random.random() < 0.1:
            overall_condition = random.choice(["Excellent", "Good", "Fair", "Poor"])
        else:
            overall_condition = str(cond)
            
        # 11. Roof_Material
        roof = random.choice(["Shingle", "Metal", "Tile", "Slate"])
        
        # 12. Exterior_Finish
        ext = random.choice(materials)
        
        # 13. Basement_Type (Missing values)
        basement = random.choice(["Full", "Crawl", "Slab", "None"]) if random.random() > 0.15 else np.nan
        
        # 14. Heating_System
        heating = random.choice(["GasA", "GasW", "Grav", "Wall"])
        
        # 15. Cooling_System
        cooling = random.choice(["Central", "Window", "None"])
        
        # 16. Garage_Capacity
        garage_cap = random.randint(0, 4)
        
        # 17. Garage_Type
        garage_type = random.choice(["Attchd", "Detchd", "BuiltIn", "None"]) if garage_cap > 0 else "None"
        
        # 18. Pool (Noise)
        if random.random() < 0.05:
            pool = random.choice(["Yes", "Y", "1", "Pool"])
        else:
            pool = random.choice(["No", "N", "0", "None"])
            
        # 19. Fireplaces
        fireplaces = random.randint(0, 3)
        
        # 20. Kitchen_Quality
        kit_qual = random.choice(["Ex", "Gd", "TA", "Fa", "Po"]) # Excellent, Good, Typical/Average, Fair, Poor
        
        # 21. Utilities
        util = random.choice(utilities)
        
        # 22. Sale_Type
        sale_type = random.choice(["WD", "New", "COD", "Con"])
        
        # 23. Sale_Condition
        sale_cond = random.choice(["Normal", "Abnorml", "Partial", "AdjLand"])
        
        # 24. Zoning_Class
        zoning = random.choice(["RL", "RM", "FV", "RH"])
        
        # 25. School_Rating
        school = random.randint(1, 10)
        
        # 26. Distance_to_City_Center
        dist = round(random.uniform(1.0, 40.0), 1)
        
        # 27. Crime_Rate_Index
        crime = round(random.uniform(0.1, 5.0), 2)
        
        # 28. Tax_Annual
        tax = (living_area * 0.8) + (random.randint(1000, 5000))
        if random.random() < 0.01:
            tax = -99 # Missing/error code
            
        # 29. Renovation_Year (Significant missing)
        renovated = random.choice([True, False])
        if renovated and random.random() > 0.6:
            renovation_year = str(random.randint(2000, 2023))
        else:
            renovation_year = "NaN"
            
        # 30. Property_Type
        prop_type = random.choice(["Single_Family", "Townhouse", "Condo", "Duplex"])
        
        # 31. Price (Target - with complexity)
        # Base price calculation
        price = (living_area * 150) + (beds * 10000) + (bathrooms * 15000) + (random.randint(-20000, 20000))
        if nb == "Downtown_Lofts": price += 50000
        if nb == "Suburban_Oasis": price += 30000
        if pool in ["Yes", "Y", "1", "Pool"]: price += 25000
        
        if random.random() < 0.01:
            price = np.nan # Missing target
        elif random.random() < 0.01:
            price = price * 10 # Massive outlier
            
        data.append([
            house_id, date_sold, year_built, bedrooms, bathrooms, living_area, lot_size,
            nb, house_style, overall_condition, roof, ext, basement, heating, cooling,
            garage_cap, garage_type, pool, fireplaces, kit_qual, util, sale_type,
            sale_cond, zoning, school, dist, crime, tax, renovation_year, prop_type, price
        ])
        
    cols = [
        "House_ID", "Date_Sold", "Year_Built", "Bedrooms", "Bathrooms", "Living_Area", "Lot_Size",
        "Neighborhood", "House_Style", "Overall_Condition", "Roof_Material", "Exterior_Finish",
        "Basement_Type", "Heating_System", "Cooling_System", "Garage_Capacity", "Garage_Type",
        "Pool", "Fireplaces", "Kitchen_Quality", "Utilities", "Sale_Type", "Sale_Condition",
        "Zoning_Class", "School_Rating", "Distance_to_City_Center", "Crime_Rate_Index",
        "Tax_Annual", "Renovation_Year", "Property_Type", "Price"
    ]
    
    df = pd.DataFrame(data, columns=cols)
    
    # Add some duplicate records
    df_dupes = df.sample(150)
    df = pd.concat([df, df_dupes], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    df.to_csv('house_prices.csv', index=False)
    print(f"Generated {len(df)} records in house_prices.csv")

if __name__ == "__main__":
    generate_dirty_house_data(6000)
