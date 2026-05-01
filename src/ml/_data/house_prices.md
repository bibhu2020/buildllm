# House Price Prediction Dataset (Dirty Version)

This dataset contains synthetic real estate records for a suburban area, intentionally injected with noise and inconsistencies to provide a challenging EDA and preprocessing exercise.

## Dataset Summary
- **Records**: 6,150 (including 150 duplicates)
- **Features**: 31
- **Target**: `Price`

## Column Descriptions & Data Quality Issues

| Column Name | Description | Data Quality Issues / Complexity |
| :--- | :--- | :--- |
| **House_ID** | Unique identifier (UUID). | No issues. |
| **Date_Sold** | Date the property was sold. | Mixed formats: ISO (2022-01-01), Unix timestamps, MM/DD/YYYY, and DD-MM-YYYY. |
| **Year_Built** | Year the house was constructed. | Inconsistent notation: e.g., "1995" vs "'95". |
| **Bedrooms** | Number of bedrooms. | Mixed types: Mostly integers, but some strings like "3+1" (indicating a den). |
| **Bathrooms** | Number of bathrooms. | Floats (e.g., 2.5) representing half-baths. |
| **Living_Area** | Internal living space in sqft. | Includes extreme outliers (10x normal size) to test robust scaling. |
| **Lot_Size** | Size of the property lot. | Unit inconsistency: Mix of raw sqft strings and strings like "0.45 ac" (acres). |
| **Neighborhood** | Categorical name of the area. | Typos and casing inconsistencies (e.g., "North_Heights" vs "northheights"). |
| **House_Style** | Architectural style of the house. | Categorical noise. |
| **Overall_Condition**| Rating of the property (1-10). | Mixed scales: Integers 1-10 mixed with strings ("Good", "Excellent"). |
| **Roof_Material** | Material used for the roof. | Categorical. |
| **Exterior_Finish** | Primary exterior material. | Categorical. |
| **Basement_Type** | Type of basement. | ~15% missing values (NaN). |
| **Heating_System** | Type of heating system. | Categorical. |
| **Cooling_System** | Type of cooling system. | Categorical. |
| **Garage_Capacity** | Number of cars garage fits. | Integers. |
| **Garage_Type** | Type of garage. | Conditional missingness (linked to Capacity). |
| **Pool** | Presence of a swimming pool. | Inconsistent labels: "Yes", "Y", "1", "Pool" vs "No", "N", "0", "None". |
| **Fireplaces** | Number of fireplaces. | Integers. |
| **Kitchen_Quality** | Quality of kitchen finish. | Coded strings: Ex, Gd, TA, Fa, Po. |
| **Utilities** | Available utility services. | Categorical labels (e.g., ELO = Electricity Only). |
| **Sale_Type** | Type of sale transaction. | Categorical codes (WD, New, COD). |
| **Sale_Condition** | Condition of sale (e.g., Normal). | Categorical. |
| **Zoning_Class** | Land zoning classification. | Categorical codes (RL, RM, FV, RH). |
| **School_Rating** | Rating of nearest schools (1-10).| Numeric. |
| **Dist_to_City** | Distance to city center (km). | Numeric float. |
| **Crime_Rate** | Local crime rate index. | Numeric float. |
| **Tax_Annual** | Annual property tax amount. | Contains -99 error codes for missing data. |
| **Renovation_Year** | Year of last major renovation. | >60% missing values; requires intelligent imputation. |
| **Property_Type** | Type of property (Condo, SF). | Categorical. |
| **Price** | **Target Variable** (Sale Price). | Contains missing values (NaN) and massive outliers (10x price). |

## Recommended EDA Steps
1. **Deduplication**: Remove the 150 exact duplicate records.
2. **Date Parsing**: Standardize `Date_Sold` into a proper datetime object.
3. **Lot Size Normalization**: Convert all `Lot_Size` values to a single unit (sqft).
4. **Categorical Cleaning**: Merge identical neighborhood names and pool indicators.
5. **Outlier Removal**: Identify and handle the 10x spikes in `Living_Area` and `Price`.
6. **Imputation**: Determine a strategy for `Renovation_Year` (e.g., 0 for no renovation) and `Basement_Type`.
