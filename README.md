# Irish-Housing-Price-Prediction-with-Machine-Learning
This project predicts **housing prices in Ireland** using real property listings data.   The goal is to build and evaluate machine learning models that can estimate house prices and identify the most important factors driving property values.


## 📂 Dataset

Data source:
- https://github.com/AnthonyBloomer/daftlistings
- https://www.kaggle.com/datasets/eavannan/daftie-house-price-data


Data Timeline - 01/12/2021 to 30/01/2022



The dataset includes information such as:
- **Price, Bedrooms, Bathrooms, Size**
- **Property Type** (Apartment, Semi-D, Detached, Bungalow, etc.)
- **Energy Rating (BER)**
- **Listing Metadata** (images, video, brochure availability)
- **Location (latitude, longitude)**
- **Seller Info** (agent, private, branded/unbranded)

---

## 🔧 Steps in the Project
1. **Data Cleaning** → removed duplicates, handled missing values  
2. **Feature Engineering** → separated price vs AMV, log-transform for skewed prices  
3. **EDA** → distribution of prices, price vs property type, BER rating impact  
4. **Train-Test Split**  
5. **Preprocessing Pipeline** → scaling numeric, one-hot encoding categorical, encoding binary  
6. **Baseline Model** → predict median price  
7. **Linear Regression (log target)**  
8. **Random Forest Regressor**  
9. **Histogram Gradient Boosting (HGB) with log target**  
10. **Model Comparison** (MAE, RMSE, R²)  
11. **Hyperparameter Tuning** for Random Forest  
12. **Permutation Feature Importance** (on best model)  
13. **Feature Name Cleaning** for readability  
14. **Top 20 Features Barplot**  
15. **Summary of Results**  



## 📊 Model Performance
<img width="405" height="131" alt="ML Result" src="https://github.com/user-attachments/assets/02160794-c2b8-4b42-a365-97426d0bc96e" />

  



| Model                        | MAE (€)     | RMSE (€)     | R²   |
|-------------------------------|-------------|-------------|------|
| Baseline (median predictor)   | 149,810     | 257,891     | -0.07 |
| Linear Regression (log)       | 105,348     | 219,033     | 0.23 |
| **Random Forest**             | **97,073**  | **156,917** | **0.60** |
| HGB Regressor (log target)    | 96,825      | 161,303     | 0.58 |

✅ The **Random Forest model** performed best overall, with an R² of ~0.60, meaning it explains about **60% of the variation** in Irish house prices.

---

## 📌 Key Feature Importances
Top drivers of house prices (from permutation importance):
1. `sellerType_UNBRANDED_AGENT` – Suggests that unbranded/independent agents list properties with very different pricing patterns compared to branded agents or private sellers.  
2. `propertyType_Site` – raw land/sites have completely different pricing dynamics than built properties.  
3. `listing_age_days` – The time a property has been listed strongly impacts price prediction.  
4. `latitude` – Location effect: properties further north/south are priced differently.  
5. `dist_to_cork` – Distance to Cork city has a measurable impact.  

Metadata features (e.g., whether the listing had a brochure or video) had **minimal impact**.

---

## 📷 Visualisations

- Price distribution (log-transformed vs raw)
    
    <img width="540" height="391" alt="Price distribution log" src="https://github.com/user-attachments/assets/fe7a7c7a-9598-4736-9cb5-05f174bdaa37" />
    <img width="540" height="391" alt="Price distribution raw" src="https://github.com/user-attachments/assets/ba5fa8cf-9197-4a3f-99b8-7533fb50301f" />

- Boxplot: price by property type
    
    <img width="984" height="584" alt="Distribution of Property Prices by Property Type" src="https://github.com/user-attachments/assets/eba65a41-741d-4e69-8fb9-38d716d1ce42" />

- Scatterplot: property size vs price
    
    <img width="981" height="584" alt="Property Size vs Price" src="https://github.com/user-attachments/assets/82c2c84c-b642-4aab-ba2a-d361cbc2d983" />

- Feature importance barplot (Top 20 features)
    
    <img width="790" height="989" alt="Top 20 Most Important Features (Random Forest)" src="https://github.com/user-attachments/assets/37b66d78-053f-4d5c-ae93-3a9987d55532" />

  

---
---

