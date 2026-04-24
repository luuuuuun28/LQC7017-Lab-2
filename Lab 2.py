import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style for the plots
sns.set_theme(style="whitegrid")

# 1. Load and Inspect Data
# Loading the World Energy dataset
df = pd.read_csv('WorldEnergy.csv')

print("--- Dataset Overview ---")
print(f"Total Records: {df.shape[0]}")
print(f"Total columns: {df.shape[1]}")
print("\nFirst 5 rows:")
print(df.head())

# 2. Data Cleaning & Missing Value Analysis
print("\n--- Missing Value Analysis (Per Column) ---")
missing_vals = df.isnull().sum()
# Showing top 10 columns with missing data
print(missing_vals.head(10))  
print("\nDuplicated rows:", df.duplicated().sum())

# Fill missing numerical values with the median
df_clean = df.copy()
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

# 3. Tabular Summaries (Basic Statistics)
print("\n--- Summary Statistics (Selected Columns) ---")
# Focus on primary metrics: Population, GDP, and Electricity Generation
# cols_of_interest = ['population', 'gdp', 'electricity_generation']
print(df_clean.describe())

# 4. Outlier Detection (IQR Method)
# Detecting outliers in 'electricity_generation'
Q1 = df_clean['electricity_generation'].quantile(0.25)
Q3 = df_clean['electricity_generation'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_clean[(df_clean['electricity_generation'] < lower_bound) |
                    (df_clean['electricity_generation'] > upper_bound)]

print("\n--- Outlier Detection (Electricity Generation) ---")
print(f"Lower Bound: {lower_bound:.2f}")
print(f"Upper Bound: {upper_bound:.2f}")
print(f"Number of Outliers detected: {len(outliers)}")

# Count anomalies by country
print ("\n--- Anomalies (By Country) ---")
print (outliers['country'].value_counts())

# 5. Univariate Analysis 
# Histogram of Electricity Generation (Log Scale)
# Filter for values > 0 because log(0) is undefined
df_positive_gen = df_clean[df_clean['electricity_generation'] > 0]

plt.figure(figsize=(10, 6))
sns.histplot(df_positive_gen['electricity_generation'], kde=True, color='royalblue', log_scale=True)
plt.title('Univariate: Global Distribution of Electricity Generation (Log Scale)', fontsize=14)
plt.xlabel('Electricity Generation (TWh) - Log Scale')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 6. Bivariate Analysis 
# Variables representing Economy, Demographics, Emissions, and Energy Sources
cols_to_correlate = [
    'gdp',                      # Economic strength
    'population',               # Demographic demand
    'greenhouse_gas_emissions',  # Environmental impact
    'electricity_generation',   # Total energy output
    'fossil_fuel_consumption',  # Traditional energy reliance
    'renewables_consumption',   # Transition to green energy
    'low_carbon_consumption'    # Nuclear + Renewables
]

# GDP vs Electricity Generation Scatter Plot
GDP_elec_correlation = df_clean['gdp'].corr(df_clean['electricity_generation'])
print(f"Correlation between GDP and Electricity Generation: {GDP_elec_correlation:.4f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_clean, x='gdp', y='electricity_generation', alpha=0.5)
plt.title('Bivariate: GDP vs Electricity Generation')
plt.xlabel('GDP ($)')
plt.ylabel('Electricity Generation (TWh)')
plt.show() 

# Population vs Emissions Plot
pop_ghg_corr = df_clean['population'].corr(df_clean['greenhouse_gas_emissions'])
print(f"Correlation between Population and GHG Emissions: {pop_ghg_corr:.4f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_clean, x='population', y='greenhouse_gas_emissions', color='red', alpha=0.5)
plt.title('Bivariate: Population vs GHG Emissions')
plt.xlabel('Population')
plt.ylabel('GHG Emissions (mtCO2e)')
plt.show()

# Fossil Fuels vs Renewables Plot
fossil_renew_corr = df_clean['fossil_fuel_consumption'].corr(df_clean['renewables_consumption'])
print(f"Correlation between Fossil Fuel and Renewable Consumption: {fossil_renew_corr:.4f}")

plt.figure(figsize=(10, 6))
sns.regplot(data=df_clean, x='fossil_fuel_consumption', y='renewables_consumption', 
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Bivariate: Fossil Fuel vs Renewable Consumption', fontsize=14)
plt.xlabel('Fossil Fuel Consumption (TWh)')
plt.ylabel('Renewable Consumption (TWh)')
plt.tight_layout()
plt.show()

# Total Generation vs. Low Carbon Energy Plot
gen_lowcarbon_corr = df_clean['electricity_generation'].corr(df_clean['low_carbon_electricity'])
print(f"Correlation between Total Generation and Low Carbon Electricity: {gen_lowcarbon_corr:.4f}")

plt.figure(figsize=(10, 6))
sns.regplot(data=df_clean, x='electricity_generation', y='low_carbon_electricity', 
            scatter_kws={'alpha':0.3, 'color':'purple'}, line_kws={'color':'orange'})
plt.title('Bivariate: Total Generation vs Low Carbon Electricity', fontsize=14)
plt.xlabel('Total Electricity Generation (TWh)')
plt.ylabel('Low Carbon Electricity (TWh)')
plt.tight_layout()
plt.show()

# 7. Multivariate Analysis
print("--- Multivariate Correlation Matrix ---")
# Calculating the Pearson correlation matrix for the selected columns
correlation_matrix = df_clean[cols_to_correlate].corr()
print(correlation_matrix)

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Multivariate: Correlation Heatmap of Energy, Economy, and Environment')
plt.tight_layout()
plt.show() 

# Graph of Global Transition Electricity Generation (1985-2022)
# Filter for 'World' aggregate data
df_world = df[df['country'] == 'World'].copy()
df_world = df_world[df_world['year'] >= 1985]

# Define columns representing the % share of electricity generation
mix_cols = ['coal_share_elec', 'gas_share_elec', 'oil_share_elec', 
            'nuclear_share_elec', 'hydro_share_elec', 'solar_share_elec', 'wind_share_elec']
df_mix = df_world[['year'] + mix_cols].fillna(0)

plt.figure(figsize=(12, 6))
plt.stackplot(df_mix['year'], 
              df_mix['coal_share_elec'], df_mix['gas_share_elec'], df_mix['oil_share_elec'],
              df_mix['nuclear_share_elec'], df_mix['hydro_share_elec'], 
              df_mix['solar_share_elec'], df_mix['wind_share_elec'],
              labels=['Coal', 'Gas', 'Oil', 'Nuclear', 'Hydro', 'Solar', 'Wind'],
              alpha=0.8, colors=['#333333', '#8e44ad', '#c0392b', '#f1c40f', '#2980b9', '#f39c12', '#27ae60'])

plt.title('Multivariate: The Global Transition of Electricity Generation Mix (%)', fontsize=14)
plt.ylabel('Share of Total Electricity (%)')
plt.xlabel('Year')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show() 

# Graph of Solar VS Wind Growth (GLOBAL TWh)
plt.figure(figsize=(10, 6))
plt.plot(df_world['year'], df_world['solar_electricity'], label='Solar Power', color='#f39c12', linewidth=2, marker='o', markersize=4)
plt.plot(df_world['year'], df_world['wind_electricity'], label='Wind Power', color='#27ae60', linewidth=2, marker='s', markersize=4)
plt.title('Multivariate: Global Growth of Solar vs Wind Electricity (TWh)', fontsize=14)
plt.ylabel('Generation (Terawatt-hours)')
plt.xlabel('Year')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show() 

# Total Generation vs Renewable Growth
# Calculate both means in one step to ensure the years match perfectly
yearly_comparison = df_clean.groupby('year')[['electricity_generation', 'renewables_consumption']].mean().tail(10)
plt.figure(figsize=(12, 6))

# Plot Total Generation
sns.lineplot(x=yearly_comparison.index, y=yearly_comparison['electricity_generation'], 
             marker='o', color='dodgerblue', label='Total Electricity Generation', linewidth=2.5)

# Plot Renewable Consumption
sns.lineplot(x=yearly_comparison.index, y=yearly_comparison['renewables_consumption'], 
             marker='s', color='forestgreen', label='Renewable Consumption', linewidth=2.5)

plt.title('Multivariate: Global Energy Transition of Total Generation vs Renewable Consumption (Last 10 Years)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Mean Energy (TWh)', fontsize=12)
plt.xticks(yearly_comparison.index)
plt.legend(frameon=True, facecolor='white')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
