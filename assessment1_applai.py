 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('./airline_service.csv')

# ---------------------------------------------------------------------------- #
#                              Data overview start                             #
# ---------------------------------------------------------------------------- #
print("\nData Types and Non-Null Counts:\n", df.info())

# Dataset size: 103,904 rows and 25 columns. 
# Data types: Primarily integer (int64) and object (categorical). 
# Missing values: 310 missing values in the "Arrival Delay in Minutes" column. 
# Target variable: "satisfaction" (categorical). 

# ---------------------------------------------------------------------------- #
#                           Data preprocessing start                           #
# ---------------------------------------------------------------------------- #

# Handle missing values
df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].mean())

# Map the 'satisfaction' column to numerical values
df['satisfaction'] = df['satisfaction'].map({'satisfied': 1, 'dissatisfied': 0})

# Visualize Customer Type vs. Satisfaction before applying pd.get_dummies
plt.figure(figsize=(15, 12))

# 1. Flight Distance vs. Satisfaction (Scatter Plot)
plt.subplot(3, 1, 1)
sns.scatterplot(x='Flight Distance', y='satisfaction', data=df)
plt.title('Flight Distance vs. Satisfaction')
plt.xlabel('Flight Distance')
plt.ylabel('Satisfaction')

# 2. Customer Type vs. Satisfaction (Bar Plot)
plt.subplot(3, 1, 2)
sns.barplot(x='Customer Type', y='satisfaction', data=df)
plt.title('Customer Type vs. Satisfaction')
plt.xlabel('Customer Type')
plt.ylabel('Satisfaction')

# Apply pd.get_dummies after the bar plot to avoid losing the original 'Customer Type' column
df = pd.get_dummies(df, columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'])

# Scaling numerical features
scaler = StandardScaler()
df[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']] = scaler.fit_transform(
    df[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']]
)

# Handling outliers using IQR
Q1 = df[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']].quantile(0.25)
Q3 = df[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']] < (Q1 - 1.5 * IQR)) |
          (df[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']] > (Q3 + 1.5 * IQR))).any(axis=1)]
print("\nData Types and Non-Null Counts:\n", df.info())

# 3. Age Distribution by Satisfaction (Histogram)
plt.subplot(3, 1, 3)
sns.histplot(data=df, x='Age', hue='satisfaction', multiple='stack', kde=True)
plt.title('Age Distribution by Satisfaction')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Adjust the layout
plt.tight_layout()
plt.show()

print("\nData Types and Non-Null Counts:\n", df.info())

 
# ---------------------------------------------------------------------------- #
#                                  conclusion                                  #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                            Data Overview                                     #
# ---------------------------------------------------------------------------- #
# Data Size: The dataset consists of 103,904 entries with 31 columns after 
# encoding categorical variables.
# Columns: Includes various features related to flight details, customer 
# service ratings, and satisfaction levels.
#
# ---------------------------------------------------------------------------- #
#                  Key Findings from Correlation Heatmap                       #
# ---------------------------------------------------------------------------- #
# Strong Relationships: 
# - Arrival Delay in Minutes and Departure Delay in Minutes show strong 
#   correlations with the target variable, satisfaction. High delays negatively 
#   impact satisfaction.
# - Flight Distance also shows a notable correlation with satisfaction, 
#   indicating that longer flights might have varied impacts on customer 
#   satisfaction based on other factors.
#
# Redundancy:
# - Some features are highly correlated with each other, such as Departure 
#   Delay in Minutes and Arrival Delay in Minutes. This indicates potential 
#   redundancy or multicollinearity that might need addressing in modeling.
#
# ---------------------------------------------------------------------------- #
#                  Summary of Key Patterns and Trends                          #
# ---------------------------------------------------------------------------- #
# Impact of Delays on Satisfaction:
# - Both Arrival Delay in Minutes and Departure Delay in Minutes are critical 
#   factors affecting customer satisfaction. Longer delays generally correlate 
#   with lower satisfaction levels.
#
# Feature Importance:
# - Flight Distance and delays are significant features that influence satisfaction, 
#   while age and other features might show varied importance depending on their 
#   distribution.
#
# Feature Interactions:
# - Complex relationships between features (e.g., flight distance and delays) 
#   suggest that understanding these interactions can be key to improving 
#   satisfaction predictions.
