import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
data = pd.read_csv('rental_house_1.csv')

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Basic statistics
print(data.describe())

# Visualization: Rent Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Rent'], bins=20, kde=True)
plt.title('Rent Distribution')
plt.xlabel('Rent ($)')
plt.ylabel('Frequency')
plt.show()

# Visualization: Average Rent by Property Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='PropertyType', y='Rent', data=data)
plt.title('Average Rent by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Rent ($)')
plt.show()

# Visualization: Rent over Time
plt.figure(figsize=(10, 6))
data.set_index('Date')['Rent'].plot()
plt.title('Rent Over Time')
plt.xlabel('Date')
plt.ylabel('Rent ($)')
plt.show()

# Visualization: Number of Bedrooms vs. Rent
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Bedrooms', y='Rent', data=data, hue='PropertyType', style='PropertyType')
plt.title('Number of Bedrooms vs. Rent')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Rent ($)')
plt.show()

# Visualization: Rent Distribution by Neighborhood
plt.figure(figsize=(10, 6))
sns.boxplot(x='Neighborhood', y='Rent', data=data)
plt.title('Rent Distribution by Neighborhood')
plt.xlabel('Neighborhood')
plt.ylabel('Rent ($)')
plt.show()

# Adding a column for the number of amenities
data['AmenitiesCount'] = data['Amenities'].apply(lambda x: len(x.split(',')))

# Visualization: Amenities Count vs. Rent
plt.figure(figsize=(10, 6))
sns.scatterplot(x='AmenitiesCount', y='Rent', data=data, hue='PropertyType', style='PropertyType')
plt.title('Amenities Count vs. Rent')
plt.xlabel('Number of Amenities')
plt.ylabel('Rent ($)')
plt.show()

# Visualization: Bar Graph - Average Rent by Neighborhood
plt.figure(figsize=(10, 6))
average_rent_by_neighborhood = data.groupby('Neighborhood')['Rent'].mean()
average_rent_by_neighborhood.plot(kind='bar')
plt.title('Average Rent by Neighborhood')
plt.xlabel('Neighborhood')
plt.ylabel('Average Rent ($)')
plt.show()

# Visualization: Pie Chart - Distribution of Property Types
plt.figure(figsize=(8, 8))
property_type_distribution = data['PropertyType'].value_counts()

# Define a color palette with distinct colors
colors = sns.color_palette("Set3", n_colors=len(property_type_distribution))

property_type_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Distribution of Property Types')
plt.ylabel('')
plt.show()


# Visualization: 3D Scatter Plot - Rent, Bedrooms, and Square Footage
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(data['Bedrooms'], data['SquareFootage'], data['Rent'], c='b', marker='o')

# Set labels
ax.set_xlabel('Bedrooms')
ax.set_ylabel('Square Footage')
ax.set_zlabel('Rent ($)')
ax.set_title('3D Scatter Plot: Rent, Bedrooms, and Square Footage')

plt.show()
