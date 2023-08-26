import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample data: Food items and their features (flavor, spiciness, sweetness)
food_items = {
    "Pizza": [0.8, 0.2, 0.1],
    "Sushi": [0.5, 0.7, 0.2],
    "Burger": [0.7, 0.4, 0.1],
    "Curry": [0.9, 0.5, 0.3],
    "Ice Cream": [0.2, 0.1, 0.9]
}

# Sample user preferences
user_preferences = np.array([0.6, 0.3, 0.2])

# Convert food items to a matrix
item_matrix = np.array([features for features in food_items.values()])

# Calculate cosine similarity between user preferences and food items
similarities = cosine_similarity(item_matrix, [user_preferences])

# Rank items based on similarity
ranked_items = np.argsort(similarities[:, 0])[::-1]

# Recommend top N items to the user
top_n = 3
recommended_items =  [list(food_items.keys())[idx] for idx in ranked_items[:top_n]]

print("Recommended items:", recommended_items)