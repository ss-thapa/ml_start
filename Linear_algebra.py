import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity




### Calculate the Euclidean distance from the origin (L2 norm)

## Define an n-dimensional vector A

# A = np.array([1, 2, 3, 4, 5,6,7,8,9,10,11,112,12])

# distance = np.linalg.norm(A)

# print("Euclidean distance from the origin:", distance)





### Calculate the Euclidean distance between A and B (L2 norm of the difference)


## Define two 2-dimensional vectors A and B
# A = np.array([1, 2, 3, 4, 5])
# B = np.array([6, 7, 8, 9, 10])

# ## Calculate the difference vector
# difference = A - B

# distance = np.linalg.norm(difference)

# print("Euclidean distance between A and B:", distance)






### we have created 5, 3Dimension vectors and plotted in the 3D graphs and after that we gave our own 3D vector and it will plot the line nearest vector by calculating Euclidean distance


# # 1. Generate 5 random 3D vectors
# vectors = np.random.rand(5, 3)

# # 2. Assign a random class (0 or 1) to each vector
# classes = np.random.randint(0, 2, 5)

# # 3. Plot the vectors on a 3D Matplotlib graph
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# for vec, cls in zip(vectors, classes):
#     ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=("r" if cls == 1 else "b"))


# # 4. Get user input for a query point (3D vector)
# query_vector = np.array(list(map(float, input("Enter the query point (3D vector) separated by space: ").split())))

# # 5. Calculate the distance from the query vector to the 5 vectors and find the nearest neighbor
# distances = [euclidean(query_vector, vec) for vec in vectors]
# nearest_neighbor_index = np.argmin(distances)

# # 6. Output the class of the nearest neighbor
# print("The class of the nearest neighbor is:", classes[nearest_neighbor_index])

# # 7. Plot the query vector with a different color
# ax.quiver(0, 0, 0, query_vector[0], query_vector[1], query_vector[2], color="g")
# ax.set_xlabel("X-axis")
# ax.set_ylabel("Y-axis")
# ax.set_zlabel("Z-axis")

# plt.show()







#### doing scaler substraction where we are doing mean centering by subtracting each data points by their means this also known as mean normalization 


# ## Generate random 2D data with 100 points
# data = np.random.rand(100, 2)

# ## Plot the original data
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.scatter(data[:, 0], data[:, 1], label="Original Data")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Before Mean Centering")
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)
# plt.legend()


# ## Calculate the mean of the data along each dimension
# data_mean = np.mean(data, axis=0)

# ## Perform mean centering
# centered_data = data - data_mean


# ## Plot the mean-centered data
# plt.subplot(1, 2, 2)
# plt.scatter(centered_data[:, 0], centered_data[:, 1], label="Mean-Centered Data", color="orange")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("After Mean Centering")
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)
# plt.legend()

# plt.show()




#### doing min max noramlization


## Generate 100 random 2D vectors
# data = np.random.rand(100, 2) * np.array([500, 0.01])


# # Compute the min and max values for each feature
# min_values = np.min(data, axis=0)
# max_values = np.max(data, axis=0)

# # Apply min-max normalization (scaling)
# normalized_data = (data - min_values) / (max_values - min_values)

# # Plot the original data
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.scatter(data[:, 0], data[:, 1], label="Original Data")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.title("Before Min-Max Normalization")
# plt.legend()

# # Plot the normalized data
# plt.subplot(1, 2, 2)
# plt.scatter(normalized_data[:, 0], normalized_data[:, 1], label="Normalized Data", color="orange")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.title("After Min-Max Normalization")
# plt.legend()

# plt.show()






### rules while performing dot product 
### commutative 

# # Define three vectors
# A = np.array([1, 2, 3])
# B = np.array([4, 5, 6])
# C = np.array([7, 8, 9])

# # Calculate dot product using numpy.dot() function
# dot_product_AB = np.dot(A, B)
# dot_product_BA = np.dot(B, A)

# print(dot_product_AB)
# print(dot_product_BA)

# # Check commutative property: A • B = B • A
# if dot_product_AB == dot_product_BA:
#     print("The dot product is commutative.")
# else:
#     print("The dot product is not commutative.")




### distributive 


# # Check distributive property: A • (B + C) = A • B + A • C

# A = np.array([1, 2, 3])
# B = np.array([4, 5, 6])
# C = np.array([7, 8, 9])


# dot_product_A_BC = np.dot(A, B + C)
# dot_product_AB_plus_AC = np.dot(A, B) + np.dot(A, C)

# print(dot_product_A_BC)
# print(dot_product_AB_plus_AC)


# if dot_product_A_BC == dot_product_AB_plus_AC:
#     print("The dot product is distributive.")
# else:
#     print("The dot product is not distributive.")





#### to check the angle between two vectors (theta)

# def angle_between_vectors(a, b):
#     dot_product = np.dot(a, b)
#     magnitude_a = np.linalg.norm(a)
#     magnitude_b = np.linalg.norm(b)
#     cosine_theta = dot_product / (magnitude_a * magnitude_b)
#     angle_rad = np.arccos(cosine_theta)
#     angle_deg = np.degrees(angle_rad)
#     return angle_deg

# # Example vectors
# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])

# # Calculate the angle between the vectors
# angle = angle_between_vectors(a, b)
# print("Angle between vectors:", angle, "degrees")



### to check the cosine similarity using scikit learn 


# # Example vectors
# vector_a = np.array([1, 2, 3])
# vector_b = np.array([4, 5, 6])

# # Reshape the vectors to be 2D arrays
# vector_a = vector_a.reshape(1, -1)
# vector_b = vector_b.reshape(1, -1)

# # Calculate cosine similarity
# cosine_sim = cosine_similarity(vector_a, vector_b)
# print("Cosine Similarity between vectors:", cosine_sim[0][0])


## A cosine similarity of 0.974 indicates that the two vectors are very similar in direction, but not perfectly aligned. 
## Since cosine similarity ranges from -1 to 1, where 1 represents perfect similarity (the vectors are in the same direction), a cosine similarity of 0.974 suggests that the angle between the two vectors is very small (close to 0 degrees), but not exactly 0. 
## In practical terms, this means that the vectors have a high degree of correlation or alignment in terms of their directions, but they are not perfectly parallel. They may be slightly off from being perfectly aligned, but still very close.
## In many applications, a cosine similarity of 0.974 would be considered quite high, indicating a strong similarity between the vectors. However, the interpretation may vary depending on the specific context and the range of cosine similarities observed in your dataset.





### to check the cosine similarity using numpy 

# def cosine_similarity(a, b):
#     dot_product = np.dot(a, b)
#     norm_a = np.linalg.norm(a)     ### Calculate the Euclidean distance from the origin of a
#     norm_b = np.linalg.norm(b)     ### Calculate the Euclidean distance from the origin of b
#     similarity = dot_product / (norm_a * norm_b)
#     return similarity

# # Example vectors
# vector_a = np.array([1, 2, 3])
# vector_b = np.array([4, 5, 6])

# # Calculate cosine similarity
# cosine_sim = cosine_similarity(vector_a, vector_b)
# print("Cosine Similarity between vectors:", cosine_sim)




### checking cosine similarity between 3 vectors 

# # Define two vectors
# A = np.array([1, 2, 3])
# B = np.array([-4, -5, -6])
# C = np.array([5,5,5])


# # Calculate the cosine similarity
# cosine_similarity = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

# print("Cosine similarity between A and B:", cosine_similarity)

# # Calculate the cosine similarity
# cosine_similarity = np.dot(A, C) / (np.linalg.norm(A) * np.linalg.norm(C))

# print("Cosine similarity between A and C:", cosine_similarity)


# a = np.array([[1,2,3],[4,5,6],[4,5,7]])


# width = 100
# height = 100
# channels = 3
# image_tensor = np.random.rand(width, height, channels)

# # Access pixel values at position (50, 50)
# pixel_values = image_tensor[50, 50, :]
# print("Pixel values at position (50, 50):", pixel_values)





# # Assuming image_tensor represents your image as a 3D tensor
# # For example, let's say you have a 100x100 image with 3 color channels (RGB)
# # You can create a random image tensor for demonstration purposes
# width = 100
# height = 100
# channels = 3
# image_tensor = np.random.rand(width, height, channels)  # Random values for demonstration

# # Specify the pixel position you want to check
# x = 50  # x-coordinate of the pixel
# y = 50  # y-coordinate of the pixel

# # Access the color intensities at the specified pixel position
# intensity_red = image_tensor[x, y, 0]  # Red channel intensity
# intensity_green = image_tensor[x, y, 1]  # Green channel intensity
# intensity_blue = image_tensor[x, y, 2]  # Blue channel intensity

# # Print the intensities
# print("Intensity of red at pixel ({}, {}): {}".format(x, y, intensity_red))
# print("Intensity of green at pixel ({}, {}): {}".format(x, y, intensity_green))
# print("Intensity of blue at pixel ({}, {}): {}".format(x, y, intensity_blue))

