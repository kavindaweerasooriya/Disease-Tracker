from sklearn.cluster import KMeans
import numpy as np


num_symptoms = int(input("How many symptoms do you have? "))
user_symptoms = []
for i in range(num_symptoms):
    symptom = input(f"Enter symptom #{i+1}: ")
    user_symptoms.append(symptom)

all_symptoms = ['Fever', 'Headache', 'Nausea', 'Fatigue', 'Cough', 'Sore throat', 'Shortness of breath', 'Muscle aches', 'Loss of taste or smell']

user_data = np.zeros(len(all_symptoms))
for symptom in user_symptoms:
    if symptom in all_symptoms:
        user_data[all_symptoms.index(symptom)] = 1


diseases = {
    "Common cold": [1, 1, 0, 1, 1, 1, 0, 0, 0],
    "Influenza": [1, 1, 1, 1, 0, 1, 1, 1, 0],
    "COVID-19": [1, 1, 1, 1, 1, 1, 1, 1, 1],
    "Pneumonia": [1, 1, 1, 1, 1, 1, 1, 1, 1],
    "Bronchitis": [1, 1, 1, 1, 1, 0, 1, 1, 0],
    "Sinusitis": [1, 1, 1, 1, 0, 1, 0, 0, 0],
    "Strep throat": [1, 1, 1, 1, 1, 1, 0, 0, 0],
    "Tonsillitis": [1, 1, 1, 1, 1, 1, 0, 0, 0]
}


# Define the number of clusters
num_clusters = len(diseases)

# Create a KMeans object with the specified number of clusters
kmeans = KMeans(n_clusters=num_clusters)

# Fit the data to the KMeans object
kmeans.fit(np.array(list(diseases.values())))

# Predict the cluster for the user input
label = kmeans.predict(np.array([user_data]))

# Get the disease name from the label
disease_name = list(diseases.keys())[label[0]]

# Print the predicted disease
print(f"You may have {disease_name}")
