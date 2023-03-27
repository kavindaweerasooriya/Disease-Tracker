import requests
from sklearn.cluster import KMeans
import numpy as np

APP_ID = 'ed13792a'
APP_KEY = 'e68ed663877f13c9307fea58c25781c4'

def get_symptoms():
    url = 'https://api.infermedica.com/v3/symptoms'
    headers = {
        'App-Id': APP_ID,
        'App-Key': APP_KEY
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return [symptom['name'] for symptom in response.json()]
    else:
        return []

def get_diseases():
    url = 'https://api.infermedica.com/v3/conditions'
    headers = {
        'App-Id': APP_ID,
        'App-Key': APP_KEY
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        diseases = {}
        for disease in response.json():
            symptoms = [symptom['id'] for symptom in disease['symptoms']]
            diseases[disease['name']] = symptoms
        return diseases
    else:
        return {}

def get_user_data(symptoms):
    num_symptoms = int(input("How many symptoms do you have? "))
    user_symptoms = []
    for i in range(num_symptoms):
        symptom = input(f"Enter symptom #{i+1}: ")
        user_symptoms.append(symptom)
    user_data = np.zeros(len(symptoms))
    for symptom in user_symptoms:
        if symptom in symptoms:
            user_data[symptoms.index(symptom)] = 1
    return user_data

def run_diagnosis():

    symptoms = get_symptoms()
    diseases = get_diseases()

    user_data = get_user_data(symptoms)

    num_clusters = len(diseases)

    kmeans = KMeans(n_clusters=num_clusters)

    kmeans.fit(np.array(list(diseases.values())))

    label = kmeans.predict(np.array([user_data]))
    disease_name = list(diseases.keys())[label[0]]
    print(f"You may have {disease_name}")

run_diagnosis()


# import requests
# from sklearn.cluster import KMeans
# import numpy as np

# # Set up the Infermedica API credentials
# APP_ID = 'ed13792a'
# APP_KEY = 'e68ed663877f13c9307fea58c25781c4'

# # Get the list of symptoms from the Infermedica API
# def get_symptoms():
#     url = 'https://api.infermedica.com/v3/symptoms'
#     headers = {
#         'App-Id': APP_ID,
#         'App-Key': APP_KEY
#     }
#     response = requests.get(url, headers=headers)
#     if response.status_code == 200:
#         return [symptom['name'] for symptom in response.json()]
#     else:
#         return []

# # Get the list of diseases and their symptoms from the Infermedica API
# def get_diseases():
#     url = 'https://api.infermedica.com/v3/conditions'
#     headers = {
#         'App-Id': APP_ID,
#         'App-Key': APP_KEY
#     }
#     response = requests.get(url, headers=headers)
#     if response.status_code == 200:
#         diseases = {}
#         for disease in response.json():
#             symptoms = [symptom['id'] for symptom in disease['symptoms']]
#             diseases[disease['name']] = symptoms
#         return diseases
#     else:
#         return {}

# # Ask the user for their symptoms and return a numpy array of 1s and 0s
# def get_user_data(symptoms):
#     num_symptoms = int(input("How many symptoms do you have? "))
#     user_symptoms = []
#     for i in range(num_symptoms):
#         symptom = input(f"Enter symptom #{i+1}: ")
#         user_symptoms.append(symptom)
#     user_data = np.zeros(len(symptoms))
#     for symptom in user_symptoms:
#         if symptom in symptoms:
#             user_data[symptoms.index(symptom)] = 1
#     return user_data

# # Main function to run the diagnosis
# def run_diagnosis():
#     # Get the list of symptoms and diseases from the Infermedica API
#     symptoms = get_symptoms()
#     diseases = get_diseases()

#     # Ask the user for their symptoms
#     user_data = get_user_data(symptoms)

#     # Define the number of clusters
#     num_clusters = len(diseases)

#     # Create a KMeans object with the specified number of clusters
#     kmeans = KMeans(n_clusters=num_clusters+1)

#     # Fit the data to the KMeans object
#     kmeans.fit(np.array(list(diseases.values())))

#     # Predict the cluster for the user input
#     label = kmeans.predict(np.array([user_data]))

#     # Get the disease name from the label
#     disease_name = list(diseases.keys())[label[0]]

#     # Print the predicted disease
#     print(f"You may have {disease_name}")

# # def run_diagnosis():
# #     # Get the list of symptoms and diseases from the Infermedica API
# #     symptoms = get_symptoms()
# #     diseases = get_diseases()

# #     # Check if the diseases dictionary is empty
# #     if not diseases:
# #         print("Unable to retrieve diseases information from the API. Please try again later.")
# #         return

# #     # Ask the user for their symptoms
# #     user_data = get_user_data(symptoms)

# #     # Define the number of clusters
# #     num_clusters = len(diseases)

# #     # Create a KMeans object with the specified number of clusters
# #     kmeans = KMeans(n_clusters=num_clusters)

# #     # Check if there are any symptoms associated with the diseases
# #     if not list(diseases.values()):
# #         print("Unable to retrieve symptom information for the diseases. Please try again later.")
# #         return

# #     # Fit the data to the KMeans object
# #     kmeans.fit(np.array(list(diseases.values())))

# #     # Predict the cluster for the user input
# #     label = kmeans.predict(np.array([user_data]))

# #     # Get the disease name from the label
# #     disease_name = list(diseases.keys())[label[0]]

# #     # Print the predicted disease
# #     print(f"You may have {disease_name}")


# # Run the diagnosis
# run_diagnosis()
