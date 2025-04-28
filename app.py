import streamlit as st

import pandas as pd
import numpy as np
# Imports needed for fetching and displaying images
import requests
from PIL import Image
from io import BytesIO

from sklearn.datasets import load_iris

from sklearn.ensemble import RandomForestClassifier

st.cache_resource

def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns = iris.feature_names)
    df['species'] = iris.target

    return df, iris.target_names

df, target_names = load_data()


model = RandomForestClassifier()

model.fit(df.iloc[:,:-1], df['species'])

st.sidebar.title("Iris Classifier")

sepal_length = st.sidebar.slider("Sepal Length", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal Width", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal Length", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))    
petal_width = st.sidebar.slider("Petal Width", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]


# prediction

prediction = model.predict(input_data)
predictted_species = target_names[prediction[0]]



# 4) Mapping species → image URL  -------------------------------------------
species_to_url = {
    "setosa":    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Irissetosa1.jpg/1200px-Irissetosa1.jpg",
    "versicolor":"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS8d8MPVvmJhmWJKlar9ekxN1TtM8k27adbgA&s",
    "virginica": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Iris_virginica_2.jpg/500px-Iris_virginica_2.jpg"
}


# # 5) Display text + image  ---------------------------------------------------
# st.title(f"Predicted Species: **{predictted_species}**")
# # Load image from URL and display
# response = requests.get(species_to_url[predictted_species])
# img = Image.open(BytesIO(response.content))
# st.image(img, caption=predictted_species,width= 150, use_column_width=False)
st.write(f"**Predicted Species:** {species}")
# Direct URL loading; Streamlit handles fetch errors
st.image(species_to_url[species], caption=species, width=300)


# st.write(f"Predicted Species: {predictted_species}")

#Trying to adding footer to the app

import streamlit as st

footer_html = """
<style>
.footer {
  position: fixed;
  left: 0;
  bottom: 0;
  width: 100%;
  background-color: #333;
  color: #ffffff;
  text-align: center;
  padding: 0.5em 0;
}
</style>
<div class="footer">
  <p>Developed with ❤️ by <a href="https://www.linkedin.com/in/nur-mahammad-7b7b97219/" target="_blank">Nur</a></p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
