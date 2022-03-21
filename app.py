import streamlit as st
from scripts.predict import generate_combinations_with_input


from PIL import Image
image = Image.open('Food recipe.png')

st.image(image)



st.title("RECIPE GURU")
st.write("""
## Unleash the GORDON RAMSAY in you
""")

ingredients = st.text_input("Enter ingredients")



text_length = st.selectbox("Text Length",("100", "400", "600", "800", "1000", "1400"))

fuzziness = st.selectbox("Fuziness",("0.1", "0.2", "0.4", "0.6", "0.8", "1", "1.2", "1.4", "1.6"))

onclick = st.button("Generate recipe")



def recipes(ingredients, fuzziness, text_length):


    new_model = tf.keras.models.load_model('my_model.h5')

    output = new_model.generate_combinations_with_input(model_1_simplified, text_length, ingredients, fuzziness)
    
if onclick:
    recipes(ingredients, fuzziness, text_length)


