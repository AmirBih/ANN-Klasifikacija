import streamlit as st
import numpy as np
#import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import pickle

#ucitavanje modela
model = tf.keras.models.load_model("model.h5")

#ucitavanje kodera i skalera
with open("label_encoder_gender.pkl","rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl","rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl","rb") as file:
    scaler = pickle.load(file)

#streamlit aplikacija 
st.title("Predviđanje Odliva Korisnika")

#polja za unošenje - korisnički unos
geography = st.selectbox("Geography", onehot_encoder_geo.categories[0])
spol = st.selectbox("Gender",label_encoder_gender.classes_)
godine = st.slider("Age",18,92)
balans = st.number_input("Balance")
kreditni_skor = st.number_input("Credit Score")
procijenjeni_prihod = st.number_input("Estimated Salary")
aktivnost = st.slider("Tenure",0,10)
broj_proizvoda = st.slider("Number of Products",1,4)
ima_kr_karticu = st.selectbox("Has Credit Card",[0,1])
je_aktivan_clan = st.selectbox("Is Active Member",[0,1])

#pripremanje podataka za unos
unos_data = pd.DataFrame({
    "Credit Score": [kreditni_skor],
    "Gender": [label_encoder_gender.transform([spol])[0]],
    "Age": [godine],
    "Tenure": [aktivnost],
    "Balance": [balans],
    "NumofProducts": [broj_proizvoda],
    "HasCrCard": [ima_kr_karticu],
    "IsActiveMember": [je_aktivan_clan],
    "EstimatedSalary": [procijenjeni_prihod]
})

#JednoVruceKodiranje Osobine Geography
geo_kodiran = onehot_encoder_geo.transform([[geography]]).toarray()
geo_kodiran_df = pd.DataFrame(geo_kodiran, columns = onehot_encoder_geo.get_feature_names_out(["Geography"]))

#Sjedinjavanje geo_kodiran as unos_data
unos_data = pd.concat([unos_data.reset_index(drop=True),geo_kodiran_df],axis=1)

#skaliranje podataka za unos
unos_data_skaliran = scaler.transform(unos_data)

#predviđanje odliva
predvidjanje = model.predict(unos_data_skaliran)
predvidjanje_vjerovatnoce = predvidjanje[0][0]


#zakljucak predviđanja
if predvidjanje_vjerovatnoce >0.5:
    st.write("Korisnik ce najvjerovatnije napustiti banku")
else:
    st.write("Korisnik najvjerovatnije ostaje u banci")





