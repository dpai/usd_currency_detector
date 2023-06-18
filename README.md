# Working example of a USD Denomination Detector in Streamlit

The user has to upload a square image of a bank note (US Dollars only) and the app will predict the denomination.

The embeddings for the training data is take from the following repo:
https://github.com/microsoft/banknote-net

The model for this repo is trained on the USD bank note embeddings only. 
The streamlit app uses the trained model to predict denomination given the input image.