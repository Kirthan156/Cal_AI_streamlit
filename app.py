import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import json
import plotly.express as px
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

# Load model
@st.cache_resource
def load_model():
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    num_classes = 101
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    
    try:
        state_dict = torch.load("models/food101_convnext_tiny_finetuned.pth", map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        raise e
    return model

# Load metadata
@st.cache_data
def load_metadata():
    try:
        with open("utils/food101_metadata.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        raise e

# Image preprocessing (ensure this matches training preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# App layout
st.set_page_config(page_title="🍽️ AI Calorie Estimator", layout="centered")
st.title("🍽️ AI-Based Food Calorie Estimator")
st.write("Upload a food image to predict its name and get nutrition information with charts, ingredients, and recipe.")

model = load_model()
metadata = load_metadata()

# Upload
uploaded_file = st.file_uploader("📸 Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with st.spinner("🔍 Analyzing image..."):
        try:
            # Forward pass
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)  # Convert logits to probabilities
            confidence, predicted = torch.max(probabilities, 1)  # Get the predicted class
            class_names = list(metadata.keys())
            predicted_class = class_names[predicted.item()]  # Get the class label

            # Confidence thresholding - Increase this threshold to filter out low-confidence predictions
            confidence_threshold = 0.7
            if confidence.item() < confidence_threshold:
                predicted_class = "Uncertain"
                food_info = {"calories": "N/A", "vegetarian": "N/A", "ingredients": "N/A", "recipe": "N/A", "nutrients": {}}
            else:
                food_info = metadata.get(predicted_class, {"calories": "N/A", "vegetarian": "N/A", "ingredients": "N/A", "recipe": "N/A", "nutrients": {}})
        except Exception as e:
            st.error(f"Error occurred during prediction: {e}")
            food_info = {"calories": "N/A", "vegetarian": "N/A", "ingredients": "N/A", "recipe": "N/A", "nutrients": {}}

    # Display result
    st.subheader(f"🍲 Prediction: **{predicted_class.replace('_', ' ').title()}**")
    st.markdown(f"**Confidence:** {confidence.item() * 100:.2f}%")

    if predicted_class != "Uncertain":
        st.markdown(f"**Calories (per 100g):** `{food_info['calories']} kcal`")
        st.markdown(f"**Vegetarian:** {'🟢 Yes' if food_info['vegetarian'] else '🔴 No'}")

        # Bar chart for nutrients
        nutrients = food_info["nutrients"]
        if nutrients:
            fig = px.bar(
                x=list(nutrients.keys()),
                y=[float(v.strip("g")) if isinstance(v, str) else v for v in nutrients.values()],
                labels={"x": "Nutrient", "y": "Grams"},
                color=list(nutrients.keys()),
                title="Nutrition Breakdown (per 100g)"
            )
            st.plotly_chart(fig)

            # Pie chart
            pie_fig = px.pie(
                names=list(nutrients.keys()),
                values=[float(v.strip("g")) if isinstance(v, str) else v for v in nutrients.values()],
                title="Nutrient Composition"
            )
            st.plotly_chart(pie_fig)
        else:
            st.write("Nutritional information is unavailable.")
    else:
        st.markdown("**The model is uncertain about the prediction. Please try another image.**")

    # Ingredients and Recipe
    st.subheader("📝 Ingredients")
    ingredients = food_info.get("ingredients", "No ingredients available.")
    st.write(ingredients)

    st.subheader("👨‍🍳 Recipe")
    recipe = food_info.get("recipe", "No recipe available.")
    st.write(recipe)

else:
    st.write("Please upload an image to predict.")
