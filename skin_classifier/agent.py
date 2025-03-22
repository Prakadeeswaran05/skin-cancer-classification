import os
import asyncio
import nest_asyncio
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image as PILImage

# Apply nest_asyncio to allow nested event loops - fixes the asyncio error
nest_asyncio.apply()

# Set device for PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define all functions first before using them

# Function to create data transformations for EfficientNet
def get_data_transforms():
    """Create data transformations for EfficientNet"""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Function to load EfficientNet model
def load_model(model_path, num_classes=2):
    """Load the EfficientNet model with saved weights"""
    from torchvision import models
    
    model = models.efficientnet_b0(weights=None)
    
    # Customize classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model

# Function to predict with EfficientNet
def predict_with_efficientnet(model, image, class_names):
    """Predict the class of a single image using EfficientNet"""
    if model is None:
        return None

    try:
        # Import here to delay until needed
        from torchvision import transforms
        
        # Preprocess image
        transform = get_data_transforms()
        img_tensor = transform(image).unsqueeze(0)
        img_tensor = img_tensor.to(device)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            predicted_class_idx = torch.argmax(probabilities).item()
            
        # Format results
        class_probs = {
            class_names[i]: probabilities[i].item() * 100 
            for i in range(len(class_names))
        }
        
        sorted_probs = sorted(
            class_probs.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        result = {
            'predicted_class': class_names[predicted_class_idx],
            'confidence': probabilities[predicted_class_idx].item() * 100,
            'all_probabilities': sorted_probs,
        }
        
        return result
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Initialize Gemini agent if API key is provided
def initialize_gemini_agent():
    if not st.session_state.GOOGLE_API_KEY:
        return None
    
    try:
        # Import agno libraries only when needed
        from agno.agent import Agent
        from agno.models.google import Gemini
        from agno.tools.duckduckgo import DuckDuckGoTools
        
        # Create agent with nested asyncio support
        medical_agent = Agent(
            model=Gemini(
                id="gemini-2.0-flash",
                api_key=st.session_state.GOOGLE_API_KEY,
                # You can adjust these parameters to "tune" the model behavior
                temperature=0.2,  # Lower temperature for more focused, deterministic responses
                top_p=0.9,        # Controls diversity
                top_k=40          # How many tokens to consider
            ),
            tools=[DuckDuckGoTools()],
            markdown=True
        )
        return medical_agent
    except Exception as e:
        st.error(f"Error initializing Gemini agent: {e}")
        return None

# Function to analyze image with Gemini
def analyze_with_gemini(agent, image_path, ml_model_section, query_template):
    try:
        from agno.media import Image as AgnoImage
        
        # Create AgnoImage object
        agno_image = AgnoImage(filepath=image_path)
        
        # Prepare complete query with ML model results
        full_query = query_template.format(ml_model_section=ml_model_section)
        
        # Run analysis with Gemini
        response = agent.run(full_query, images=[agno_image])
        return response.content
    except Exception as e:
        return f"Error analyzing image with Gemini: {str(e)}"

# Initialize session state variables
if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = None
if "model" not in st.session_state:
    st.session_state.model = None
if "class_names" not in st.session_state:
    st.session_state.class_names = ["benign", "malignant"]

# Medical Analysis Query specialized for skin lesions
query_template = """
You are a highly skilled dermatologist with expertise in skin cancer detection. Analyze the patient's skin lesion image and structure your response as follows:

### 1. Image Assessment
- Comment on image quality and visibility of the lesion
- Describe the anatomical location (if identifiable)
- Note any technical limitations in the image

### 2. Lesion Characteristics
- Describe the lesion's appearance systematically using ABCDE criteria:
  - Asymmetry
  - Border irregularity
  - Color variation
  - Diameter/size approximation
  - Evolution/elevation (if discernible)
- Note texture, surrounding skin, and any secondary features

### 3. Gemini Classification Analysis
- Evaluate benign vs. malignant visual indicators
- Assign a probability estimate (e.g., "80% likely benign") based on visual assessment
- Support classification with observed evidence
- State clearly that this is Gemini's AI assessment based on visual patterns

### 4. Supervised Machine Learning Assessment
{ml_model_section}

### 5. Patient-Friendly Explanation
- Explain the findings in simple, clear language
- Clarify what the classification means in practical terms
- Outline appropriate next steps based on the analysis
- Address common concerns about skin lesions

### 6. Research Context
Use the DuckDuckGo search tool to:
- Find recent research on similar lesion types
- Search for standard diagnostic and treatment protocols for this classification
- Provide relevant medical resources about skin cancer detection
- Include 2-3 key references specific to this type of lesion

Remember to emphasize that this analysis should be followed by an in-person dermatological consultation for definitive diagnosis. Format your response using clear markdown headers and bullet points.
"""

# Create sidebar for configuration
with st.sidebar:
    st.title("ℹ️ Configuration")
    
    # Google API key configuration
    if not st.session_state.GOOGLE_API_KEY:
        api_key = st.text_input(
            "Enter your Google API Key:",
            type="password"
        )
        st.caption(
            "Get your API key from [Google AI Studio]"
            "(https://aistudio.google.com/apikey) 🔑"
        )
        if api_key:
            st.session_state.GOOGLE_API_KEY = api_key
            st.success("API Key saved!")
            st.rerun()
    else:
        st.success("API Key is configured")
        if st.button("🔄 Reset API Key"):
            st.session_state.GOOGLE_API_KEY = None
            st.rerun()
    
    # Model section in sidebar
    st.header("🧠 Model Configuration")
    model_path = st.text_input(
        "EfficientNet Model Path",
        value="model.pth" if os.path.exists("model.pth") else "",
        help="Path to your trained EfficientNet model (.pth file)"
    )
    
    # Class names - fixed for skin lesion
    class_names = ["benign", "malignant"]
    st.text("Classification Categories: Benign, Malignant")
    st.session_state.class_names = class_names
    
    # Load model button
    if model_path:
        if st.button("Load Skin Lesion Model"):
            with st.spinner("Loading model..."):
                try:
                    # Load the model with fixed 2 classes (benign, malignant)
                    model = load_model(model_path, num_classes=2)
                    st.session_state.model = model
                    st.success("Skin lesion classification model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {e}")
    
    # Info and disclaimer
    st.info(
        "This tool specializes in skin lesion classification (benign vs. malignant) "
        "using EfficientNet and AI-powered medical analysis."
    )
    st.warning(
        "⚠DISCLAIMER: This tool is for educational and informational purposes only. "
        "All analyses should be reviewed by qualified healthcare professionals. "
        "Do not make medical decisions based solely on this analysis."
    )

# Main app UI
st.title("🔬 Skin Lesion Classification & Analysis")
st.write("Upload a skin lesion image to detect if it's benign or malignant")

# Create containers for better organization
upload_container = st.container()
image_container = st.container()
analysis_container = st.container()

# Upload widget
with upload_container:
    uploaded_file = st.file_uploader(
        "Upload Skin Lesion Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of the skin lesion for analysis"
    )
    
    st.markdown("""
    ℹ️ **Tips for best results:**
    - Use well-lit, clear images
    - Center the lesion in the frame
    - Include some surrounding skin for context
    - Avoid shadows or glare
    """)

# Display and analyze image if uploaded
if uploaded_file is not None:
    with image_container:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Open and resize image
            image = PILImage.open(uploaded_file)
            width, height = image.size
            aspect_ratio = width / height
            new_width = 500
            new_height = int(new_width / aspect_ratio)
            resized_image = image.resize((new_width, new_height))
            
            st.image(
                resized_image,
                caption="Uploaded Skin Lesion Image",
                use_container_width=True
            )
            
            # Analysis buttons
            col_left, col_right = st.columns(2)
            with col_left:
                analyze_button = st.button(
                    "🔍 Full Lesion Analysis",
                    type="primary",
                    use_container_width=True,
                    disabled=not st.session_state.GOOGLE_API_KEY
                )
            with col_right:
                classify_button = st.button(
                    "🔬 Quick Classification",
                    type="secondary",
                    use_container_width=True,
                    disabled=st.session_state.model is None
                )
    
    with analysis_container:
        # EfficientNet classification
        if classify_button and st.session_state.model is not None:
            with st.spinner("🔄 Classifying image..."):
                try:
                    # Run EfficientNet prediction
                    prediction = predict_with_efficientnet(
                        st.session_state.model, 
                        image, 
                        st.session_state.class_names
                    )
                    
                    if prediction:
                        st.success("Classification complete!")
                        
                        # Create visualization for prediction results
                        st.markdown("## 📊 Skin Lesion Classification Results")
                        
                        # Display main prediction with confidence
                        is_malignant = prediction['predicted_class'].lower() == "malignant"
                        
                        # Use color coding for results
                        if is_malignant:
                            st.error(f"⚠️ **PREDICTION: MALIGNANT** with {prediction['confidence']:.2f}% confidence")
                            st.warning("This is an AI prediction only. Please consult a dermatologist immediately for proper diagnosis.")
                        else:
                            st.success(f"✅ **PREDICTION: BENIGN** with {prediction['confidence']:.2f}% confidence")
                            st.info("While the AI suggests this lesion is benign, regular skin checks with a dermatologist are still recommended.")
                        
                        # Show all class probabilities as a bar chart
                        probs_data = {cls: prob for cls, prob in prediction['all_probabilities']}
                        st.bar_chart(probs_data)
                        
                        st.markdown("---")
                        st.caption(
                            "Note: This classification is generated by an AI model and should be confirmed by "
                            "a qualified dermatologist. No AI system can replace professional medical diagnosis."
                        )
                        
                        # Add ABCDE criteria reminder for self-checking
                        with st.expander("What should I look for in a skin lesion? (ABCDE criteria)"):
                            st.markdown("""
                            The ABCDE rule can help you remember features that might suggest melanoma:
                            
                            - **A**symmetry: One half doesn't match the other half
                            - **B**order: Irregular, ragged, notched, or blurred edges
                            - **C**olor: Variation in color or multiple colors (black, brown, tan, red, white, or blue)
                            - **D**iameter: Larger than 6mm (about the size of a pencil eraser)
                            - **E**volving: Changing in size, shape, color, or features
                            
                            If you notice any of these warning signs, please consult a dermatologist promptly.
                            """)
                except Exception as e:
                    st.error(f"Classification error: {e}")
        
        # Full analysis with Gemini
        if analyze_button:
            with st.spinner("🔄 Analyzing image... Please wait."):
                try:
                    # First get EfficientNet prediction if model is loaded
                    ml_model_section = ""
                    if st.session_state.model is not None:
                        prediction = predict_with_efficientnet(
                            st.session_state.model, 
                            image, 
                            st.session_state.class_names
                        )
                        
                        if prediction:
                            ml_model_section = f"""
- **Machine Learning Model**: EfficientNet B0
- **Primary classification**: {prediction['predicted_class']} (Confidence: {prediction['confidence']:.2f}%)
- **All classifications**:
"""
                            for cls, prob in prediction['all_probabilities']:
                                ml_model_section += f"  - {cls}: {prob:.2f}%\n"
                    else:
                        ml_model_section = "- No machine learning model prediction available"
                    
                    # Save image to temporary file
                    temp_path = "temp_resized_image.png"
                    resized_image.save(temp_path)
                    
                    # Initialize Gemini agent
                    medical_agent = initialize_gemini_agent()
                    
                    if medical_agent:
                        # Run analysis with Gemini
                        response_content = analyze_with_gemini(
                            medical_agent, 
                            temp_path, 
                            ml_model_section, 
                            query_template
                        )
                        
                        st.markdown("## 📋 Complete Analysis Results")
                        st.markdown("---")
                        st.markdown(response_content)
                        st.markdown("---")
                        st.caption(
                            "Note: This analysis is generated by AI and should be reviewed by "
                            "a qualified healthcare professional."
                        )
                    else:
                        st.error("Unable to initialize Gemini agent. Please check your API key.")
                except Exception as e:
                    st.error(f"Analysis error: {e}")
else:
    st.info("👆 Please upload a skin lesion image to begin analysis")