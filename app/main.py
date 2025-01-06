import numpy as np
import pandas as pd
import streamlit as st
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="Breast Cancer Diagnosis Tool",
                       page_icon="👩🏻‍⚕️",
                       layout="wide",
                       initial_sidebar_state="expanded")

def load_and_prepare_data():
    data = pd.read_csv("static/data.csv")
    
    # Remove unnecessary columns
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    
    # Map diagnosis labels to numerical values
    data['diagnosis'] = data['diagnosis'].map({'M' : 1, 'B': 0})
    
    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    
    data = load_and_prepare_data()
    
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (standard error)", "radius_se"),
        ("Texture (standard error)", "texture_se"),
        ("Perimeter (standard error)", "perimeter_se"),
        ("Area (standard error)", "area_se"),
        ("Smoothness (standard error)", "smoothness_se"),
        ("Compactness (standard error)", "compactness_se"),
        ("Concavity (standard error)", "concavity_se"),
        ("Concave points (standard error)", "concave points_se"),
        ("Symmetry (standard error)", "symmetry_se"),
        ("Fractal dimension (standard error)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]
    
    input_dict = {}
    
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value = float(0),
            max_value = float(data[key].max()),
            value = float(data[key].mean())
        )
        
    return input_dict

def get_scaled_values(input_dict):
    data = load_and_prepare_data()
    
    X = data.drop(['diagnosis'], axis=1)
    
    scaled_dict = {}
    
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val)/ (max_val - min_val)
        scaled_dict[key] = scaled_value
        
    return scaled_dict
    
def get_radar_chart(input_data):
    
    input_data = get_scaled_values(input_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                  'Smoothness', 'Compactness', 'Concavity', 
                  'Concave Points', 'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
      r=[
         input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'],
         input_data['area_se'], input_data['smoothness_se'], input_data['compactness_se'],
         input_data['concavity_se'], input_data['concave points_se'], input_data['symmetry_se'],
         input_data['fractal_dimension_se']
         ],
      theta=categories,
      fill='toself',
      name='Mean Value'
    ))
    
    fig.add_trace(go.Scatterpolar(
      r=[
         input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
         input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
         input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
         input_data['fractal_dimension_mean']
         ],
      theta=categories,
      fill='toself',
      name='Standard Error'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']],
        theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
        fill='toself',
        name='Worst Value'
        )
    )

    fig.update_layout(
        polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1],
        tickfont=dict(color="black")
    )),
    showlegend=True,
    legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",  # Align legend to the bottom of the plot
            y=1.1,  # Move legend below the chart
            xanchor="center",  # Center the legend horizontally
            x=0.5,  # Center the legend horizontally
            font=dict(size=10)  # Adjust the legend font size
        ),
     modebar=dict(orientation="h")
    )


    return fig

def add_predictions(input_data):
    try:
        # Load the model
        model = pickle.load(open("model/model.pkl", "rb"))
        scaler = pickle.load(open("model/scaler.pkl", "rb"))
    except FileNotFoundError as e:
        st.error(f"File not found: {e.filename}. Ensure the model and scaler files are in the correct directory.")
        return
    except Exception as e:
        st.error(f"An error occurred while loading the files: {str(e)}")
        return

    try:
        # Prepare the input array
        input_array = np.array(list(input_data.values())).reshape(1, -1)
        input_array_scaled = scaler.transform(input_array)
        
        # Make predictions
        prediction = model.predict(input_array_scaled)
        st.subheader("Cell Cluster Prediction")
        st.write("The cell cluster is: ")
        if prediction[0] == 0:
            st.markdown('<div class="diagnosis benign">Benign</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="diagnosis malicious">Malicious</div>', unsafe_allow_html=True)
        
        # Display probabilities
        probas = model.predict_proba(input_array_scaled)
        st.write("Probability of being Benign: ", probas[0][0])
        st.write("Probability of being Malicious: ", probas[0][1])
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        return
    
    # Disclaimer
    st.write("This application is designed to support medical professionals in the diagnostic process; however, it is not intended to replace a professional medical diagnosis.")

def main():
    
    
    # Injecting custom CSS
    st.markdown("""
    <style>
    .stTitle {
        text-align: center;
    }

        /* Style all containers to ensure proper alignment */
    [data-testid="stVerticalBlock"], section[data-testid="stSidebar"] {
        width: 100%; /* Ensure containers occupy full width */
        max-width: 100%; /* Prevent child elements from overflowing */
        box-sizing: border-box; /* Include padding and borders in width */
        padding: 6px; /* Add padding inside containers */
        overflow: auto; /* Allow scrollbars if content overflows */
        margin: 0px 0; /* Add spacing between containers */
        background-color: #1e1e1e; /* Dark background */
        border: 1px solid #333; /* Subtle border for visibility */
        border-radius: 8px; /* Rounded corners */
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.5);
    }

    /* Benign Diagnosis */
    .diagnosis.benign {
        background-color: #90EE90;
        color: green;
        padding: 5px;
        padding-top: 0px;
        font-size: 18px;
        font-weight: 900;
        text-align: center;
    }

    /* Malicious Diagnosis */
    .diagnosis.malicious {
        background-color: #EC7063;
        color: #7b241c;
        padding: 5px;
        padding-top: 0px;
        font-size: 18px;
        font-weight: 900;
        text-align: center;
    }
     /* Ensure all content fits within its container */
    [data-testid="stVerticalBlock"] * {
        max-width: 100%; /* Prevent content from overflowing horizontally */
        word-wrap: break-word; /* Wrap text to prevent horizontal overflow */
    }

    /* Justify text within content blocks */
    [data-testid="stVerticalBlock"] p, [data-testid="stVerticalBlock"] h1, [data-testid="stVerticalBlock"] h2, [data-testid="stVerticalBlock"] h3, [data-testid="stVerticalBlock"] li {
        text-align: justify; /* Justify text to align evenly */
       /* Add space after text blocks */
    }

    [data-testid="stVerticalBlock"] h1, [data-testid="stVerticalBlock"] h2, [data-testid="stVerticalBlock"] h3 {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


    input_data = add_sidebar()
    
    with st.container():
        st.title("Breast Cancer Diagnosis Tool")
        st.write(
            "Connect this application to your cytology lab to assist in diagnosing breast cancer from tissue samples. "
            "Powered by a machine learning model, the app predicts whether a breast mass is benign or malignant based on measurements provided by your lab. "
            "Alternatively, you can manually adjust the measurements using the sliders available in the sidebar."
        )
        
    col1, col2 = st.columns([3, 1])  # Ist col will be 3 times as big as the IInd one
    
    with col1:
        st.plotly_chart(get_radar_chart(input_data), use_container_width=True)

        
    with col2:
        add_predictions(input_data)

if __name__ == '__main__':
    main()
