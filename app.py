import streamlit as st
import pandas as pd
import joblib
import os


@st.cache_resource
def load_assets():
    """Loads all necessary models, scalers, and data."""
    df = pd.read_csv(os.path.join('data', 'raw', 'customer_data.csv'))
    rf_model = joblib.load(os.path.join('models', 'rf_model.pkl'))
    kmeans_model = joblib.load(os.path.join('models', 'kmeans_model.pkl'))
    scaler = joblib.load(os.path.join('models', 'scaler.pkl'))
    model_columns = joblib.load(os.path.join('models', 'model_columns.pkl'))
    return df, rf_model, kmeans_model, scaler, model_columns

df, rf_model, kmeans_model, scaler, model_columns = load_assets()


def personalize_message(offer, customer_details):
    """Uses NLP (template filling) to personalize the marketing message."""
    city = customer_details.get('city', 'there')
    last_seen = customer_details.get('last_seen_days_ago', 100)
    
    if last_seen < 30:
        greeting = f"Hey! As one of our valued customers in {city},"
    else:
        greeting = f"We've missed you! To welcome you back,"
        
    return f"{greeting} we're recommending this for you: **{offer}**"


def get_recommendation(customer_id):
    """Generates a personalized recommendation for a customer."""
    customer_data = df[df['customer_id'] == customer_id]
    if customer_data.empty:
        return None

    
    segment_features = ['age', 'avg_monthly_spend', 'items_purchased_last_6_months']
    customer_segment_features_scaled = scaler.transform(customer_data[segment_features])
    segment = kmeans_model.predict(customer_segment_features_scaled)[0]
    
    
    customer_data['segment'] = segment
    features_to_include = ['age', 'avg_monthly_spend', 'items_purchased_last_6_months', 'last_seen_days_ago', 'segment', 'preferred_channel', 'preferred_timing']
    customer_prepared = pd.get_dummies(customer_data[features_to_include])
    customer_aligned = customer_prepared.reindex(columns=model_columns, fill_value=0)

    
    base_probability = rf_model.predict_proba(customer_aligned)[0][1]

    
    if segment in [0, 3]: 
        offer_A, prob_A = "Early access to our new Premium Collection", base_probability * 1.20
        offer_B, prob_B = "A 15% 'thank you' discount", base_probability * 1.05
    else: 
        offer_A, prob_A = "A special 25% discount on your next purchase", base_probability * 1.20
        offer_B, prob_B = "A welcome bonus of 100 loyalty points", base_probability * 1.10

    
    return {
        "customer_details": customer_data.to_dict(orient='records')[0],
        "offer_A": {"name": offer_A, "probability": prob_A},
        "offer_B": {"name": offer_B, "probability": prob_B},
        "winner": "A" if prob_A > prob_B else "B"
    }


st.set_page_config(layout="wide")
st.title("AI Marketing Personalization Engine")
st.write("Enter a customer ID to get a personalized recommendation and a simulated A/B test.")

customer_id = st.number_input("Enter Customer ID:", min_value=1, max_value=1000, step=1, value=101)

if st.button("Generate Recommendation"):
    recommendation = get_recommendation(customer_id)
    if recommendation:
        winner_key = recommendation['winner']
        loser_key = "B" if winner_key == "A" else "A"
        winner = recommendation[f'offer_{winner_key}']
        loser = recommendation[f'offer_{loser_key}']

        # Call NLP function for the final message
        personalized_text = personalize_message(winner['name'], recommendation['customer_details'])
        
        st.subheader(f"Recommendations for Customer #{recommendation['customer_details']['customer_id']}")
        st.success("**Personalized Message:**")
        st.markdown(f"> {personalized_text}")
        
        st.markdown("---")
        st.info("**A/B Test Simulation Results:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label=f"Winning Offer ({winner_key})", value=f"{winner['probability']:.2%}")
        with col2:
            st.metric(label=f"Alternative Offer ({loser_key})", value=f"{loser['probability']:.2%}", delta=f"{(loser['probability']-winner['probability']):.2%}")

        with st.expander("View Full Customer Details"):
            st.json(recommendation['customer_details'])
    else:
        st.error(f"Customer ID {customer_id} not found.")