# File: src/data/generate_data.py

import pandas as pd
import numpy as np
from faker import Faker
import random
import os

def create_customer_dataset(num_customers=1000):
   

    fake = Faker('en_IN')
    
    print(f"Generating {num_customers} rows of customer data...")

    
    customer_ids = list(range(1, num_customers + 1))
    ages = [random.randint(18, 70) for _ in range(num_customers)]
    cities = [fake.city() for _ in range(num_customers)]
    last_seen_days = [random.randint(0, 180) for _ in range(num_customers)]

   
    preferred_channels = ['App Notification' if age < 35 else 'Email' for age in ages]
    preferred_timings = ['Evening (6 PM - 9 PM)' if age < 40 else 'Morning (9 AM - 12 PM)' for age in ages]
    
    
    avg_monthly_spend = [max(1000, 2500 + age * 50 + random.uniform(-1500, 1500)) for age in ages]
    
    items_purchased = [max(1, int(spend / 1000) + random.randint(-2, 2)) for spend in avg_monthly_spend]

    
    max_spend = max(avg_monthly_spend)
    spend_normalized = [s / max_spend for s in avg_monthly_spend]
    age_normalized = [a / 70 for a in ages]
    
    last_campaign_response = []
    for i in range(num_customers):
        
        prob = 0.6 * spend_normalized[i] + 0.2 * (1 - age_normalized[i]) + 0.1
        
        prob = np.clip(prob, 0.05, 0.95)
        
        
        last_campaign_response.append(1 if random.random() < prob else 0)

    print("Logical correlations have been applied.")

    
    data = {
        'customer_id': customer_ids,
        'age': ages,
        'city': cities,
        'preferred_channel': preferred_channels,
        'preferred_timing': preferred_timings,
        'avg_monthly_spend': [round(s, 2) for s in avg_monthly_spend],
        'items_purchased_last_6_months': items_purchased,
        'last_seen_days_ago': last_seen_days,
        'last_campaign_response': last_campaign_response
    }
    
    df = pd.DataFrame(data)
    print("Pandas DataFrame created successfully.")
    
    return df


if __name__ == "__main__":
    dataset = create_customer_dataset()
    
    
    output_dir = 'data/raw'
    file_path = os.path.join(output_dir, 'customer_data.csv')
    
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    dataset.to_csv(file_path, index=False)
    
    print(f"\nSuccess! Dataset saved to '{file_path}'.")
    print("\nHere's a preview of your data:")
    print(dataset.head())