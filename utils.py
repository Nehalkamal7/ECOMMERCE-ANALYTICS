# utils.py
import pandas as pd
import numpy as np
import json
import ast
import re
from datetime import datetime

# Data loading function
def load_data(file):
    """Load and parse the dataset"""
    df = pd.read_csv(file)
    return df

# Advanced feature extraction functions
def robust_parse_json(data_str):
    """Parse inconsistent JSON/dict formats"""
    if pd.isna(data_str):
        return []
    
    try:
        if not isinstance(data_str, str):
            data_str = str(data_str)
        
        data_str = data_str.strip()
        data_str = data_str.replace("'", '"')
        data_str = data_str.replace("None", "null")
        data_str = data_str.replace("True", "true")
        data_str = data_str.replace("False", "false")
        
        try:
            parsed = json.loads(data_str)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(data_str)
                return parsed if isinstance(parsed, list) else [parsed]
            except:
                json_pattern = r'\[.*\]|\{.*\}'
                matches = re.findall(json_pattern, data_str, re.DOTALL)
                if matches:
                    parsed = json.loads(matches[0])
                    return parsed if isinstance(parsed, list) else [parsed]
                return []
    except Exception as e:
        return []

def extract_purchase_features(purchase_str):
    """Extract advanced features from purchase history"""
    purchases = robust_parse_json(purchase_str)
    
    total_spent = 0
    total_items = 0
    categories = []
    dates = []
    prices = []
    brands = []
    
    for item in purchases:
        if isinstance(item, dict):
            # Extract price
            price_keys = ['Price', 'price', 'Amount', 'amount', 'value']
            for key in price_keys:
                if key in item:
                    try:
                        price = float(item[key])
                        total_spent += price
                        prices.append(price)
                        total_items += 1
                        break
                    except:
                        pass
            
            # Extract category
            category_keys = ['Category', 'category', 'Product Category', 'product_category', 'type']
            for key in category_keys:
                if key in item and item[key]:
                    categories.append(str(item[key]))
                    break
            
            # Extract date
            date_keys = ['Date', 'date', 'Purchase Date', 'purchase_date', 'timestamp']
            for key in date_keys:
                if key in item and item[key]:
                    dates.append(item[key])
                    break
            
            # Extract brand
            brand_keys = ['Brand', 'brand', 'Manufacturer', 'manufacturer']
            for key in brand_keys:
                if key in item and item[key]:
                    brands.append(str(item[key]))
                    break
    
    # Calculate advanced metrics
    avg_spent = total_spent / total_items if total_items > 0 else 0
    unique_categories = len(set(categories)) if categories else 0
    price_std = np.std(prices) if len(prices) > 1 else 0
    price_range = max(prices) - min(prices) if prices else 0
    unique_brands = len(set(brands)) if brands else 0
    
    # Calculate purchase pattern metrics
    if len(dates) > 1:
        try:
            date_objs = [pd.to_datetime(d, errors='coerce') for d in dates]
            date_objs = [d for d in date_objs if pd.notna(d)]
            if len(date_objs) > 1:
                date_objs.sort()
                time_diffs = [(date_objs[i+1] - date_objs[i]).days for i in range(len(date_objs)-1)]
                avg_time_between_purchases = np.mean(time_diffs) if time_diffs else 0
                purchase_regularity = np.std(time_diffs) if len(time_diffs) > 1 else 0
            else:
                avg_time_between_purchases = 0
                purchase_regularity = 0
        except:
            avg_time_between_purchases = 0
            purchase_regularity = 0
    else:
        avg_time_between_purchases = 0
        purchase_regularity = 0
    
    return pd.Series({
        'Total_Spent': total_spent,
        'Total_Purchases': total_items,
        'Avg_Purchase_Value': avg_spent,
        'Unique_Categories': unique_categories,
        'Price_Variability': price_std,
        'Price_Range': price_range,
        'Unique_Brands': unique_brands,
        'Avg_Time_Between_Purchases': avg_time_between_purchases,
        'Purchase_Regularity': purchase_regularity,
        'Last_Purchase_Date': dates[-1] if dates else None
    })

def extract_browsing_features(browsing_str):
    """Extract features from browsing history"""
    sessions = robust_parse_json(browsing_str)
    
    total_sessions = len(sessions)
    timestamps = []
    categories_browsed = []
    session_durations = []
    
    for session in sessions:
        if isinstance(session, dict):
            # Extract timestamp
            time_keys = ['Timestamp', 'timestamp', 'Time', 'time', 'start_time']
            for key in time_keys:
                if key in session:
                    try:
                        ts = pd.to_datetime(session[key], errors='coerce', utc=True)
                        if pd.notna(ts):
                            timestamps.append(ts)
                    except:
                        pass
                    break
            
            # Extract product category browsed
            category_keys = ['Product Category', 'product_category', 'Category', 'category', 'page_type']
            for key in category_keys:
                if key in session and session[key]:
                    categories_browsed.append(str(session[key]))
                    break
            
            # Extract session duration
            duration_keys = ['Duration', 'duration', 'time_spent', 'session_length']
            for key in duration_keys:
                if key in session:
                    try:
                        duration = float(session[key])
                        session_durations.append(duration)
                    except:
                        pass
                    break
    
    # Calculate metrics
    avg_session_duration = np.mean(session_durations) if session_durations else 0
    total_browsing_time = sum(session_durations) if session_durations else 0
    unique_browsed_categories = len(set(categories_browsed)) if categories_browsed else 0
    
    # Time-based patterns
    if len(timestamps) >= 2:
        try:
            timestamps_sorted = sorted(timestamps)
            time_diffs = [(timestamps_sorted[i+1] - timestamps_sorted[i]).total_seconds() 
                         for i in range(len(timestamps_sorted)-1)]
            avg_session_gap = np.mean(time_diffs) if time_diffs else 0
            browsing_regularity = np.std(time_diffs) if len(time_diffs) > 1 else 0
        except:
            avg_session_gap = 0
            browsing_regularity = 0
    else:
        avg_session_gap = 0
        browsing_regularity = 0
    
    return pd.Series({
        'Total_Sessions': total_sessions,
        'Avg_Session_Duration': avg_session_duration,
        'Total_Browsing_Time': total_browsing_time,
        'Unique_Browsed_Categories': unique_browsed_categories,
        'Avg_Session_Gap': avg_session_gap,
        'Browsing_Regularity': browsing_regularity,
        'Browsing_Intensity': total_sessions / 30 if total_sessions > 0 else 0
    })

def extract_text_features(review_str):
    """Extract text-based features from reviews"""
    try:
        reviews = robust_parse_json(review_str)
        
        all_reviews = []
        ratings = []
        
        if isinstance(reviews, dict):
            # Handle different structures
            for key, value in reviews.items():
                if isinstance(value, dict):
                    if 'Review' in value:
                        all_reviews.append(str(value['Review']))
                    elif 'Review Text' in value:
                        all_reviews.append(str(value['Review Text']))
                    elif 'review' in value:
                        all_reviews.append(str(value['review']))
                    
                    if 'Rating' in value:
                        try:
                            ratings.append(float(value['Rating']))
                        except:
                            pass
        
        # Calculate metrics
        avg_rating = np.mean(ratings) if ratings else 0
        review_count = len(ratings)
        
        # Text-based features
        if all_reviews:
            review_lengths = [len(str(r).split()) for r in all_reviews]
            avg_review_length = np.mean(review_lengths) if review_lengths else 0
            review_sentiment = analyze_sentiment(all_reviews)
        else:
            avg_review_length = 0
            review_sentiment = 0
        
        return pd.Series({
            'Avg_Rating': avg_rating,
            'Review_Count': review_count,
            'Avg_Review_Length': avg_review_length,
            'Review_Sentiment': review_sentiment,
            'Rating_Variance': np.var(ratings) if len(ratings) > 1 else 0
        })
    except Exception as e:
        return pd.Series({
            'Avg_Rating': 0,
            'Review_Count': 0,
            'Avg_Review_Length': 0,
            'Review_Sentiment': 0,
            'Rating_Variance': 0
        })

def analyze_sentiment(reviews):
    """Simple sentiment analysis"""
    positive_words = ['great', 'excellent', 'awesome', 'love', 'perfect', 
                     'good', 'nice', 'satisfied', 'happy', 'recommend',
                     'best', 'amazing', 'wonderful', 'fantastic', 'superb']
    
    negative_words = ['bad', 'poor', 'terrible', 'horrible', 'awful',
                     'disappointed', 'waste', 'broken', 'defective', 'useless']
    
    positive_count = 0
    negative_count = 0
    total_words = 0
    
    for review in reviews:
        words = str(review).lower().split()
        total_words += len(words)
        positive_count += sum(1 for word in words if word in positive_words)
        negative_count += sum(1 for word in words if word in negative_words)
    
    if total_words > 0:
        sentiment = (positive_count - negative_count) / total_words
        return sentiment
    return 0