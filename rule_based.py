# Rule_based_approach

def predict_investment_risk(credit_rating_impact, total_E, total_S, total_G, predicted_sentiment):
    # Combine ESG scores
    combined_ESG = total_E + total_S + total_G
    
    # Define thresholds and conditions
    if credit_rating_impact == 0 and predicted_sentiment == 1 and combined_ESG >= 30:
        return "Low to Moderate Investment Risk - Stable Financial Health with Positive News and Moderate ESG Score"
    
    elif credit_rating_impact == 1 and predicted_sentiment == -1 and combined_ESG > 45:
        return "High Investment Risk - Potential Financial Instability with Negative News and High ESG Score"
    
    elif credit_rating_impact == 0 and predicted_sentiment == 1 and combined_ESG < 30:
        return "Low to Moderate Investment Risk - Stable Financial Health with Positive News but Low ESG Score"
    
    elif credit_rating_impact == 1 and predicted_sentiment == -1 and combined_ESG <= 45:
        return "High Investment Risk - Potential Financial Instability with Negative News but Moderate ESG Score"
    
    elif credit_rating_impact == 0 and predicted_sentiment == 0 and combined_ESG >= 30:
        return "Low to Moderate Investment Risk - Stable Financial Health with Neutral News and Moderate ESG Score"
    
    elif credit_rating_impact == 0 and predicted_sentiment == 0 and combined_ESG < 30:
        return "Low to Moderate Investment Risk - Stable Financial Health with Neutral News but Low ESG Score"
    
    elif credit_rating_impact == 1 and predicted_sentiment == 0 and combined_ESG > 45:
        return "High Investment Risk - Potential Financial Instability with Neutral News and High ESG Score"
    
    elif credit_rating_impact == 1 and predicted_sentiment == 0 and combined_ESG <= 45:
        return "High Investment Risk - Potential Financial Instability with Neutral News but Moderate ESG Score"

    else:
        return "Please provide valid inputs"
