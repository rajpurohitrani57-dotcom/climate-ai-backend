import torch
import numpy as np
from captum.attr import IntegratedGradients, FeatureAblation, Occlusion

def explain_prediction(model, input_tensor, method='integrated_gradients'):
    """
    Explain why the model made a specific prediction.
    Returns: Attribution scores + Human-readable explanation.
    """
    model.eval()
    
    if method == 'integrated_gradients':
        ig = IntegratedGradients(model)
        attributions = ig.attribute(input_tensor, target=0, n_steps=50)
        
    elif method == 'feature_ablation':
        fa = FeatureAblation(model)
        attributions = fa.attribute(input_tensor, target=0)
        
    elif method == 'occlusion':
        occ = Occlusion(model)
        attributions = occ.attribute(
            input_tensor,
            target=0,
            sliding_window_shapes=(1, 1, 3, 3),
            strides=(1, 1, 2, 2)
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    attributions_np = attributions.detach().numpy()
    attributions_np = (attributions_np - np.min(attributions_np)) / (np.max(attributions_np) - np.min(attributions_np) + 1e-8)
    
    # Generate human-readable explanation
    explanation = generate_human_explanation(attributions_np)
    
    return {
        'attributions': attributions_np.tolist(),
        'method': method,
        'human_explanation': explanation['text'],
        'summary': explanation['summary'],
        'confidence': explanation['confidence']
    }

def generate_human_explanation(attributions):
    """
    Generate human-readable explanation from attribution scores.
    """
    avg_importance = np.mean(np.abs(attributions))
    max_importance = np.max(attributions)
    min_importance = np.min(attributions)
    std_importance = np.std(attributions)
    
    # Determine pattern type
    if max_importance > 0.8:
        pattern = "The model identified specific regions in the data that strongly influenced its prediction. This suggests localized weather patterns were most important."
    elif avg_importance > 0.4:
        pattern = "The model used a combination of features across the region. Multiple areas contributed to the prediction."
    elif avg_importance > 0.15:
        pattern = "The model showed relatively balanced importance across the region. The prediction was influenced by widespread patterns rather than specific hotspots."
    else:
        pattern = "The model found this input similar to patterns in its training data. No single feature dominated the prediction."
    
    # Confidence level
    if std_importance < 0.1:
        confidence = "HIGH"
        confidence_text = "The model is very confident about this prediction. The attribution patterns are consistent and clear."
    elif std_importance < 0.3:
        confidence = "MEDIUM"
        confidence_text = "The model is moderately confident. Some uncertainty exists in the feature importance."
    else:
        confidence = "LOW"
        confidence_text = "The model shows lower confidence. The prediction relies on diverse features with varying importance."
    
    # Generate complete explanation
    full_explanation = f"""
🔍 **How the model made this prediction:**

{pattern}

📊 **Key Insights:**
- The average feature importance was {avg_importance:.2f} out of 1.0
- The most important feature scored {max_importance:.2f}, indicating significant influence
- The least important feature scored {min_importance:.2f}

🎯 **Confidence Level: {confidence}**
{confidence_text}

💡 **What this means:**
{generate_meaning(avg_importance, max_importance)}
"""
    
    return {
        'text': full_explanation.strip(),
        'summary': {
            'avg_importance': float(avg_importance),
            'max_importance': float(max_importance),
            'min_importance': float(min_importance),
            'std_importance': float(std_importance),
            'pattern': pattern,
            'confidence': confidence
        },
        'confidence': confidence
    }

def generate_meaning(avg_importance, max_importance):
    """
    Generate practical meaning based on importance scores.
    """
    if avg_importance > 0.5:
        return "The model found strong, clear patterns in the data. This prediction is reliable for decision-making."
    elif avg_importance > 0.3:
        return "The model identified meaningful patterns. This prediction can be used with moderate confidence."
    elif avg_importance > 0.15:
        return "The model found weak but noticeable patterns. Additional data would improve confidence."
    else:
        return "The model did not find strong patterns. Consider collecting more data or using different features."

def feature_importance_summary(attributions, top_k=5):
    """
    Get summary of top features with natural language.
    """
    flat = attributions.flatten()
    top_indices = np.argsort(np.abs(flat))[-top_k:][::-1]
    top_values = flat[top_indices]
    
    # Generate natural language description
    description = f"The top {top_k} most important features are:\n"
    for i, (idx, val) in enumerate(zip(top_indices, top_values)):
        description += f"  {i+1}. Feature with importance score {val:.3f}\n"
    
    return {
        'top_feature_indices': top_indices.tolist(),
        'top_feature_values': top_values.tolist(),
        'description': description
    }

def model_prediction_with_explanation(model, input_tensor, method='integrated_gradients'):
    """
    Get prediction along with human-readable explanation.
    """
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy()
    
    explanation = explain_prediction(model, input_tensor, method)
    feature_summary = feature_importance_summary(np.array(explanation['attributions']))
    
    # Interpret the prediction value
    pred_value = float(prediction[0][0])
    if pred_value > 0.5:
        pred_meaning = "higher than normal rainfall expected"
    elif pred_value > 0:
        pred_meaning = "slightly above normal rainfall expected"
    elif pred_value > -0.5:
        pred_meaning = "slightly below normal rainfall expected"
    else:
        pred_meaning = "significantly below normal rainfall expected"
    
    return {
        'prediction': pred_value,
        'prediction_meaning': pred_meaning,
        'explanation': explanation,
        'feature_summary': feature_summary
    }