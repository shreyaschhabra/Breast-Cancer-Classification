#!/usr/bin/env python3
"""
Breast Cancer Classification Model Deployment Script

This script provides a simple deployment interface for the trained breast cancer
classification model, allowing for real-time predictions on new patient data.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from typing import Union, Dict, List, Any
import argparse

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class BreastCancerPredictor:
    """
    A class for loading and using trained breast cancer classification models.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path (str): Path to the saved model file
        """
        try:
            self.model = joblib.load(model_path)
            self.feature_names = [
                'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                'smoothness_mean', 'compactness_mean', 'concavity_mean',
                'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                'smoothness_se', 'compactness_se', 'concavity_se',
                'concave points_se', 'symmetry_se', 'fractal_dimension_se',
                'radius_worst', 'texture_worst', 'perimeter_worst',
                'area_worst', 'smoothness_worst', 'compactness_worst',
                'concavity_worst', 'concave points_worst', 'symmetry_worst',
                'fractal_dimension_worst'
            ]
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict(self, patient_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame, np.ndarray]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Make predictions on patient data.
        
        Args:
            patient_data: Patient features as dictionary, list of dictionaries, or DataFrame
            
        Returns:
            Dict: Prediction results with probabilities and class
        """
        try:
            # Convert input to DataFrame
            if isinstance(patient_data, dict):
                df = pd.DataFrame([patient_data])
            elif isinstance(patient_data, list):
                df = pd.DataFrame(patient_data)
            elif isinstance(patient_data, np.ndarray):
                df = pd.DataFrame(patient_data, columns=self.feature_names)
            elif isinstance(patient_gata, pd.DataFrame):
                df = patient_data.copy()
            else:
                raise TypeError(f"Unsupported input type: {type(patient_data)}")
            
            # Ensure all required features are present
            missing_features = set(self.feature_names) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Reorder columns to match training data
            df = df[self.feature_names]
            
            # Make predictions
            predictions = self.model.predict(df)

            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(df)
            else:
                raise AttributeError("Model does not have predict_proba method.")
            
            # Format results
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                result = {
                    'prediction': int(pred),
                    'prediction_label': 'Malignant' if pred == 1 else 'Benign',
                    'benign_probability': float(prob[0]),
                    'malignant_probability': float(prob[1]),
                    'confidence': float(max(prob)),
                    'risk_level': self._assess_risk_level(float(prob[1]))
                }
                results.append(result)
            
            return results[0] if len(results) == 1 else results
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            raise
    
    def _assess_risk_level(self, malignant_prob: float) -> str:
        """
        Assess risk level based on malignant probability.
        
        Args:
            malignant_prob (float): Probability of malignancy
            
        Returns:
            str: Risk level classification
        """
        if malignant_prob < 0.3:
            return "Low Risk"
        elif malignant_prob < 0.6:
            return "Moderate Risk"
        elif malignant_prob < 0.8:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def batch_predict(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Make predictions on a batch of patients from a CSV file.
        
        Args:
            csv_path (str): Path to input CSV file
            output_path (str): Path to save results (optional)
            
        Returns:
            pd.DataFrame: Results with original data and predictions
        """
        try:
            # Load data
            df = pd.read_csv(csv_path)
            print(f"📁 Loaded {len(df)} patient records from {csv_path}")
            
            # Make predictions
            results = self.predict(df)
            
            # Add predictions to original data
            if isinstance(results, list):
                df = df.copy()
                df['prediction'] = [r['prediction_label'] for r in results]
                df['malignant_probability'] = [r['malignant_probability'] for r in results]
                df['risk_level'] = [r['risk_level'] for r in results]
            else:
                df = df.copy()
                df['prediction'] = results['prediction_label']
                df['malignant_probability'] = results['malignant_probability']
                df['risk_level'] = results['risk_level']
            
            # Save results if output path provided
            if output_path:
                df.to_csv(output_path, index=False)
                print(f"💾 Results saved to {output_path}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error during batch prediction: {e}")
            raise


def main():
    """
    Main function for command-line interface.
    """
    parser = argparse.ArgumentParser(description='Breast Cancer Classification Predictor')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--input', type=str,
                       help='Input CSV file for batch prediction')
    parser.add_argument('--output', type=str,
                       help='Output CSV file for batch prediction results')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = BreastCancerPredictor(args.model)
    
    if args.interactive:
        print("\n🏥 Breast Cancer Classification - Interactive Mode")
        print("Enter patient data (type 'quit' to exit):")
        print("=" * 50)
        
        while True:
            try:
                print("\nEnter patient features (comma-separated values):")
                print("Order: radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave points_worst, symmetry_worst, fractal_dimension_worst")
                
                user_input = input("> ").strip()
                if user_input.lower() == 'quit':
                    break
                
                # Parse input
                values = [float(x.strip()) for x in user_input.split(',')]
                if len(values) != 30:
                    print("❌ Error: Please enter exactly 30 feature values")
                    continue
                
                # Create patient data dictionary
                patient_data = dict(zip(predictor.feature_names, values))
                
                # Make prediction
                result = predictor.predict(patient_data)
                
                # Display results
                print(f"\n🔍 Prediction Results:")
                print(f"   Diagnosis: {result['prediction_label']}")
                print(f"   Confidence: {result['confidence']:.1%}")
                print(f"   Malignant Probability: {result['malignant_probability']:.1%}")
                print(f"   Risk Level: {result['risk_level']}")
                print("=" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
    
    elif args.input:
        # Batch prediction mode
        results = predictor.batch_predict(args.input, args.output)
        print(f"\n✅ Batch prediction completed for {len(results)} patients")
        
        # Display summary statistics
        if 'prediction' in results.columns:
            prediction_counts = results['prediction'].value_counts()
            print(f"\n📊 Prediction Summary:")
            for pred_type, count in prediction_counts.items():
                print(f"   {pred_type}: {count} cases")
    
    else:
        print("❌ Please specify either --interactive or --input mode")


if __name__ == "__main__":
    main()
