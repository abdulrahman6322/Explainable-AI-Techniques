# XAI Techniques for Income Prediction

This notebook explores the application of Explainable AI (XAI) techniques to interpret a Random Forest model trained to predict income levels using the UCI Adult Income dataset. 

## Key features:

- **Data Preprocessing:** The notebook includes data loading, cleaning, and preprocessing steps to prepare the Adult Income dataset for model training. This involves handling missing values, encoding categorical features, and scaling numerical features.
- **Model Training:** A Random Forest classifier is trained using a scikit-learn pipeline to predict income levels based on various demographic and socioeconomic factors.
- **XAI Explanations:** The notebook demonstrates the use of two popular XAI techniques:
    - **LIME (Local Interpretable Model-agnostic Explanations):** Provides local explanations for individual predictions, highlighting the most influential features.
    - **SHAP (SHapley Additive exPlanations):** Offers both global and local interpretations, including feature importance rankings and individual prediction explanations.
- **Comparison of XAI Techniques:** The advantages and disadvantages of LIME and SHAP are discussed in terms of model interpretability, considering factors like model compatibility, explanation scope, theoretical foundation, computational cost, and stability. A comparison table is included to summarize these differences.

## Table Comparing LIME & SHAP:

| Feature | LIME | SHAP |
|---|---|---|
| **Model Compatibility** | Model-agnostic | Model-specific and model-agnostic implementations available |
| **Explanation Scope** | Local | Global and Local |
| **Interpretability** | Simple linear models, easy to understand | Based on game theory, can be more complex |
| **Theoretical Foundation** | Less rigorous | Strong theoretical foundation based on Shapley values |
| **Computational Cost** | Relatively low | Can be high, especially for complex models |
| **Stability** | Can be unstable for similar instances | More stable due to theoretical foundation |

## Dependencies:

- numpy
- pandas
- scikit-learn
- matplotlib
- lime
- shap
