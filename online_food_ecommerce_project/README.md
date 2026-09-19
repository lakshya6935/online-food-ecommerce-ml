# Online Food / E-Commerce Delivery ML Project

## Dataset
100,000 rows were supplied in `ecommerce_delivery_analytics.csv`.
Duplicate rows were removed before training.

Original rows: **100,000**
Duplicate rows removed: **0**
Final rows: **100,000**

## Two predictions

### 1. Delivery Delay
Predicts whether an order is delayed from delivery time.

Test accuracy: **100%**

### 2. Service Rating
Predicts the 1–5 service rating from the customer's feedback text.

Test accuracy: **100%**

## Important accuracy note
The 100% results are genuine test-set accuracy for this dataset, but the targets contain deterministic relationships:
- `Delivery Delay` changes at the dataset's delivery-time threshold.
- Each `Customer Feedback` value maps to one `Service Rating`.

Therefore, the high accuracy reflects the structure of this particular dataset and should not be presented as proof that the model will achieve 100% on new real-world data.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub
Upload all files to a GitHub repository. For a live Streamlit app, connect that repository to Streamlit Community Cloud.
