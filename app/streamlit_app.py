"""Optional enhancement: interactive smartphone price estimator.

Run from the project root:
    pip install streamlit
    streamlit run app/streamlit_app.py

The app loads models/final_model.joblib — the exact pipeline produced by `python -m src.train`
(feature engineering + encoding + model) — and calls src.predict, so its estimates are identical
to the notebooks' and the training script's.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import load_metadata, predict_price  # noqa: E402

st.set_page_config(page_title="Smartphone Price Estimator", page_icon="📱", layout="centered")

meta = load_metadata()
ranges = meta["training_ranges"]
test = meta["holdout_test_metrics"]

st.title("📱 Smartphone Price Estimator")
st.caption("Preliminary price estimate from specifications · trained on Flipkart India smartphone listings")

with st.sidebar:
    st.header("About the model")
    st.write(f"**Model:** {meta['model']}")
    st.write(f"**Trained on:** {meta['rows_deploy_fit']} smartphone listings")
    st.write("**Accuracy on unseen phone models (hold-out test):**")
    st.write(f"- Typical error (MAE): ₹{test['MAE']:,.0f}")
    st.write(f"- R²: {test['R2']:.2f} · MAPE: {test['MAPE_%']:.1f} %")
    st.write("**Most important inputs:** processor, brand / model name, RAM, front camera, battery, storage.")
    st.info("Most reliable below ~₹20k. Premium phones (≥ ₹30k) are usually under-estimated.")

st.subheader("Specifications")
c1, c2 = st.columns(2)
with c1:
    brand = st.selectbox("Brand", sorted(set(meta["brands_seen"]) | {"Other"}), index=sorted(set(meta["brands_seen"]) | {"Other"}).index("Samsung") if "Samsung" in meta["brands_seen"] else 0)
    model_name = st.text_input("Model name (optional)", placeholder="e.g. Galaxy M34 5G",
                               help="Used only to detect '5G' and Pro/Plus/Ultra/Max suffixes.")
    processors = meta["known_processors"]
    processor = st.selectbox("Processor", processors + ["Other / not listed"],
                             index=processors.index("Qualcomm Snapdragon 695") if "Qualcomm Snapdragon 695" in processors else 0)
    if processor == "Other / not listed":
        processor = st.text_input("Type the processor name", placeholder="e.g. Snapdragon 7 Gen 1") or "Unknown"
    storage = st.selectbox("Storage (GB)", [16, 32, 64, 128, 256, 512], index=3)
    ram = st.selectbox("RAM (GB)", [2, 3, 4, 6, 8, 12, 16], index=4)
with c2:
    battery = st.number_input("Battery (mAh)", min_value=500, max_value=10000, value=5000, step=100)
    rear = st.number_input("Main rear camera (MP)", min_value=0, max_value=300, value=50, step=1)
    front = st.number_input("Front camera (MP)", min_value=0, max_value=100, value=13, step=1)
    display = st.number_input("Display size (cm, diagonal)", min_value=5.0, max_value=25.0, value=16.7, step=0.1,
                              help="Flipkart lists e.g. '16.76 cm (6.6 inch)'. 1 inch = 2.54 cm.")
    ai_lens = st.checkbox("Listing mentions an AI lens", value=False)

spec = {
    "model": f"{brand if brand != 'Other' else 'Unknown'} {model_name}".strip(),
    "processor": processor,
    "storage_gb": storage, "ram_gb": ram, "battery_mah": battery,
    "rear_camera_mp": rear, "front_camera_mp": front, "ai_lens": int(ai_lens), "display_size_cm": display,
}

if st.button("Estimate price", type="primary", use_container_width=True):
    try:
        result = predict_price(spec)
    except ValueError as err:
        st.error(f"Please check the inputs: {err}")
    else:
        price = result["predicted_price"]
        mape = test["MAPE_%"] / 100
        st.metric("Estimated price", f"₹{price:,.0f}")
        st.write(f"Typical range for phones like this: **₹{price * (1 - mape):,.0f} – ₹{price * (1 + mape):,.0f}** "
                 f"(± the model's average percentage error on unseen phones).")
        for w in result["warnings"]:
            st.warning(w)
        if price >= 30000:
            st.warning("Premium segment: the model tends to under-estimate phones above ₹30k. Use with care.")
        with st.expander("Inputs sent to the model"):
            st.dataframe(pd.DataFrame([spec]).T.rename(columns={0: "value"}))

st.divider()
st.caption(
    "**Disclaimer:** this is a statistical estimate based on historical listing prices, not a guaranteed market price. "
    "It reflects associations in the data (e.g. bigger batteries appear in cheaper phones), not causal effects of "
    "individual specifications. It does not know launch date, brand series, build quality or current promotions, and "
    "it does not cover feature phones."
)
