

import streamlit as st

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from service import face_analyser

st.title("📍 Facial analysis")

# Upload de l'image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Chargement de l'image avec PIL
    pil_image = Image.open(uploaded_file)

    
    st.image(pil_image, caption="🖼️ Input image", use_container_width=True)


    # Appel de ton modèle pour obtenir les coordonnées
    analyser = face_analyser(pil_image)


    sort_order = st.radio(
        "🔢 Select the display order of results:",
        ("Original Order", "Ascending", "Descending")
    )

    if sort_order == "Original Order":
        labels = analyser.ratio_data_labels
    elif sort_order == "Ascending":
        labels = analyser.sorted_ratio_data_labels
    elif sort_order == "Descending":
        labels = analyser.reverse_sorted_ratio_data_labels

    print(analyser.reverse_sorted_ratio_data_labels)
    print(analyser.sorted_ratio_data_labels)

    for label in labels:

        ratio_data = analyser.ratio_data[label]
        result = analyser.operate_data(ratio_data)

        # st.write(f"{label} : {result},")
        # st.write(f"Ideal range : {ratio_data['ideal']}")

        st.markdown(f"""
        ### 📌 Result for **{label}**

        - ✅ **Calculated result** : `{result}`
        - 🎯 **Ideal range** : `{ratio_data['ideal']}`
        """, )

        #st.metric(label="Result", value=f"{result:.2f}", border=True)

        with st.expander(f"📌 show details"):

            fig, _ = analyser.show_detail_view(pil_image, ratio_data, label, result)

            st.pyplot(fig)


    if st.button("📄 Download PDF report", key="a"):
        pdf_file = analyser.generate_pdf_report(pil_image)
        st.download_button(
            label="📥 Click to download PDF",
            data=pdf_file,
            file_name="rapport_analyse.pdf",
            mime="application/pdf",
        )
