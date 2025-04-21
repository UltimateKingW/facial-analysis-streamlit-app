

import streamlit as st

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from service import face_analyser

st.title("📍 Facial analysis")

url = "https://www.youtube.com/@thelooksmaxxer"
st.markdown("# [ The looksmaxxer youtube channel ](%s)" % url)
url2 = "https://github.com/Thomcle"
st.markdown("# This application was designed and created by [Thomcle](%s)" % url2)

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
        ("Original Order", "Ascending", "Descending"),
    )


    if sort_order == "Original Order":
        labels_order = analyser.ratio_data_labels
    elif sort_order == "Ascending":
        labels_order = analyser.sorted_ratio_data_labels
    elif sort_order == "Descending":
        labels_order = analyser.reverse_sorted_ratio_data_labels

    st.markdown("""
    ### 📊 About Sorting by Relative Deviation

    Sorting is based on the **relative deviation**:

    - A value of `0.0` means the attribute is **ideal** ✅.
    - The **higher** the value, the more it **deviates** from the ideal.
    - From a value of **0.2** upwards, the deviation can generally be considered a **noticeable imperfection** ⚠️.

    ---

    #### 🔼 Ascending Order
    - Starts with the **lowest** relative deviation (i.e. from the **best** attribute to the **worst**).
    - Useful to highlight what is already good.

    #### 🔽 Descending Order
    - Reverses the same logic: from **worst** to **best**.
    - Useful to **prioritize the biggest issues** first.
                

    ---
    """)

    for label in labels_order:

        ratio_data = analyser.ratio_data[label]
        result = analyser.operate_data(ratio_data)

        # st.write(f"{label} : {result},")
        # st.write(f"Ideal range : {ratio_data['ideal']}")

        st.markdown(f"""
        ### 📌 Result for **{label}**

        - ✅ **Calculated result** : `{ratio_data['result']}`
        - 🎯 **Ideal range** : `{ratio_data['ideal']}`
        - 📐 **Relative deviation** : `{ratio_data['relative_deviation']}`
        """, )

        #st.metric(label="Result", value=f"{result:.2f}", border=True)

        with st.expander(f"📌 show details"):

            fig, _ = analyser.show_detail_view(pil_image, ratio_data, label, result)

            st.pyplot(fig)


    if st.button("📄 Download PDF report", key="a"):
        pdf_file = analyser.generate_pdf_report(pil_image, labels_order)
        st.download_button(
            label="📥 Click to download PDF",
            data=pdf_file,
            file_name="rapport_analyse.pdf",
            mime="application/pdf",
        )
