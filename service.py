import dlib
from imutils import face_utils
import matplotlib.image as mpimg
import cv2
import io
import matplotlib.patheffects as PathEffects

from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class face_analyser:

    def __init__(self, uploaded_file):

        shape = self.mock_model_predict(uploaded_file)

        #top_head = (3*shape[74]+shape[73])/4
        top_head = (shape[72] + shape[76])/2
        bottom_head = (shape[7] + shape[8] + shape[9])/3

        top_nose = shape[27]
        bottom_nose = shape[33]

        top_lips = (shape[50]+shape[52])/2
        bottom_lips = shape[57]

        left_cheek = (shape[0] + shape[1])/2
        right_cheek = (shape[15] + shape[16])/2

        left_eye_center = (shape[37] + shape[38] + shape[40] + shape[41])/4
        right_eye_center = (shape[43] + shape[44] + shape[46] + shape[47])/4
        middle_eye = (left_eye_center + right_eye_center)/2

        right_eyebrow_inner_corner = shape[22]
        right_eyebrow_peak = (shape[25] + shape[26])/2

        right_lateral_canthus = shape[45]
        right_medial_canthus = shape[42]

        right_upper_eyelid = (shape[43] + shape[44])/2
        right_lower_eyelid = (shape[47] + shape[46])/2

        left_nose = shape[31]
        right_nose = shape[35]

        # left_mouth = shape[48]
        # right_mouth = shape[54]

        left_mouth = shape[60]
        right_mouth = shape[64]

        middle_brown = (shape[21] + shape[22])/2

        philtrum_lenght = self.distance(bottom_nose, top_lips)
        chin_lenght = self.distance(bottom_lips, bottom_head)

        eye_distance = self.distance(left_eye_center, right_eye_center)
        eye_height = self.distance(right_upper_eyelid, right_lower_eyelid)
        eye_width = self.distance(right_lateral_canthus, right_medial_canthus)

        bizygomatic_distance = self.distance(left_cheek, right_cheek)


        left_top_jaw = (shape[4] + shape[5])/2
        left_bottom_jaw = shape[7]
        right_top_jaw = (shape[11] + shape[12])/2
        right_bottom_jaw = shape[9]

        left_gonion = (shape[3] + shape[4])/2
        right_gonion = (shape[12] + shape[13])/2

        upper_facial_distance = self.distance(middle_brown, top_lips)
        midface_distance = self.distance(middle_eye, top_lips)

        nose_width = self.distance(left_nose, right_nose)
        mouth_width = self.distance(left_mouth, right_mouth)

        jaw_width = self.distance(left_gonion, right_gonion)



        # Operator:
        # 0 : division
        # 1 : degree
        # 2 : equal third
        self.ratio_data = {



            "chin_philtrum": {
                "1": chin_lenght,
                "2": philtrum_lenght,
                "operator": "0",
                "ideal":2.5,

                "couple_1": (bottom_nose, top_lips),
                "couple_2": (bottom_lips, bottom_head)

            },



            "interpupillary_distance": {
                "1": eye_distance,
                "2": bizygomatic_distance,
                "operator": "0",
                "ideal":0.5,

                "couple_1": (left_eye_center, right_eye_center),
                "couple_2": (left_cheek, right_cheek)
            },



            "width-to-height": {
                "1": bizygomatic_distance,
                "2": upper_facial_distance,
                "operator": "0",
                "ideal":1.82,

                "couple_1": (left_cheek, right_cheek),
                "couple_2": (middle_brown, top_lips)

            },



            "midface": {
                "1": midface_distance,
                "2": eye_distance,
                "operator": "0",
                "ideal":1,

                "couple_1": (left_eye_center, right_eye_center),
                "couple_2": (top_nose, top_lips)

            },



            "mouth_to_nose": {
                "1": mouth_width,
                "2": nose_width,
                "operator": "0",
                "ideal":1.618,

                "couple_1": (left_mouth, right_mouth),
                "couple_2": (left_nose, right_nose)

            },



            "lower_thirds_width": {
                "1": jaw_width,
                "2": bizygomatic_distance,
                "operator": "0",
                "ideal":0.9,

                "couple_1": (left_gonion, right_gonion),
                "couple_2": (left_cheek, right_cheek)

            },



            "eye_seperating_ratio": {
                "1": eye_distance,
                "2": bizygomatic_distance,
                "operator": "0",
                "ideal":0.5,

                "couple_1": (left_eye_center, right_eye_center),
                "couple_2": (left_cheek, right_cheek)

            },



            "eyebrows_height": {
                "1": midface_distance,
                "2": upper_facial_distance,
                "operator": "0",
                "ideal":0.90,

                "couple_1": (middle_brown, top_lips),
                "couple_2": (top_nose, top_lips)

            },



            "eye_height_width": {
                "1": eye_width,
                "2": eye_height, 
                "operator": "0",
                "ideal":3,

                "couple_1": (right_lateral_canthus, right_medial_canthus),
                "couple_2": (right_upper_eyelid, right_lower_eyelid)

            },



            "eye_mouth_eye_angle": {
                "operator": "1",
                "ideal":50,

                "couple_1": (left_eye_center, top_lips),
                "couple_2": (right_eye_center, top_lips)

            },



            "jaw_frontal_angle": {
                "operator": "1",
                "ideal":90,

                "couple_1": (left_top_jaw, left_bottom_jaw),
                "couple_2": (right_top_jaw, right_bottom_jaw)

            },


            "canthal_tilt": {
                "operator": "1",
                "ideal":5,

                "couple_1": (left_cheek, right_cheek),
                "couple_2": (right_lateral_canthus, right_medial_canthus)

            },


            "eyebrow_tilt": {
                "operator": "1",
                "ideal":5,

                "couple_1": (left_cheek, right_cheek),
                "couple_2": (right_eyebrow_inner_corner, right_eyebrow_peak)

            },



            "cheekbones_height": {
                "operator": "2",
                "ideal":0.9,

                "couple_1": (middle_eye, top_lips),
                "couple_2": (left_cheek, right_cheek)

            },


            "facial_thirds": {
                "operator": "3",
                "ideal":0.33,

                "couple_1": (top_head, middle_brown),
                "couple_2": (middle_brown, bottom_nose),
                "couple_3": (bottom_nose, bottom_head)

            },


            "jaw_thirds": {
                "operator": "3",
                "ideal":0.33,

                "couple_1": (left_gonion, left_mouth),
                "couple_2": (left_mouth, right_mouth),
                "couple_3": (right_mouth, right_gonion)

            },
        }

        for data in self.ratio_data:
            data_dict = self.ratio_data[data]

            if (data_dict["operator"] != "3"):
                data_dict["result"] = self.operate_data(data_dict)
                data_dict["relative_deviation"] = abs(1-(data_dict["result"]/data_dict["ideal"]))
            else:
                data_dict["result"] = self.operate_data(data_dict)
                data_dict["relative_deviation"] = sum([abs(1-(result/data_dict["ideal"])) for result in data_dict["result"]])/3
                #data_dict["relative_deviation"] = abs(1-(data_dict["result"]/data_dict["ideal"]))

            data_dict["relative_deviation"] = round(data_dict["relative_deviation"], 2)

        
        self.ratio_data_labels = [label for label in self.ratio_data]
        self.sorted_ratio_data_labels = sorted(self.ratio_data, key=lambda label : self.ratio_data[label]["relative_deviation"])
        self.reverse_sorted_ratio_data_labels = reversed(self.sorted_ratio_data_labels)
        



    
    def show_detail_view(self, pil_image, ratio_data, label, result):
        fig, ax = plt.subplots()
        ax.imshow(pil_image)
        ax.axis("off")

        x_y = lambda couple_n : (np.array(ratio_data[couple_n])[:, 0], np.array(ratio_data[couple_n])[:, 1])
        couple_1 = x_y("couple_1")
        couple_2 = x_y("couple_2")

        if (ratio_data["operator"] != "3"):


            text_coord = [couple_1[0][0]+100, couple_1[0][1]+100]
            print("----------", couple_1, "-------------")

            ax.plot(couple_1[0], couple_1[1], linestyle="dashed")
            ax.plot(couple_2[0], couple_2[1], linestyle="dashed")
            ax.text(text_coord[0], text_coord[1], f"{label} : {result} \n Ideal range : {ratio_data['ideal']}", fontsize=6).set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
        
        else:
            couple_3 = x_y("couple_3")

            ax.plot(couple_1[0], couple_1[1])
            ax.text(couple_1[0][0], couple_1[1][0], f"{result[0]}", fontsize=6).set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

            ax.plot(couple_2[0], couple_2[1])
            ax.text(couple_2[0][0]+30, couple_2[1][0]+30, f"{result[1]}", fontsize=6).set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

            ax.plot(couple_3[0], couple_3[1])
            ax.text(couple_3[0][1], couple_3[1][1], f"{result[2]}", fontsize=6).set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

        return fig, ax

    def operate_data(self, data):
        if (data["operator"] == "0"):
            return round(data["1"]/data["2"], 2)
        
        elif (data["operator"] == "1"):

            epsilon = 1e-6
            coeff_1 = (data["couple_1"][0][1] - data["couple_1"][1][1])/(data["couple_1"][0][0] - data["couple_1"][1][0] + epsilon)
            coeff_2 = (data["couple_2"][0][1] - data["couple_2"][1][1])/(data["couple_2"][0][0] - data["couple_2"][1][0] + epsilon)

            diff = (np.arctan(coeff_1) - np.arctan(coeff_2))*180/np.pi
            ideal = data["ideal"]

            print("-----\n")
            print((180 - diff), diff)
            print("\n-----")

            if (abs(ideal - (180 - diff)) > abs(ideal-diff)):
                return round(diff, 2)
            else:
                return round(180 - diff, 2)
        
        elif (data["operator"] == "2"):
            x1, y1 = data["couple_1"][0]
            x2, y2 = data["couple_1"][1]
            x3, y3 = data["couple_2"][0]
            x4, y4 = data["couple_2"][1]

            denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)

            det1 = x1*y2 - y1*x2
            det2 = x3*y4 - y3*x4

            x = (det1*(x3 - x4) - (x1 - x2)*det2) / denom
            y = (det1*(y3 - y4) - (y1 - y2)*det2) / denom

            return (round(self.distance((x, y), (x2, y2))/self.distance((x1, y1), (x2, y2)), 2))
        
        elif (data["operator"] == "3"):

            total_distance = self.distance(data["couple_1"][0], data["couple_3"][1])

            third_one = float(round(self.distance(data["couple_1"][0], data["couple_1"][1])/total_distance, 2))
            third_two = float(round(self.distance(data["couple_2"][0], data["couple_2"][1])/total_distance, 2))
            third_three = float(round(self.distance(data["couple_3"][0], data["couple_3"][1])/total_distance, 2))

            return (third_one, third_two, third_three)


    def mock_model_predict(self, pil_image):

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

        # Convertit en tableau NumPy (H x W x 3) au format RGB
        image_np = np.array(pil_image.convert("RGB"))

        # Optionnel : convertir en BGR pour OpenCV
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        rects = detector(image_bgr, 1)

        r = rects[0]
        x1, y1, x2, y2 = r.top(), r.left(), r.bottom(), r.right()

        shape = predictor(image_bgr, r)
        shape = face_utils.shape_to_np(shape)
        X = shape[:, 0]
        Y = shape[:, 1]

        def permute(array:list, permutations:list[tuple]):
            copy1 = array.copy()
            copy2 = array.copy()

            for permutation in permutations:
                copy1[permutation[0]] = copy2[permutation[1]]
                copy1[permutation[1]] = copy2[permutation[0]]

                copy2 = copy1.copy()

            return copy1

        permuted_shape = permute(shape, [(68, 77), (69, 75), (70, 76), (71, 77), (72, 75), (73, 76), (74, 77), (75, 80), (76, 80), (77, 80), (78, 79), (79, 80)])

        return permuted_shape

    def distance(self, point_A, point_B):
        return (((point_B[0]-point_A[0])**2) + (point_B[1]-point_A[1])**2)**0.5

    def generate_pdf_report(self, pil_image, labels_order):
        # Crée un buffer mémoire pour le PDF
        buffer = io.BytesIO()

        # Ouvre un document PDF "virtuel"
        with PdfPages(buffer) as pdf:
            for label in labels_order:
                ratio_data = self.ratio_data[label]
                result = self.operate_data(ratio_data)

                # Génère la figure
                fig, ax = self.show_detail_view(pil_image, ratio_data, label, result)

                ax.set_title(" ".join(label.split("_")))
                
                # Ajoute le texte en-dessous ou au-dessus
                ax.text(0, 120, f"Result : {result}", fontsize=10).set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
                ax.text(0, 240, f"Ideal : {ratio_data['ideal']}", fontsize=10).set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
                ax.text(0, 360, f"Standard deviation : {ratio_data['relative_deviation']}", fontsize=10).set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

                # Tu peux ajouter aussi scatter / annotate comme avant

                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        buffer.seek(0)
        return buffer