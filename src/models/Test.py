import os
import comet_ml

import numpy as np
from natsort import natsorted

from ultralytics import YOLO
from src.data.json_service import write_json
from src.metrics.confusion_matrix import conf_matrix_calc
from src.data.dataset_services import convert_coco, generate_txt_files

class Prediction_CM():
	"""docstring for ClassName"""
	def __init__(self, pred_classes, pred_bboxes, true_classes, true_bboxes):
		self.pred_classes = pred_classes
		self.pred_bboxes = pred_bboxes
		self.true_classes = true_classes
		self.true_bboxes = true_bboxes


def load_images(txt_path):

	lines = []
	with open(txt_path) as f:
		while True:
			line = f.readline()

			if line:
				lines.append(line.strip())
			else:
				f.close()
				break

	return lines


def get_true_data(txt_label_path, image_shape):

	classes = []
	bboxes = []

	with open(txt_label_path) as f:

		while True:
			line = f.readline().strip().split()
			
			classes.append(int(line[0]))

			points = line[1:]
			contour = np.array([[float(x),float(y)] for x, y in zip(points[::2],points[1::2])])

			bbox = np.array([int(np.min(contour[:,0])*image_shape[0]), int(np.min(contour[:,1])*image_shape[1]),
							 int(np.max(contour[:,0])*image_shape[0]), int(np.max(contour[:,1])*image_shape[1])])
			
			bboxes.append(bbox)

			break

	return np.array(classes), np.array(bboxes)


def test(data_dir, category):

	# generar las etiquetas nuevas de las an onotaciones propias (con el coco_convert normal)
	convert_coco(labels_dir=f"{data_dir}/{category}/Propio/CoCo_Annotations",
				 images_dir=f"{data_dir}/{category}/Propio/Images",
				 use_segments=True,
				 cls91to80=False)

	# hacer lo mismo para el de ADE20K (este hay que hacerlo siempre para modificar las anotaciones)
	convert_coco(labels_dir=f"{data_dir}/{category}/ADE20K",
				 images_dir=f"{data_dir}/ADE20K",
				 use_segments=True,
				 cls91to80=False)

	yaml_path = f"configurations/{category}.yaml"

	model = YOLO(f"Projects/{category}_project/train/weights/best.pt")

	classes = list(model.names.values())

	images_list = load_images(os.path.join(data_dir, category, "test.txt"))
	predictions_list = []

	index = 0
	for image in images_list:

		result = model.predict(image, conf=0.7)

		img_plotted = result[0].plot()
		import cv2
		cv2.imwrite(f"output/{index}.png", img_plotted)
		index += 1

		if not result[0]:
			continue
		
		pred_classes = result[0].boxes.cls.cpu().numpy().astype(int)
		pred_bboxes = result[0].boxes.xyxy.cpu().numpy().astype(int)
		image_shape = result[0].masks.orig_shape

		txt_label_path = image.split('.')[0] + ".txt"
		true_classes, true_bboxes = get_true_data(txt_label_path, image_shape)

		predictions_list.append(Prediction_CM(pred_classes, pred_bboxes, true_classes, true_bboxes))

	metrics = conf_matrix_calc(predictions_list, classes)
	dict_metrics = {'precision':{},'recall':{}}

	for index, class_name in enumerate(classes):
		TP = metrics[class_name]["TP"]
		FN = metrics[class_name]["FN"]
		FP = metrics[class_name]["FP"]

		if TP == 0:
			dict_metrics['recall'][class_name] = 0
			dict_metrics['precision'][class_name] = 0

		else:
			dict_metrics['recall'][class_name] = round(TP/(TP+FP),2)
			dict_metrics['precision'][class_name] = round(TP/(TP+FN),2)

	metrics_save_path = os.path.join(f"Projects/{category}_project", "train", "calculated_metrics.json")
	write_json(dict_metrics, metrics_save_path)


if __name__=="__main__":

	data_dir = "/media/fernando/DATA/DATASETS/DATASET_SEGMENTACION"
	categories = [category for category in os.listdir(data_dir) if category != "ADE20K"]

	for category in categories:
		try:
			test(data_dir, category)

		except Exception as e:
			print(e)