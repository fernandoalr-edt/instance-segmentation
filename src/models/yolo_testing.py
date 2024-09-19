import time
import concurrent
import cv2

import numpy as np

from ultralytics import YOLO
from glob import glob

def predict(model_path, images_paths):

	model = YOLO(model_path)
	results = []

	for image_path in images_paths:
		results.append(model.predict(image_path))

	return results


def paralelo():

	images_paths = glob("data/test/*.jpg")
	models_paths = ["yolov8n-seg.pt"]
	index = 0

	with concurrent.futures.ThreadPoolExecutor() as executor:

		future_to_model = {executor.submit(predict, model_path, images_paths): model_path for model_path in models_paths}

		for future in concurrent.futures.as_completed(future_to_model):

			model_path = future_to_model[future]
			try:
				results = future.result()
				for result in results:
					data = result[0].masks.data.cpu().numpy()
					classes_det = result[0].boxes.cls.cpu().numpy()

					for object_det, class_det in zip(data, classes_det):

						mask = np.where(object_det == 1, class_det, 0)
						cv2.imwrite(f"{index}.png", mask)
						index += 1

			except Exception as exc:
				print(exc)


def lineal():

	images_paths = glob("data/test/*.jpg")[:100]
	models_paths = ["yolov8n-seg.pt"]*15

	results = {}
	for model_path in models_paths:
		results[model_path] = predict(model_path, images_paths)

	print(results)


if __name__=="__main__":

	paralelo()

