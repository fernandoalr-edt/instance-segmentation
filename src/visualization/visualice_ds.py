from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
import cv2
import random
import matplotlib.pyplot as plt

class DemoTest():

	def __init__(self, dataset_names_list):
		""" A class to run demo to check the converted COCO format
		Args:
			datasetDir (str): path to the ADE20K dataset directory
		"""
		self.dataset_names_list = dataset_names_list

	def startDemo(self):

		for dataset_name in self.dataset_names_list:
			print(dataset_name)

			dataset = DatasetCatalog.get(dataset_name)

			for data in random.sample(dataset, 25):

				file_name = data["file_name"]
				img = cv2.imread(file_name)
				visualizer = Visualizer(img[:, :, ::-1],
										metadata=MetadataCatalog.get(dataset_name))
				out = visualizer.draw_dataset_dict(data)
				plt.title(file_name.split('/')[-1])
				plt.imshow(
					cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
				plt.show()


register_coco_instances("ds",
					    {},
						"/media/fernando/DATA/DATASETS/DATASET_SEGMENTACION/Nature/Antiguo/CoCo_Annotations/coco.json",
						"/media/fernando/DATA/DATASETS/DATASET_SEGMENTACION/Nature/Antiguo/Images")

show_dataset = DemoTest(["ds"])
show_dataset.startDemo()