# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/datasets/prepare_ade20k_ins_seg.py
# ADE20K dataset: https://groups.csail.mit.edu/vision/datasets/ADE20K/
from pathlib import Path
import argparse
import json
import pickle
import numpy as np
from tqdm import tqdm
import pycocotools.mask as mask_util

# For demo
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2, random
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
from detectron2.data.datasets import register_coco_instances
from src.data.json_service import write_json, load_json


def pickleload(path):
	with open(path, "rb") as f:
		data = pickle.load(f)
	return data


def saveJson(data, path):
	with open(path, 'w') as f:
		json.dump(data, f)

def change_ids(data_dict, category):

	class_mapper = load_json('src/features/ADE20K_mapper.json')[category]
	categories = load_json("src/features/categories_ADE20K.json")[category]

	new_annotations = []
	for annotation in data_dict['annotations']:

		annotation_id = annotation['category_id']-1

		new_id = class_mapper[str(annotation_id)]['id']

		annotation['category_id'] = new_id

		annotation['segmentation'] = annotation['segmentation']

		new_annotations.append(annotation)

	data_dict['annotations'] = new_annotations

	data_dict['categories'] = categories

	return data_dict


class AdeToCOCO():
	""" A class to convert ADE20K to COCO format
	Attributes:
		statics (dict): ADE20K index pickle file data
		datasetDir (str): path to the ADE20K dataset directory
		objectNames (list): list of object names to convert
		annId (int): annotation id start from 1
	-----------------
	ADE20K index pickle data structure:
	N: number of images, C: number of object categories
	statics["filename"]: list of image file name with size N
	statics["folder"] : list of image folder name with size N
	statics["objectnames"]: list of object names with size C
	statics["objectPresence"]: list of object presence with size CxN, 
								objectPresence(c,i) = n means image i contains n objects of category c
	"""

	def __init__(self, pklPath, datasetDir, objectNames):
		"""
		Args:
			pklPath (str): path to the ADE20K index pickle file
			datasetDir (str): path to the ADE20K dataset directory
			objectNames (list): list of object names to convert
		"""
		self.statics = pickleload(pklPath)
		self.datasetDir = datasetDir
		self.objectNames = objectNames
		self.annId = 1


	def getObjectIdbyName(self, name):
		"""Get object id by object name
		
		Args:
			name (str): object name
		Returns:
			objId (int): object id
		"""
		objId = np.where(np.array(self.statics["objectnames"]) == name)[0][0]
		print(f"id of {name} is {objId}")
		return int(objId)


	def getImageIds(self, names):
		"""Get image ids by object names
		
		Args:
			names (list): list of object names
		Returns:
			imgIds (list): list of image ids
		"""
		all_image_ids = []

		for name in names:
			objId = self.getObjectIdbyName(name)
			current_image_ids = np.where(
				self.statics["objectPresence"][objId] > 0)[0]
			all_image_ids.append(current_image_ids)

		imgIds = np.unique(np.concatenate(all_image_ids))
		return imgIds.tolist()


	def getImagePathbyId(self, imageId):
		"""Get image path by image id
		
		Args:
			imageId (int): image id
		Returns:
			path (str): image path
		"""
		path = Path(self.datasetDir) / \
			self.statics["folder"][imageId] / self.statics["filename"][imageId]

		assert path.exists(), f"Image file not exist"
		return str(path)


	def getInfoJsonbyId(self, imageId):
		"""Get image information json file path by image id,
		Each image has a json file to store image information
		
		Args:
			imageId (int): image id
		Returns:
			path (str): image information json file path
		"""
		path = Path(self.datasetDir) / self.statics["folder"][imageId] / \
			self.statics["filename"][imageId].replace("jpg", "json")
		assert path.exists(), f"Image information json file not exist"
		return str(path)


	def generateAnnotations(self, imageId, imageInfo):
		""" Generate annotations for a single image in COCO format
		Args:
			imageId (int): image id
			imageInfo (dict): image information
		Returns:
			annotations (list): list of annotations
		"""
		objects = imageInfo["object"]

		annotations = []

		for obj in objects:

			if obj["name"] not in objectNames:
				continue

			annotation = {
				"id": int(self.annId),
				"image_id": int(imageId),
				"category_id": int(obj["name_ndx"]),
				"segmentation": [],
				"area": float,
				"bbox": [],
				"iscrowd": int(0)
			}

			# trans polygan to segmentation
			polygon = obj["polygon"]
			xmin, xmax = 1e8, -1e8
			ymin, ymax = 1e8, -1e8
			for x, y in zip(polygon['x'], polygon['y']):
				annotation["segmentation"].extend([x, y])
				xmin, xmax = min(xmin, x), max(xmax, x)
				ymin, ymax = min(ymin, y), max(ymax, y)

			# calculate bounding box
			annotation["bbox"] = [
				int(xmin),
				int(ymin),
				int(xmax - xmin + 1),
				int(ymax - ymin + 1)
			]
			# get rle (Run-Length Encoding)
			#rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
			h, w = imageInfo["imsize"][0], imageInfo["imsize"][1]
			poly = [annotation["segmentation"]]
			rle = mask_util.frPyObjects(poly, h, w)[0]

			rle["counts"] = rle["counts"].decode("utf-8")
			#annotation["segmentation"] = rle

			# get area
			annotation["area"] = int(mask_util.area(rle))

			annotation["segmentation"] = [annotation["segmentation"]]

			# print(annotation)
			annotations.append(annotation)
			self.annId += 1
		return annotations


	def generateImage(self, imageId, imagePath, imageInfo):
		""" Generate image information for a single image in COCO format
		Args:
			imageId (int): image id
			imagePath (str): image path
			imageInfo (dict): image information in ADE20K format
		Returns:
			image (dict): image information in COCO format
		"""
		image = {"id": int, "file_name": str, "width": int, "height": int}
		image["id"] = int(imageId)
		image["file_name"] = '/'.join(str(imagePath).split('/')[-4:])
		image["width"] = int(imageInfo["imsize"][1])
		image["height"] = int(imageInfo["imsize"][0])
		return image


	def convert(self, output_dir, name_file):
		# Convert Category
		adeCategories = []
		for name in self.objectNames:
			print(f"Convert {name}")
			categoryDict = {"id": int, "name": str}
			id = self.getObjectIdbyName(
				name) + 1  # consist with seg json name_ndx
			categoryDict["id"] = id
			categoryDict["name"] = name
			adeCategories.append(categoryDict)

		trainDict = {}
		valDict = {}

		trainImages = []
		trainCategory = adeCategories
		trainAnnotations = []

		valImages = []
		valCategory = adeCategories
		valAnnotations = []
		decodeFailCount = 0

		for imgId in tqdm(self.getImageIds(objectNames)):
			jsonFile = self.getInfoJsonbyId(imgId)

			# TODO: handle decode fail
			with open(jsonFile, 'r', encoding='utf-8') as f:
				try:
					imageInfo = json.load(f)['annotation']
				except:
					print(f"fail to decode {jsonFile}")
					decodeFailCount += 1
					continue

			imagePath = self.getImagePathbyId(imgId)
			# print(imagePath)
			image = self.generateImage(imgId, imagePath, imageInfo)
			annotations = self.generateAnnotations(imgId, imageInfo)
			if "ADE/training" in imagePath:
				trainImages.append(image)
				trainAnnotations.extend(annotations)
			elif "ADE/validation" in imagePath:
				valImages.append(image)
				valAnnotations.extend(annotations)
			else:
				print(f"{imagePath} is not in training or validation set")

		trainDict["images"] = trainImages
		trainDict["categories"] = trainCategory
		trainDict["annotations"] = trainAnnotations

		valDict["images"] = valImages
		valDict["categories"] = valCategory
		valDict["annotations"] = valAnnotations

		# print(trainAnnotations)
		trainOutputFilePath = Path(output_dir) / \
			f"ADE20K/train.json"
		valOutputFilePath = Path(output_dir) / \
			f"ADE20K/val.json"
		testOutputFilePath = Path(output_dir) / \
			f"ADE20K/test.json"
		change_ids(trainDict, name_file)
		saveJson(trainDict, trainOutputFilePath)
		change_ids(valDict, name_file)
		saveJson(valDict, valOutputFilePath)
		saveJson(valDict, testOutputFilePath)


class DemoTest():

	def __init__(self, datasetDir):
		""" A class to run demo to check the converted COCO format
		Args:
			datasetDir (str): path to the ADE20K dataset directory
		"""
		self.datasetDir = datasetDir

	def startDemo(self,k,name_file):
		trainJsonFilePath = Path(datasetDir) / \
			f"{name_file}/ADE20K/train.json"
		register_coco_instances(name_file, {}, trainJsonFilePath, f"{datasetDir}/ADE20K")
		dataset = DatasetCatalog.get(name_file)
		for data in random.sample(dataset, k):
			fileName = data["file_name"]
			img = cv2.imread(fileName)
			visualizer = Visualizer(img[:, :, ::-1],
									metadata=MetadataCatalog.get(name_file))
			out = visualizer.draw_dataset_dict(data)
			plt.title(fileName.split('/')[-1])
			plt.imshow(
				cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
			plt.show()


if __name__ == "__main__":

	datasetDir = '/media/fernando/DATA/DATASETS/NEW_ADE/fernandoaonsoromero_bcf1579/'   
	names = []

	#################
	#### Vehicle ####
	#################
	name = "Vehicles"
	objectNames = [
		"bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle",
		"bicycle, bike, wheel, cycle",
		"crane truck", "truck, motortruck", "truck crane",
		"car, auto, automobile, machine, motorcar", "race car", "remote control car",
		"minibike, motorbike", "motorbike cart",
		"railroad train", "train, railroad train",
		"helicopter",
		"caravan",
		"warship", "ship",
		"airplane, aeroplane, plane",
		"tractor",
		"crane"
	]
	names.append(name)
	pklPath = '/media/fernando/DATA/DATASETS/NEW_ADE/fernandoaonsoromero_bcf1579/ADE20K_2021_17_01/index_ade20k.pkl'
	print(f"Convert {objectNames} in {datasetDir}")
	converter = AdeToCOCO(pklPath, datasetDir, objectNames)
	print("Start Converting.....")
	output_dir = "/media/fernando/DATA/DATASETS/DATASET_SEGMENTACION/Vehicles"
	converter.convert(output_dir, name)
	print("Finish Conversion")
	
	#######################
	#### Vial y Urbana ####
	#######################
	name = "Urban-Structures"
	objectNames = [
		"sidewalk, pavement",
		"path",
		"road, route","runway",
		"rocky wall", "rock wall", "stone wall","wall",
		"suspension bridge","bridge, span",
		"roundabout",
		"tunnel",
		"bike path","bicycle path",
		"crossing, crosswalk, crossover",
		"parking space",
		"parking"
	]
	names.append(name)
	pklPath = '/media/fernando/DATA/DATASETS/NEW_ADE/fernandoaonsoromero_bcf1579/ADE20K_2021_17_01/index_ade20k.pkl'
	print(f"Convert {objectNames} in {datasetDir}")
	converter = AdeToCOCO(pklPath, datasetDir, objectNames)
	print("Start Converting.....")
	output_dir = "/media/fernando/DATA/DATASETS/DATASET_SEGMENTACION/Urban-Structures"
	converter.convert(output_dir, name)
	print("Finish Conversion")

	###################
	#### Buildings ####
	###################
	name = "Buildings"
	objectNames = [
		"building, edifice","skyscraper","hovel, hut, hutch, shack, shanty","tower","booth, cubicle, stall, kiosk",
		"earth, ground","field","rock, stone","sand","hill","dirt track","land, ground, soil",
		"fence, fencing","bannister, banister, balustrade, balusters, handrail",
		"fuel station",
		"stadium","football stadium",
		"roof",
		"floor, flooring",
		"ceiling",
		"column, pillar",
		"swimming pool, swimming bath, natatorium", "heated swimming pool",
		"pitch",
		"spiral staircase","stairway, staircase","staircase"
	]
	names.append(name)
	pklPath = '/media/fernando/DATA/DATASETS/NEW_ADE/fernandoaonsoromero_bcf1579/ADE20K_2021_17_01/index_ade20k.pkl'
	print(f"Convert {objectNames} in {datasetDir}")
	converter = AdeToCOCO(pklPath, datasetDir, objectNames)
	print("Start Converting.....")
	output_dir = "/media/fernando/DATA/DATASETS/DATASET_SEGMENTACION/Buildings"
	converter.convert(output_dir, name)
	print("Finish Conversion")

	###################
	#### Mobiliario ####
	###################
	name = "Urban-Furniture"
	objectNames = [
		"bench",
		"bollard",
		"table","desk","coffee table, cocktail table", "chest of drawers, chest, bureau, dresser", "pool table, billiard table, snooker table", "buffet, counter, sideboard",
		"chair","seat","swivel chair",
		"streetlight, street lamp",
		"guard rail",
		"traffic light, traffic signal, stoplight",
		"bus stop",
		"drinking fountain",
		"ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin",
		"postbox, mailbox, letter box", "postbox",
		"water fountain", "fountain",
		"street sign", "signboard, sign",
		"billboard, hoarding",
		"garage door",
		"awning, sunshade, sunblind",
		"solar panel", "solar panels",
		"container", "containers",
		"towel radiator","radiator",
		"dish, dish aerial, dish antenna, saucer"
	]
	names.append(name)
	pklPath = '/media/fernando/DATA/DATASETS/NEW_ADE/fernandoaonsoromero_bcf1579/ADE20K_2021_17_01/index_ade20k.pkl'
	print(f"Convert {objectNames} in {datasetDir}")
	converter = AdeToCOCO(pklPath, datasetDir, objectNames)
	print("Start Converting.....")
	output_dir = "/media/fernando/DATA/DATASETS/DATASET_SEGMENTACION/Urban-Furniture"
	converter.convert(output_dir, name)
	print("Finish Conversion")

	###################
	#### Nature #######
	###################
	name = "Nature"
	objectNames = [
		"grass",
		"tree","palm, palm tree",
		"plant", "plant box", "plant pots", "plant stand", "plant, flora, plant life", "flower", "pot, flowerpot",
		"stones beach","beach lake",
		"water","sea","river","waterfall, falls","lake"
	]
	names.append(name)
	pklPath = '/media/fernando/DATA/DATASETS/NEW_ADE/fernandoaonsoromero_bcf1579/ADE20K_2021_17_01/index_ade20k.pkl'
	print(f"Convert {objectNames} in {datasetDir}")
	converter = AdeToCOCO(pklPath, datasetDir, objectNames)
	print("Start Converting.....")
	output_dir = "/media/fernando/DATA/DATASETS/DATASET_SEGMENTACION/Nature"
	converter.convert(output_dir, name)
	print("Finish Conversion")

	###########################
	#### Electrodomésticos ####
	###########################
	name = "Electronic"
	objectNames = [
		"computer, computing machine, computing device, data processor, electronic computer, information processing system",
		"television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box",
		"air conditioner, air conditioning",
		"radio",
		"washer, automatic washer, washing machine",
		"microwave, microwave oven", "microwave",
		"oven",
		"ceramic hob",
		"toaster",
		"coffee maker",
		"hair dryer",
		"refrigerator, icebox",
		"loudspeaker", "loudspeaker, speaker, speaker unit, loudspeaker system, speaker system", "speaker",
		"clock", "alarm clock, alarm"
	]
	names.append(name)
	pklPath = '/media/fernando/DATA/DATASETS/NEW_ADE/fernandoaonsoromero_bcf1579/ADE20K_2021_17_01/index_ade20k.pkl'
	print(f"Convert {objectNames} in {datasetDir}")
	converter = AdeToCOCO(pklPath, datasetDir, objectNames)
	print("Start Converting.....")
	output_dir = "/media/fernando/DATA/DATASETS/DATASET_SEGMENTACION/Electronic"
	converter.convert(output_dir, name)
	print("Finish Conversion")
	
	##################
	#### Security ####
	##################
	name = "Security"
	objectNames = [
		"gun",
		"rifle",
		"fire extinguisher, extinguisher, asphyxiator",
		"traffic cone",
		"helmet"
	]
	names.append(name)
	pklPath = '/media/fernando/DATA/DATASETS/NEW_ADE/fernandoaonsoromero_bcf1579/ADE20K_2021_17_01/index_ade20k.pkl'
	print(f"Convert {objectNames} in {datasetDir}")
	converter = AdeToCOCO(pklPath, datasetDir, objectNames)
	print("Start Converting.....")
	output_dir = "/media/fernando/DATA/DATASETS/DATASET_SEGMENTACION/Security"
	converter.convert(output_dir, name)
	print("Finish Conversion")
	
	##################
	#### Bathroom ####
	##################
	name = "Bathroom"
	objectNames = [
		"bathtub, bathing tub, bath, tub",
		"sink",
		"water faucet, water tap, tap, hydrant",
		"toilet, can, commode, crapper, pot, potty, stool, throne",
		"shower head", "showerhead",
		"bidet",
		"bath sponge", "sponge", "sponges",
		"urinal",
		"shower tray",
		"towel",
		"dish soap", "soap dispenser"
	]
	names.append(name)
	pklPath = '/media/fernando/DATA/DATASETS/NEW_ADE/fernandoaonsoromero_bcf1579/ADE20K_2021_17_01/index_ade20k.pkl'
	print(f"Convert {objectNames} in {datasetDir}")
	converter = AdeToCOCO(pklPath, datasetDir, objectNames)
	print("Start Converting.....")
	output_dir = "/media/fernando/DATA/DATASETS/DATASET_SEGMENTACION/Bathroom"
	converter.convert(output_dir, name)
	print("Finish Conversion")
	
	##################
	#### Kitchen ####
	##################
	name = "Kitchen"
	objectNames = [
		"knife",
		"frying pan", "frying pans",
		"cockpot","crock pot",
		"mug","coffee cup","teacup"
	]
	names.append(name)
	pklPath = '/media/fernando/DATA/DATASETS/NEW_ADE/fernandoaonsoromero_bcf1579/ADE20K_2021_17_01/index_ade20k.pkl'
	print(f"Convert {objectNames} in {datasetDir}")
	converter = AdeToCOCO(pklPath, datasetDir, objectNames)
	print("Start Converting.....")
	output_dir = "/media/fernando/DATA/DATASETS/DATASET_SEGMENTACION/Kitchen"
	converter.convert(output_dir, name)
	print("Finish Conversion")
	
	############################
	#### Decorative Objects ####
	############################
	name = "Decoration"
	objectNames = [
		"curtain, drape, drapery, mantle, pall",
		"cushion",
		"rug, carpet, carpeting",
		"fireplace, hearth, open fireplace",
		"painting, picture",
		"mirror",
		"lamp",
		"cabinet","wardrobe, closet, press",
		"sofa, couch, lounge", "armchair",
		"bunk bed", "bed",
		"windowpane, window",
		"double door","door","screen door, screen"
	]
	names.append(name)
	pklPath = '/media/fernando/DATA/DATASETS/NEW_ADE/fernandoaonsoromero_bcf1579/ADE20K_2021_17_01/index_ade20k.pkl'
	print(f"Convert {objectNames} in {datasetDir}")
	converter = AdeToCOCO(pklPath, datasetDir, objectNames)
	print("Start Converting.....")
	output_dir = "/media/fernando/DATA/DATASETS/DATASET_SEGMENTACION/Decoration"
	converter.convert(output_dir, name)
	print("Finish Conversion")

	####################
	#### Accesories ####
	####################
	name = "Accesories"
	objectNames = [
		"book","notebook","magazine",
		"cellular telephone, cellular phone, cellphone, cell, mobile phone","mobile phone","telephone","telephone, phone, telephone set","wireless phone",
		"eyeglasses", "spectacles, specs, eyeglasses, glasses", "sunglasses",
		"bag, handbag, pocketbook, purse", "handbag",
		"plastic bottles", "bottle", "bottle pack", "bottle rack", "feeding bottle",
		"camera, photographic camera", "security camera", "television camera", "video camera",
		"bag","golf bag", "sports bag", "sport bag",
		"watch"
	]
	names.append(name)
	pklPath = '/media/fernando/DATA/DATASETS/NEW_ADE/fernandoaonsoromero_bcf1579/ADE20K_2021_17_01/index_ade20k.pkl'
	print(f"Convert {objectNames} in {datasetDir}")
	converter = AdeToCOCO(pklPath, datasetDir, objectNames)
	print("Start Converting.....")
	output_dir = "/media/fernando/DATA/DATASETS/DATASET_SEGMENTACION/Accesories"
	converter.convert(output_dir, name)
	print("Finish Conversion")
	
	################
	#### Others ####
	################
	name = "Other"
	objectNames = [
		"multiple socket", "wall socket, wall plug, electric outlet, electrical outlet, outlet, electric receptacle",
		"light switch", "switch, electric switch, electrical switch", "switch",
		"can", "can, tin, tin can",
		"candle", "candle, taper, wax light", "candles", "candlestick, candle holder",
		"flashlight","torch"
	]
	names.append(name)
	pklPath = '/media/fernando/DATA/DATASETS/NEW_ADE/fernandoaonsoromero_bcf1579/ADE20K_2021_17_01/index_ade20k.pkl'
	print(f"Convert {objectNames} in {datasetDir}")
	converter = AdeToCOCO(pklPath, datasetDir, objectNames)
	print("Start Converting.....")
	output_dir = "/media/fernando/DATA/DATASETS/DATASET_SEGMENTACION/Other"
	converter.convert(output_dir, name)
	print("Finish Conversion")
	
	################
	#### Cleaning ####
	################
	name = "Cleaning"
	objectNames = [
		"broom",
		"swab, swob, mop",
		"dustpan"
	]
	names.append(name)
	pklPath = '/media/fernando/DATA/DATASETS/NEW_ADE/fernandoaonsoromero_bcf1579/ADE20K_2021_17_01/index_ade20k.pkl'
	print(f"Convert {objectNames} in {datasetDir}")
	converter = AdeToCOCO(pklPath, datasetDir, objectNames)
	print("Start Converting.....")
	output_dir = "/media/fernando/DATA/DATASETS/DATASET_SEGMENTACION/Cleaning"
	converter.convert(output_dir, name)
	print("Finish Conversion")
	
	################
	#### People ####
	################
	name = "People"
	objectNames = [
		"person, individual, someone, somebody, mortal, soul","person",
		"shoe","gym shoe, sneaker, tennis shoe","heeled shoe","high-heeled shoe","high-heeled shoes",
		"head",
		"cap", "peaked cap",
		"hat, chapeau, lid"
	]
	names.append(name)
	pklPath = '/media/fernando/DATA/DATASETS/NEW_ADE/fernandoaonsoromero_bcf1579/ADE20K_2021_17_01/index_ade20k.pkl'
	print(f"Convert {objectNames} in {datasetDir}")
	converter = AdeToCOCO(pklPath, datasetDir, objectNames)
	print("Start Converting.....")
	output_dir = "/media/fernando/DATA/DATASETS/DATASET_SEGMENTACION/People"
	converter.convert(output_dir, name)
	print("Finish Conversion")
	
	datasetDir = "/media/fernando/DATA/DATASETS/DATASET_SEGMENTACION"
	test = DemoTest(datasetDir)
	for name in names:
			print("Start Demo.....")
			test.startDemo(3, name)
			print("Finish Demo")