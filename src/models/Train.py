import comet_ml

from ultralytics import YOLO
from src.data.dataset_services import convert_coco, generate_txt_files

def train(data_dir, category, epochs, update_dataset=False):

	project_name = f"{category}_project"
	comet_ml.login(project_name=project_name)

	# Cuando hay alguna actualización de clases o cambio de dataset, es necesario actualizar el dataset con las nuevas clases/imagenes
	if update_dataset:

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

		# Generar los ficheros txt nuevos si se ha generado un nuevo dataset o leer el último.
		generate_txt_files(data_dir, category)
		# Leer las rutas desde los COCO para facilitar la lectura y así es más facil
		# Generar los 3 ds, test, val y train

	yaml_path = f"{category}.yaml"

	model = YOLO("trained_models/yolov8n-seg.pt")

	results = model.train(data=yaml_path,
						  project=project_name,
						  epochs=epochs,
						  degrees=180,
						  translate=0,
						  scale=0,
						  fliplr=0,
						  mosaic=0,
						  auto_augment="",
						  erasing=0,
						  crop_fraction=0)


if __name__=="__main__":

	categories = ["Bathroom", "Buildings",
				  "Cleaning", "Decoration", "Electronic",
				  "Kitchen", "Nature", "Other",
				  "People", "Security", "Urban-Furniture",
				  "Urban-Structures", "Vehicles"]

	data_dir = "/media/fernando/DATA/DATASETS/DATASET_SEGMENTACION"
	epochs = 500

	for category in categories:
		try:
			train(data_dir, category, epochs, update_dataset=True)

		except Exception as e:
			print(e)
