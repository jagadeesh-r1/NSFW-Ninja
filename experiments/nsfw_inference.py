# Inference results on our dataset from Library - https://pypi.org/project/nsfw-detector/

from nsfw_detector import predict
from PIL import Image
import yaml
import pickle
from tqdm import tqdm
import utils


with open('configs/nsfw_config.yaml') as f:
    config = yaml.safe_load(f)

model_path = config["nsfw_model_path"]
model = predict.load_model(model_path)


with open(config["bumble_dump"], 'rb') as file:
    bumble_predictions = pickle.load(file)


nsfw_results = [([], [], []) for _ in range(10)]

for i in tqdm(range(len(nsfw_results))):
    predictions, images_path = bumble_predictions[i]
    for j in range(len(images_path)):
        pred = predict.classify(model, images_path[j])
        results = pred[images_path[j]]
        category = max(results, key=lambda k: results[k])

        nsfw_results[i][0].append(predictions[j])
        nsfw_results[i][1].append(images_path[j])
        nsfw_results[i][2].append(category)


utils.dump_nsfw_results(nsfw_results, config['dump_file'], config['dump_results'],  config['print_results'])