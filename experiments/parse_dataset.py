import numpy as np
import pickle

nsfw_dump = "pkls/nsfw_predictions_test.pkl"
# nsfw_dump = "pkls/nsfw_predictions.pkl"

categories = {'sexy':0, 'neutral':1, 'porn':2, 'hentai':3, 'drawings':4}
allowed_categories = ['sexy', 'porn', 'hentai']
allowed_tight_categories = ['porn', 'hentai']

per_bin_analysis = np.zeros((10,5))
per_bin_length = np.zeros((10,1))


with open(nsfw_dump, 'rb') as file:
    nsfw_dump = pickle.load(file)

for i in range(len(nsfw_dump)):
    predictions, images_path, nsfw_categories = nsfw_dump[i]
    if i >= 3 and i <= 5:
        for j in range(len(predictions)):
            if nsfw_categories[j] in allowed_tight_categories:
                print(images_path[j])
    if i == 6:
        for j in range(len(predictions)):
            if nsfw_categories[j] in allowed_categories:
                print(images_path[j])
    if i >= 7:
        for j in range(len(predictions)):
            print(images_path[j])