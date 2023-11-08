import numpy as np
import pickle

bumble_dump = "pkls/bumble_predictions.pkl"
nsfw_dump = "pkls/nsfw_predictions.pkl"

categories = {'sexy':0, 'neutral':1, 'porn':2, 'hentai':3, 'drawings':4}
rev_categories = {categories[k]:k for k in categories}

per_bin_analysis = np.zeros((10,5))
per_bin_length = np.zeros((10,1))


with open(nsfw_dump, 'rb') as file:
    nsfw_dump = pickle.load(file)

for i in range(len(nsfw_dump)):
    predictions, images_path, nsfw_categories = nsfw_dump[i]
    # print(predictions)
    for j in range(len(predictions)):
        per_bin_analysis[i][categories[nsfw_categories[j]]] += 1

    per_bin_length[i][0] = len(predictions)

per_bin_analysis = 100.0 * per_bin_analysis / per_bin_length

for i in range(len(nsfw_dump)):
    print("For Bumble Predictions between {}%-{}% : {} Images".format(i*100, (i+1)*100, per_bin_length[i][0]))
    for j in range(5):
        print("Category {} : {}".format(rev_categories[j], per_bin_analysis[i][j]))
    print("")
