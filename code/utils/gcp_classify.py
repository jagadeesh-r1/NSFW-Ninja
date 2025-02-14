from google.cloud import vision
import os
import csv


vision_client = vision.ImageAnnotatorClient()


def classify_image(image_path, writer):
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    response = vision_client.safe_search_detection(image=image)
    safe = response.safe_search_annotation
    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = (
        "UNKNOWN",
        "VERY_UNLIKELY",
        "UNLIKELY",
        "POSSIBLE",
        "LIKELY",
        "VERY_LIKELY",
    )
    # print("Safe search:")

    # print(f"adult: {likelihood_name[safe.adult]}")
    # print(f"medical: {likelihood_name[safe.medical]}")
    # print(f"spoofed: {likelihood_name[safe.spoof]}")
    # print(f"violence: {likelihood_name[safe.violence]}")
    # print(f"racy: {likelihood_name[safe.racy]}")
    writer.writerow(
            [
                image_path,
                likelihood_name[safe.adult],
                likelihood_name[safe.medical],
                likelihood_name[safe.spoof],
                likelihood_name[safe.violence],
                likelihood_name[safe.racy],
            ]
        )


    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

if __name__ == "__main__":
    result_file = "code/dataset/gcp_result.csv"
    if os.path.exists(result_file):
        file = open(result_file, "a")
        writer = csv.writer(file)
        # os.remove(result_file)
    else:
        file = open(result_file, "w")
        writer = csv.writer(file)
        writer.writerow(["image", "adult", "medical", "spoofed", "violence", "racy"])
    existing_images = []
    with open(result_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            existing_images.append(row[0])
    print("Existing images: ", len(existing_images))
    print("Classifying images...")
    dataset_path = "/home/jaggu/nsfw_data_scraper/dataset/"
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            image_path = os.path.join(root, file)
            if image_path in existing_images:
                continue
            else:
                try:
                    classify_image(image_path, writer)
                except Exception as e:
                    print(e)
                    continue
    