import requests
import os
import selenium

source_directory = "/home/jaggu/nsfw_data_source_urls/raw_data"

# for directory in os.listdir(source_directory):
for i in os.walk(source_directory):
    image_counter = 0
    try:
        text_path = i[0] + '/' + i[2][0]
        # print(text_path)
        with open(text_path, 'r') as file:
            for index,image_url in enumerate(file):
                # print(index, image_url)
                try:
                    img_req = requests.get(image_url)
                    print(img_req.status_code, image_url)
                    img_data = img_req.content
                    if img_req.status_code == 200:
                        if 'jpg' in image_url:
                            with open(f'code/dataset/images/{image_counter}.jpg', 'wb') as handler:
                                handler.write(img_data)
                        if 'png' in image_url:
                            with open(f'code/dataset/images/{image_counter}.png', 'wb') as handler:
                                handler.write(img_data)
                        image_counter += 1
                except requests.ConnectionError:
                    print(image_url)
            if image_counter > 100:
                break
    except IndexError:
        pass

