import requests
import os
import selenium

""" 
TODO:  
1. create a queue of urls to download
2. write a multi-threaded program to create headless browsers and subscribe to the queue
3. download the images and save them to a folder/s3 bucket?
"""
def headless_browser():
    """create a headless browser"""
    options = selenium.webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('window-size=1200x600')
    return selenium.webdriver.Chrome(chrome_options=options)

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

