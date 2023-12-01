import os

filename = "/home/neelesh/research-project-security-sheriffs/code/dataset_log/test.txt"
file = open(filename)

img_names = file.readlines()
allowed_imgs = []
for img_name in img_names:
    img_name = img_name.strip()
    img_name = img_name.split("/")[-1]
    allowed_imgs.append(img_name)


target_folder = "/home/neelesh/research-project-security-sheriffs/dataset/test/nsfw"
target_img_list = os.listdir(target_folder)

to_delete = 0
for target_img in target_img_list:
    if target_img not in allowed_imgs:
        # print(target_img)
        target_img = os.path.join(target_folder, target_img)
        del_str = "rm {}".format(target_img)
        os.system(del_str)
        to_delete += 1

print(to_delete)