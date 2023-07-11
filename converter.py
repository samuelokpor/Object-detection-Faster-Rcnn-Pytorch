# import os
# import glob
# from PIL import Image
# from lxml import etree

# # directory containing .bmp images and .xml annotation files
# image_directory = './data/door_panel/'
# output_directory = './data/door_train/'


# # ensure output directory exists
# os.makedirs(output_directory, exist_ok=True)

# # convert .bmp images to .jpg
# for bmp_file in glob.glob(image_directory + '/*.bmp'):
#     img = Image.open(bmp_file)
#     jpg_file = os.path.splitext(bmp_file)[0] + '.jpg'
#     img.save(output_directory + '/' + os.path.basename(jpg_file), 'JPEG')

# # update .xml annotation files
# for xml_file in glob.glob(image_directory + '/*.xml'):
#     tree = etree.parse(xml_file)
#     for path in tree.xpath("//path"):
#         path.text = os.path.splitext(path.text)[0] + '.jpg'
#     for filename in tree.xpath("//filename"):
#         filename.text = os.path.splitext(filename.text)[0] + '.jpg'
#     output_xml = output_directory + '/' + os.path.basename(xml_file)
#     tree.write(output_xml, pretty_print=True)


import os
import glob
from PIL import Image
from lxml import etree

# directory containing images and .xml annotation files
image_directory = './data/door_panel/'
output_directory = './data/door_train/'

# ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# image types
image_types = ['*.bmp', '*.png', '*.jpeg']  # add or remove as needed

# convert images to .jpg
for image_type in image_types:
    for file in glob.glob(image_directory + '/' + image_type):
        img = Image.open(file)
        jpg_file = os.path.splitext(file)[0] + '.jpg'
        img.save(output_directory + '/' + os.path.basename(jpg_file), 'JPEG')

# update .xml annotation files
for xml_file in glob.glob(image_directory + '/*.xml'):
    tree = etree.parse(xml_file)
    for path in tree.xpath("//path"):
        base, ext = os.path.splitext(path.text)
        if ext.lower() in ['.bmp', '.png', '.jpeg']:  # add or remove as needed
            path.text = base + '.jpg'
    for filename in tree.xpath("//filename"):
        base, ext = os.path.splitext(filename.text)
        if ext.lower() in ['.bmp', '.png', '.jpeg']:  # add or remove as needed
            filename.text = base + '.jpg'
    output_xml = output_directory + '/' + os.path.basename(xml_file)
    tree.write(output_xml, pretty_print=True)