import os
import xml.etree.ElementTree as ET
import csv

# Define the folder containing the XML files
xml_folder = os.path.join(os.getcwd(),'stretched-anno')


# Define the output CSV file
csv_file =  os.path.join(os.getcwd(),'output-stretched.csv')

# Define the header row for the CSV file
header = ['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

# Initialize a list to store the data
data = []

# Loop through all the XML files in the folder
for filename in os.listdir(xml_folder):
    if not filename.endswith('.xml'):
        continue
    xml_file = os.path.join(xml_folder, filename)

    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get the filename
    filename = root.find('filename').text

    # Loop through all the objects in the XML file
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        obj_bbox = obj.find('bndbox')
        obj_xmin = int(obj_bbox.find('xmin').text)
        obj_ymin = int(obj_bbox.find('ymin').text)
        obj_xmax = int(obj_bbox.find('xmax').text)
        obj_ymax = int(obj_bbox.find('ymax').text)

        # Append the data to the list
        if len(root.findall('object')) == 1:
            data.append([filename, obj_name, obj_xmin,
                        obj_ymin, obj_xmax, obj_ymax])
        else:
            obj_id = obj_name.split(' ')[-1]
            data.append([filename, f"{obj_name}_{obj_id}",
                        obj_xmin, obj_ymin, obj_xmax, obj_ymax])

# Write the data to the CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(data)
