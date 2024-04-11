import os
import xml.etree.ElementTree as ET
from PIL import Image
from xml.dom import minidom


def create_object_element(object_data):
    obj_elem = ET.Element('object')
    ET.SubElement(obj_elem, 'name').text = str(object_data['label'])
    ET.SubElement(obj_elem, 'pose').text = 'Unspecified'
    ET.SubElement(obj_elem, 'truncated').text = '0'
    ET.SubElement(obj_elem, 'difficult').text = '0'
    
    bndbox = ET.SubElement(obj_elem, 'bndbox')
    ET.SubElement(bndbox, 'xmin').text = str(object_data['x1'])
    ET.SubElement(bndbox, 'ymin').text = str(object_data['y1'])
    ET.SubElement(bndbox, 'xmax').text = str(object_data['x2'])
    ET.SubElement(bndbox, 'ymax').text = str(object_data['y2'])
    
    return obj_elem

def create_voc_xml(filename, objects, output_dir):
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'filename').text = os.path.basename(filename)
    
    image = Image.open(filename)
    width, height = image.size
    
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(len(image.getbands()))
    
    for object_data in objects:
        annotation.append(create_object_element(object_data))
    
    # Create a new XML document
    xml_str = ET.tostring(annotation, encoding='utf-8')
    dom = minidom.parseString(xml_str)
    
    # Pretty-print the XML with indentation
    pretty_xml = dom.toprettyxml(indent='    ')
    
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(filename))[0]}.xml")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)