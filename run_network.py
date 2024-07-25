#!/sur/bin/python3

import jetson_inference
import jetson_utils
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file_path', type=str, help="Path of input file")

opt = parser.parse_args()

file_path = opt.file_path

model_path = 'skin_cancer96.onnx'
labels_path = '/home/nvidia/jetson-inference/python/training/classification/models/skin_cancer/labels.txt'

net = jetson_inference.imageNet(argv=[
	'--model='+model_path,
	'--input_blob=input_0',
	'--output_blob=output_0',
	'--labels='+labels_path
])

img = jetson_utils.loadImage(file_path)
class_id, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_id)

print(f"Processed {file_path}\nClass: {class_desc}, Confidence: {confidence:.2f}")

if(class_desc == 'Benign'):
    if(confidence < 0.85):
        print("Benign lesion detected with low confidence.\nMore testing suggested")
    else:
        print("Benign lesion detected with high confidence.")
else:
    print("**Malignant lesion detected**\nFurther testing required")

