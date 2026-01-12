
import json

nb_path = r'e:\Competitions\MXB2026-Dhaka-Trio-Leveling-AgriSmartBD\plant-disease-classification-resnet-99-2.ipynb'

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if ".classes" in source:
                 print("POTENTIAL CLASSES SOURCE:")
                 print(source)
                 print("-" * 20)
except Exception as e:
    print(f"Error: {e}")
