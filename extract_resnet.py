
import json

nb_path = r'e:\Competitions\MXB2026-Dhaka-Trio-Leveling-AgriSmartBD\plant-disease-classification-resnet-99-2.ipynb'
out_path = r'e:\Competitions\MXB2026-Dhaka-Trio-Leveling-AgriSmartBD\extracted_model.py'

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    with open(out_path, 'w', encoding='utf-8') as f_out:
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = "".join(cell['source'])
                if "class ResNet9" in source or "class ImageClassificationBase" in source or "def ConvBlock" in source:
                    f_out.write("# CELL SOURCE:\n")
                    f_out.write(source)
                    f_out.write("\n\n")
    print("Done writing to extracted_model.py")

except Exception as e:
    print(f"Error: {e}")
