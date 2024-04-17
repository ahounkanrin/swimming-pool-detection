import pandas as pd
import numpy as np
import os

DATA_DIR = "/home/ahounkanrin/swimming-pool-detection/DATASET/BH-POOLS"


indices = {"train": (1, 7), "test":(7, 9)}
def create_csv(subset):
    imgs = []
    masks = []
    df = pd.DataFrame()
    for region_id in range(indices[subset][0], indices[subset][1]):
        for x in os.listdir(os.path.join(DATA_DIR, f"{subset.upper()}", f"REGION_{region_id}", "ANNOTATION_COLOR")):
            if x.endswith(".png"):
                img_id = x.split(".")[0]
                masks.append(os.path.join(f"{subset.upper()}", f"REGION_{region_id}", "ANNOTATION_COLOR", x))
                imgs.append(os.path.join(f"{subset.upper()}", f"REGION_{region_id}", "IMAGES", f"{img_id}.jpg"))
            
    df["image"] = imgs
    df["mask"] = masks
    df.to_csv(f"{subset}.csv", sep=",")
    print(f"{subset}.csv file saved.")


# imgs_test = []
# masks_test = []
# df_test = pd.DataFrame()

# for region_id in range(7, 9):
#     for x in os.listdir(os.path.join(DATA_DIR, "TEST", f"REGION_{region_id}", "ANNOTATION_COLOR")):
#         if x.endswith(".png"):
#             mg_id = x.split(".")[0]
#             print(x)
#             masks_test.append(os.path.join("TEST", f"REGION_{region_id}", "ANNOTATION_COLOR", x))
#             imgs_test.append(os.path.join("TEST", f"REGION_{region_id}", "IMAGES", f"{img_id}.jpg"))
        
# df_test["image"] = imgs_test
# df_test["mask"] = masks_test
# df_test.to_csv("test.csv", sep=",")

if __name__ == "__main__":
    create_csv("train")
    create_csv("test")