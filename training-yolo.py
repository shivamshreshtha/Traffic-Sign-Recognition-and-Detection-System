"""
Traffic Sign Recognition and Detection System
Deep Learning Project | Department of FET AIML
Under the Guidance of IBM | Academic Session: 2024 - 2026

Authors:
- SHIVAM SHRESHTHA (FET-BAML-2024-28-084)
- K. DIGANT PATRO (FET-BAML-2024-28-082)
- AMRITANSH MANOJ SOHANI (FET-BAML-2024-28-083)
- ANSHUMAN NATH (FET-BAML-2024-28-085)
"""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("./runs/detect/train2/weights/best.pt")
    # model.train(data='./Datasets/GTSDB/data.yaml', workers=4)
    model.predict(source=0, show=True)
