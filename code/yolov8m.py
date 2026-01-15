# from ultralytics import YOLO
# # from ultralytics.utils.torch_utils import select_device
# #
# # Initialize a YOLOv8 model (medium-size backbone).
# # model = YOLO("yolov8m.pt")
# #
# # model.train(
# #     data="/path/to/your/data.yaml",
# #     epochs=100,                 # Train for 100 epochs
# #     imgsz=1280,                 # Higher resolution can help small-object detection
# #     batch=4,                    # Keep batch small to avoid OOM on Apple Silicon
# #     device="mps",               # Use Metal (Apple Silicon) acceleration
# #     workers=2,                  # Data loader workers
# #     name="bat_detector_final",  # Run name
# #     patience=20,                # Early-stopping patience
# #     save=True,
# #     save_period=10,             # Save a checkpoint every 10 epochs
# #     verbose=True,
# #     close_mosaic=10,            # Disable mosaic after N epochs for stability
# #     rect=True,                  # Rectangular training can help anchor/shape fitting
# #     single_cls=False,           # Keep multi-class mode if you have >1 class
# #     cache=False,
# #     augment=False,              # Disable built-in augmentation if you already augmented offline
# # )

from ultralytics import YOLO

# Load the previously trained best checkpoint and continue training from it.
# Note: This loads weights but does NOT resume the optimizer/epoch state unless resume=True.
model = YOLO(
    "/Users/test/Desktop/Ex_Files_OpenCV_Python_Developers/bat_project/code/code2/runs/detect/bat_detector_final/weights/best.pt"
)

model.train(
    data="/Users/test/Desktop/Ex_Files_OpenCV_Python_Developers/bat_project/dataset2/my_yolo_dataset/data.yaml",
    epochs=100,
    imgsz=760,
    batch=2,
    device="mps",                    # Apple Silicon Metal acceleration
    name="bat_detector_final_v2",
    project="bat_project/code/code2/runs/detect",
    workers=2,
    patience=20,
    save=True,
    save_period=10,
    verbose=True,
    single_cls=False,                # Set True only if you want to treat all classes as one
    rect=True,                       # Rectangular training can help with small objects
    close_mosaic=5,                  # Disable mosaic after 5 epochs for stability
    mosaic=True,                     # Enable mosaic augmentation
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # HSV color jitter
    translate=0.05,
    scale=0.4,
    fliplr=0.5,
    cache=False,
    augment=True,                    # Enable Ultralytics augmentation pipeline
    resume=False                     # Do not resume epoch/optimizer state; only load weights
)
