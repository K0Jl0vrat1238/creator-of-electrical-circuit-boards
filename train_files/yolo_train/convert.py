from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="C:/Users/fedor/Desktop/yolo_train/cocolabels",  # directory containing your JSON files
    save_dir="C:/Users/fedor/Desktop/yolo_train/converted",  # where to save converted labels
    cls91to80=False,  # IMPORTANT: set False for custom datasets
    use_segments=True,
)