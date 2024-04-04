from data_flywheel import DataFlywheel

# Create a configuration dictionary
config = {
    'annotation_path': '/root/data-flywheel/notebooks/objectlab/pascal_voc_annotations/',
    'image_path': '/workspace/yolo_v8_training/oiv7_full/validation/',
    'log_wandb': True,
    'wandb_project': 'data-flywheel-test'
}

# Instantiate the DataFlywheel
flywheel = DataFlywheel(config)


flywheel.load_annotations()

flywheel.train_model(batch_size=32, epoch=10, freeze_epoch=3)

annotations_to_review = flywheel.get_most_wrong(method='top-loss')

flywheel.relabel_data()