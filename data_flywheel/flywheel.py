import subprocess

from fastai.callback.tracker import SaveModelCallback, ShowGraphCallback
from icevision.all import *
from icevision.models.checkpoint import *
from fastai.callback.wandb import *

from loguru import logger
import random

class DataFlywheel:
    def __init__(self, config):
        self.config = config
        self.annotation_path = config["annotation_path"]
        self.image_path = config["image_path"]
        self.wandb_project = config["wandb_project"]
        self.log_wandb = config["log_wandb"]
        self.image_size = config["image_size"]
        self.object_class_name = config["object_class_name"]




        self.annotations_to_review = []

        if self.log_wandb:
            wandb.init(project=self.wandb_project, reinit=True)

            # Log wandb annnotations
            artifact = wandb.Artifact("xmls", type="annotations")
            artifact.add_dir(self.annotation_path)
            wandb.log_artifact(artifact)


    def load_annotations(self, show=False):
        """Load existing image annotations."""
        logger.info("Loading image annotations...")

        logger.info(f"  Annotations folder: {self.annotation_path}")
        logger.info(f"  Image folder: {self.image_path}")

        self._parser = parsers.VOCBBoxParser(
            annotations_dir=self.annotation_path, images_dir=self.image_path
        )

        _train_records, _valid_records = self._parser.parse()  # Defaults to 80:20 split

        _train_tfms = tfms.A.Adapter(
            [*tfms.A.aug_tfms(size=self.image_size, presize=int(self.image_size * 1.2)), tfms.A.Normalize()]
        )
        _valid_tfms = tfms.A.Adapter(
            [*tfms.A.resize_and_pad(self.image_size), tfms.A.Normalize()]
        )

        self._train_ds = Dataset(_train_records, _train_tfms)
        self._valid_ds = Dataset(_valid_records, _valid_tfms)

        logger.info(f"  Train on {len(self._train_ds)} images")
        logger.info(f"  Validate on {len(self._valid_ds)} images")

        if show:
            random_int = random.randint(0, len(self._train_ds) - 1)
            samples = [self._train_ds[random_int] for _ in range(6)]
            show_samples(samples, ncols=3, display_label=False, color_map={self.object_class_name:"limegreen"}, bbox_thickness=5)


    def load_model(self, batch_size=16):
        logger.info("Loading model...")

        self._model_type = models.mmdet.vfnet
        self._backbone = self._model_type.backbones.resnet50_fpn_mstrain_2x
        self._model = self._model_type.model(
            backbone=self._backbone(pretrained=True), num_classes=len(self._parser.class_map)
        )

        _train_dl = self._model_type.train_dl(
            self._train_ds, batch_size=batch_size, num_workers=16, shuffle=True
        )
        _valid_dl = self._model_type.valid_dl(
            self._valid_ds, batch_size=batch_size, num_workers=16, shuffle=False
        )

        cbs = [ShowGraphCallback(), SaveModelCallback(fname='best_model')]

        if self.log_wandb:
            cbs.append(WandbCallback())

        metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
        
        self.learn = self._model_type.fastai.learner(
            dls=[_train_dl, _valid_dl],
            model=self._model,
            metrics=metrics,
            cbs=cbs
        )

        logger.info("Finding optimal learning rate..")
        self.learn.lr_find()

    def train_model(self, lr=1e-3, epoch=3, freeze_epoch=1):
        logger.info("Training model...")
        self.learn.fine_tune(epoch, lr, freeze_epochs=freeze_epoch)

        # Save checkpoint
        save_icevision_checkpoint(self._model,
                        model_name='mmdet.vfnet', 
                        backbone_name='resnet50_fpn_mstrain_2x',
                        img_size=640,
                        classes=self._parser.class_map.get_classes(),
                        filename='./models/model_checkpoint.pth',
                        meta={'icevision_version': 'master'})

    def get_most_wrong(self, method="top-loss"):
        logger.info("Identifying most incorrect examples...")

        sorted_samples, sorted_preds, losses_stats = (
            self._model_type.interp.plot_top_losses(
                self._model,
                self._valid_ds,
                sort_by="loss_total",
                n_samples=20,
                display_label=False,
                color_map={self.object_class_name: "limegreen"},
                bbox_thickness=5,
            )
        )

        if self.log_wandb:
            # Log top loss images
            wandb_images = wandb_img_preds(sorted_preds, add_ground_truth=True) 
            wandb.log({"Highest loss images": wandb_images})
            wandb.finish()

        self.annotations_to_review = [pred.record_id + ".xml" for pred in sorted_preds]

        with open("relabel_list.txt", "w") as file:
            file.write("\n".join(self.annotations_to_review))

        return self.annotations_to_review

    def relabel_data(self):
        logger.info("Launching streamlit to review annotations...")
        logger.info("Review annotations here: http://0.0.0.0:8501")

        subprocess.run(
            [
                "streamlit",
                "run",
                "data_flywheel/st_relabel.py",
                "--server.address",
                "0.0.0.0",
                "--",
                self.annotation_path,
                self.image_path
            ]
        )

    def run(self, lr=1e-3, batch_size=16, epoch=10, freeze_epoch=3, image_size=720):
        """Execute the full DataFlywheel workflow."""
        logger.info("Running one full cycle of the flywheel...")
        self.load_annotations(image_size=image_size)
        self.load_model(batch_size=batch_size)
        self.train_model(lr=lr, epoch=epoch, freeze_epoch=freeze_epoch)
        self.get_most_wrong(method='top-loss')
        self.relabel_data()
