import subprocess

from fastai.callback.tracker import SaveModelCallback, ShowGraphCallback
from icevision.all import *
from icevision.models.checkpoint import *
from fastai.callback.wandb import *

from loguru import logger


class DataFlywheel:
    def __init__(self, config):
        self.config = config
        self.annotation_path = config["annotation_path"]
        self.image_path = config["image_path"]
        self.wandb_project = config["wandb_project"]
        self.log_wandb = config["log_wandb"]

        if self.log_wandb:
            wandb.init(project=self.wandb_project, reinit=True)

            # Log wandb annnotations
            artifact = wandb.Artifact("xmls", type="annotations")
            artifact.add_dir(self.annotation_path)
            wandb.log_artifact(artifact)


    def load_annotations(self):
        """Load existing image annotations."""
        logger.info("Loading image annotations...")

        logger.info(f"  Annotations folder: {self.annotation_path}")
        logger.info(f"  Image folder: {self.image_path}")

        self._parser = parsers.VOCBBoxParser(
            annotations_dir=self.annotation_path, images_dir=self.image_path
        )

        train_records, valid_records = self._parser.parse()  # Defaults to 80:20 split

        image_size = 640
        train_tfms = tfms.A.Adapter(
            [*tfms.A.aug_tfms(size=image_size, presize=720), tfms.A.Normalize()]
        )
        valid_tfms = tfms.A.Adapter(
            [*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()]
        )

        self._train_ds = Dataset(train_records, train_tfms)
        self._valid_ds = Dataset(valid_records, valid_tfms)

        logger.info(f"  Train on {len(self._train_ds)} images")
        logger.info(f"  Validate on {len(self._valid_ds)} images")

    def train_model(self, batch_size=16, lr=1e-3, epoch=3):
        logger.info("Loading model...")

        self._model_type = models.mmdet.vfnet
        backbone = self._model_type.backbones.resnet50_fpn_mstrain_2x
        self._model = self._model_type.model(
            backbone=backbone(pretrained=True), num_classes=len(self._parser.class_map)
        )

        train_dl = self._model_type.train_dl(
            self._train_ds, batch_size=batch_size, num_workers=16, shuffle=True
        )
        valid_dl = self._model_type.valid_dl(
            self._valid_ds, batch_size=batch_size, num_workers=16, shuffle=False
        )

        metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
        learn = self._model_type.fastai.learner(
            dls=[train_dl, valid_dl],
            model=self._model,
            metrics=metrics,
            cbs=[ShowGraphCallback(), SaveModelCallback(fname='best_model')],
        )

        logger.info("Training model...")
        learn.fine_tune(epoch, lr, freeze_epochs=1)

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
                color_map={"person": "cyan"},
                bbox_thickness=5,
            )
        )

        if self.log_wandb:
            # Log top loss images
            wandb_images = wandb_img_preds(sorted_preds, add_ground_truth=True) 
            wandb.log({"Highest loss images": wandb_images})
            wandb.finish()

        annotations_to_review = [pred.record_id + ".xml" for pred in sorted_preds]

        with open("relabel_list.txt", "w") as file:
            file.write("\n".join(annotations_to_review))

        return annotations_to_review

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

    # def run_flywheel(self):
    #     """Execute the full DataFlywheel workflow."""
    #     logger.info("Starting DataFlywheel workflow...")
    #     self.load_annotations()
    #     self.train_model()
    #     self.relabel_data()
    #     self.export_labels()
    #     most_wrong = self.get_most_wrong()
    #     logger.info(
    #         f"DataFlywheel workflow completed. Most incorrect examples: {most_wrong}"
    #     )
