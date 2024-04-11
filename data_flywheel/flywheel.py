import os
import random
import subprocess
import xml.etree.ElementTree as ET

import pandas as pd
from fastai.callback.tracker import SaveModelCallback, ShowGraphCallback
from fastai.callback.wandb import *
from icevision.all import *
from icevision.models.checkpoint import *
from loguru import logger
from tqdm.auto import tqdm

from .utils import create_voc_xml


class DataFlywheel:
    def __init__(self, config):
        self.config = config
        self.annotation_parquet = config["annotation_parquet"]  # parquet files downloaded from S3
        self.voc_annotations_folder = config["voc_annotations_folder"] # where to store the voc format
        self.image_path = config["image_path"]
        
        self.log_wandb = config["log_wandb"]
        self.wandb_project = config["wandb_project"]
        
        self.image_size = config["image_size"]
        self.object_class_name = config["object_class_name"]

        self.annotations_to_review = []

        if self.log_wandb:
            wandb.init(project=self.wandb_project, reinit=True)

    def load_annotations(self, show=False, reinit=False):
        """Load existing image annotations."""
        logger.info("Loading image annotations...")

        df = pd.read_parquet(self.annotation_parquet)
        df = df[df["label"] == self.object_class_name]

        df = pd.DataFrame(
            {
                "filename": df["filename"],
                "label": df["label"],
                "confidence": df["confidence"],
                "x1": df["col_x"],
                "y1": df["row_y"],
                "x2": df["col_x"] + df["width"],
                "y2": df["row_y"] + df["height"],
            }
        )

        df = df.astype({'x1': int, 'y1': int, 'x2': int,  'y2': int})

        if reinit:
            if os.path.exists(self.voc_annotations_folder):
                shutil.rmtree(self.voc_annotations_folder)

        os.makedirs(
            self.voc_annotations_folder, exist_ok=True
        )

        grouped_df = (
            df.groupby("filename")
            .apply(lambda x: x.to_dict("records"))
            .reset_index(name="objects")
        )

        for row in tqdm(
            grouped_df.itertuples(index=False),
            total=len(grouped_df),
            desc="Converting to VOC",
            unit=" imgs",
        ):
            filename = row.filename
            objects = row.objects
            create_voc_xml(filename, objects, self.voc_annotations_folder)

        logger.info(f"  VOC Annotations folder: {self.voc_annotations_folder}")
        logger.info(f"  Image folder: {self.image_path}")

        if self.log_wandb:
            artifact = wandb.Artifact("input_annotations_xml", type="xml")
            artifact.add_dir(self.voc_annotations_folder)
            wandb.log_artifact(artifact)

            artifact = wandb.Artifact("input_annotations_df", type="dataframe")
            artifact.add_file(self.annotation_parquet)
            wandb.log_artifact(artifact)


        self._parser = parsers.VOCBBoxParser(
            annotations_dir=self.voc_annotations_folder, images_dir=self.image_path
        )

        _train_records, _valid_records = self._parser.parse()  # Defaults to 80:20 split

        _train_tfms = tfms.A.Adapter(
            [
                *tfms.A.aug_tfms(
                    size=self.image_size, presize=int(self.image_size * 1.2)
                ),
                tfms.A.Normalize(),
            ]
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
            show_samples(
                samples,
                ncols=3,
                display_label=False,
                color_map={self.object_class_name: "limegreen"},
                bbox_thickness=5,
            )

    def export_annotations(self, output_filename):

        def process_xml_file(file_path):
            tree = ET.parse(file_path)
            root = tree.getroot()

            filename = root.find("filename").text
            size = root.find("size")
            width = int(size.find("width").text)
            height = int(size.find("height").text)
            depth = int(size.find("depth").text)

            data = []
            for obj in root.findall("object"):
                name = obj.find("name").text
                xmin = int(obj.find("bndbox/xmin").text)
                ymin = int(obj.find("bndbox/ymin").text)
                xmax = int(obj.find("bndbox/xmax").text)
                ymax = int(obj.find("bndbox/ymax").text)
                data.append(
                    {
                        "filename": filename,
                        "width": width,
                        "height": height,
                        "depth": depth,
                        "name": name,
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                    }
                )

            return data

        data = []
        for filename in os.listdir(self.voc_annotations_folder):
            if filename.endswith(".xml"):
                file_path = os.path.join(self.voc_annotations_folder, filename)
                xml_data = process_xml_file(file_path)
                data.extend(xml_data)

        df_annots = pd.DataFrame(data)

        df_annots = df_annots[["filename", "name", "xmin", "ymin", "xmax", "ymax"]]
        df_annots.rename(columns={"name": "label"}, inplace=True)
        df_annots["filename"] = self.image_path + df_annots["filename"]

        # Convert bounding boxes
        df_annots["col_x"] = (df_annots["xmin"] + df_annots["xmax"]) / 2
        df_annots["row_y"] = (df_annots["ymin"] + df_annots["ymax"]) / 2
        df_annots["width"] = df_annots["xmax"] - df_annots["xmin"]
        df_annots["height"] = df_annots["ymax"] - df_annots["ymin"]

        df_annots = df_annots[
            ["filename", "label", "col_x", "row_y", "width", "height"]
        ]

        df_annots.to_parquet(f"{output_filename}")

        if self.log_wandb:
            artifact = wandb.Artifact("output_annotations_xml", type="xml")
            artifact.add_dir(self.voc_annotations_folder)
            wandb.log_artifact(artifact)
        
            artifact = wandb.Artifact("output_annotations_df", type="dataframe")
            artifact.add_file(output_filename)
            wandb.log_artifact(artifact)
            wandb.finish()

    def load_model(self, batch_size=16):
        logger.info("Loading model...")

        self._model_type = models.mmdet.vfnet
        self._backbone = self._model_type.backbones.resnet50_fpn_mstrain_2x
        self._model = self._model_type.model(
            backbone=self._backbone(pretrained=True),
            num_classes=len(self._parser.class_map),
        )

        _train_dl = self._model_type.train_dl(
            self._train_ds, batch_size=batch_size, num_workers=16, shuffle=True
        )
        _valid_dl = self._model_type.valid_dl(
            self._valid_ds, batch_size=batch_size, num_workers=16, shuffle=False
        )

        cbs = [ShowGraphCallback(), SaveModelCallback(fname="best_model")]

        if self.log_wandb:
            cbs.append(WandbCallback())

        metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

        self.learn = self._model_type.fastai.learner(
            dls=[_train_dl, _valid_dl], model=self._model, metrics=metrics, cbs=cbs
        )

        logger.info("Finding optimal learning rate..")
        self.learn.lr_find()

    def train_model(self, lr=1e-3, epoch=3, freeze_epoch=1):
        logger.info("Training model...")
        self.learn.fine_tune(epoch, lr, freeze_epochs=freeze_epoch)

        # Save checkpoint
        save_icevision_checkpoint(
            self._model,
            model_name="mmdet.vfnet",
            backbone_name="resnet50_fpn_mstrain_2x",
            img_size=640,
            classes=self._parser.class_map.get_classes(),
            filename="./models/model_checkpoint.pth",
            meta={"icevision_version": "master"},
        )

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

        self.annotations_to_review = [pred.record_id + ".xml" for pred in sorted_preds]

        with open("relabel_list.txt", "w") as file:
            file.write("\n".join(self.annotations_to_review))

        return self.annotations_to_review

    def relabel_data(self, relabel_filename):
        logger.info("Launching streamlit to review annotations...")
        logger.info("Review annotations here: http://0.0.0.0:8501")

        try:
            result = subprocess.run(
                [
                    "streamlit",
                    "run",
                    "/root/data-flywheel/data_flywheel/st_relabel.py",
                    "--server.address",
                    "0.0.0.0",
                    "--",
                    self.voc_annotations_folder,
                    self.image_path,
                    relabel_filename,
                    "--custom_labels",
                    self.object_class_name,
                ],
                capture_output=True,
                text=True,
            )

            # Check if the command executed successfully
            if result.returncode == 0:
                logger.info("Command executed successfully.")
            else:
                logger.error("Command failed with error:")
                logger.error(result.stderr)

        except subprocess.CalledProcessError as e:
            logger.error("Command failed with error:")
            logger.error(e.stderr)

    def run(self, lr=1e-3, batch_size=16, epoch=10, freeze_epoch=3):
        """Execute the full DataFlywheel workflow."""
        logger.info("Running one full cycle of the flywheel...")
        self.load_annotations()
        self.load_model(batch_size=batch_size)
        self.train_model(lr=lr, epoch=epoch, freeze_epoch=freeze_epoch)
        self.get_most_wrong(method="top-loss")
        self.relabel_data()
