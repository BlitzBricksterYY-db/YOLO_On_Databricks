# Databricks notebook source
# MAGIC %md
# MAGIC ## Quick Overview: 
# MAGIC
# MAGIC <br> 
# MAGIC
# MAGIC #### YOLO Model
# MAGIC [YOLO (You Only Look Once)](https://ieeexplore.ieee.org/document/7780460) is a state-of-the-art, real-time object detection system. It frames object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities. This approach allows YOLO to achieve high accuracy and speed, making it suitable for real-time applications.    
# MAGIC
# MAGIC <!-- ![Computer Vision Tasks supported by Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-tasks-banner.avif)    -->
# MAGIC
# MAGIC Offered as part of the [Ultralytics AI framework](https://www.ultralytics.com/), [YOLO11 supports multiple computer vision tasks](https://docs.ultralytics.com/tasks/).    
# MAGIC
# MAGIC
# MAGIC #### Instance Segmentation
# MAGIC Recent updates to the YOLO model have introduced capabilities for instance segmentation. [Instance Segmentation](https://docs.ultralytics.com/tasks/segment/#models) not only detects objects but also delineates the exact shape of each object, providing pixel-level masks. 
# MAGIC
# MAGIC ![what_is_instance_segmentation](./imgs/what_is_instance_segmentation.png)
# MAGIC
# MAGIC <!-- ![https://manipulation.csail.mit.edu/segmentation.html](https://manipulation.csail.mit.edu/data/coco_instance_segmentation.jpeg)  -->
# MAGIC       
# MAGIC
# MAGIC This is particularly useful in applications requiring precise object boundaries, for example in medical imaging, autonomous driving, as well as robotics e.g. : 
# MAGIC
# MAGIC <img src="https://github.com/ultralytics/docs/releases/download/0/instance-segmentation-examples.avif"
# MAGIC      alt="YOLO Instance Segmentation Example"
# MAGIC      width="800"
# MAGIC      style="margin: 200px;"/>
# MAGIC
# MAGIC <!-- ![YOLO Instance Segmentation Example](https://github.com/ultralytics/docs/releases/download/0/instance-segmentation-examples.avif) -->
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC #### Transfer Learning
# MAGIC [Transfer learning](https://www.ultralytics.com/glossary/transfer-learning) involves taking a pre-trained model and fine-tuning it on a new dataset. This approach leverages the knowledge gained from a large dataset and applies it to a specific task, reducing the need for extensive computational resources and training time. In the context of YOLO, transfer learning allows us to adapt the model to new object classes or domains with limited data.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC #### Note on architecture and segmentation implementation 
# MAGIC - **YOLO Architecture**: The YOLO architecture consists of convolutional layers followed by fully connected layers, designed to predict bounding boxes and class probabilities directly from the input image.
# MAGIC - **Instance Segmentation**: The recent YOLO models incorporate segmentation heads that output pixel-level masks for each detected object, enhancing the model's ability to perform instance segmentation.   
# MAGIC       
# MAGIC <br> 
# MAGIC
# MAGIC
# MAGIC For more detailed information, refer to the original YOLO model and the latest research on instance segmentation.    
# MAGIC `YOLO Refs:`
# MAGIC [`v1`](https://arxiv.org/abs/1506.02640), [`v2`](https://arxiv.org/abs/1612.08242), [`v3`](https://arxiv.org/abs/1804.02767), [`v4`](https://arxiv.org/abs/2004.10934), 
# MAGIC [`v5`](https://docs.ultralytics.com/models/yolov5/), 
# MAGIC [`v6`](https://arxiv.org/abs/2209.02976), [`v7`](https://arxiv.org/abs/2207.02696), [`v8`](https://docs.ultralytics.com/models/yolov8/), ... [`v11`](https://docs.ultralytics.com/models/11/#key-features) (recent version focused on improved performance and ease of use, used in this example)
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---    

# COMMAND ----------

# MAGIC %md
# MAGIC ## Applying `YOLO_v11` Instance Segmentation within Databricks 
# MAGIC
# MAGIC In the rest of this notebook, we will provide an example of how to leverage [YOLO Instance Segmentation](https://docs.ultralytics.com/tasks/segment/) model in _transfer learning_. 
# MAGIC
# MAGIC #### CellType Nuclei Instance Segmentation 
# MAGIC <!-- - dataset -->
# MAGIC Specifically, we will _finetune_ the `YOLO_v11 Instance Segmentation` model on a new dataset, the [NuInsSeg Dataset](https://github.com/masih4/NuInsSeg/tree/main?tab=readme-ov-file#nuinsseg--a-fully-annotated-dataset-for-nuclei-instance-segmentation-in-he-stained-histological-images), _one of the largest publicly available datasets of segmented nuclei in [H&E-Stained](https://en.wikipedia.org/wiki/H%26E_stain) Histological Images_ (images below of the flow of processes illustrate how these sample data are typically derived).   
# MAGIC
# MAGIC ---     
# MAGIC
# MAGIC [<img src="https://raw.githubusercontent.com/masih4/NuInsSeg/main/git%20images/prepration.png" width="800"/>](https://raw.githubusercontent.com/masih4/NuInsSeg/main/git%20images/prepration.png) 
# MAGIC
# MAGIC [<img src="https://raw.githubusercontent.com/masih4/NuInsSeg/main/git%20images/segmentation%20sample.jpg" width="800"/>](Segmentation Sample](https://raw.githubusercontent.com/masih4/NuInsSeg/main/git%20images/segmentation%20sample.jpg) 
# MAGIC
# MAGIC ---     
# MAGIC
# MAGIC We will run the _finetuning_ on the Databricks Intelligence platform using [serverless compute](https://www.databricks.com/glossary/serverless-computing). 
# MAGIC
# MAGIC <!-- preprocessed in workspace folder -->
# MAGIC To focus our example on the application of YOLO Instance Segmentation, we have already pre-processed the [NuInsSeg Dataset](https://github.com/masih4/NuInsSeg/tree/main?tab=readme-ov-file#nuinsseg--a-fully-annotated-dataset-for-nuclei-instance-segmentation-in-he-stained-histological-images) images in [YOLO format](https://docs.ultralytics.com/datasets/segment/) and included them within the `datasets` folder within the workspace path where this notebook resides.    
# MAGIC
# MAGIC Along with the `datasets`, we also have information on how the data is organized within the corresponding `data.yaml`. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### What this notebook walks you through:  
# MAGIC **[1] YOLO continued training on serverless compute (SGC) using multipe remote GPUs    
# MAGIC [2] Log Model itself and the related paramters/metrics/artifacts with [Databricks managed MLflow](https://www.databricks.com/product/managed-mlflow)**
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC > - This notebook is an extension example of how to run YOLO Instance Segmentation on a custom dataset with **remote multiple serverless gpu (A10) compute**. 
# MAGIC > - For single SGC gpu run with more granular details like customizing your per-epoch logging artifacts via callbacks, see the previous **"01_CellTypes_InstanceSeg_TransferLearn_sgcA10"** notebook.  
# MAGIC > - The example solution will be part of the assets within the forthcoming [databricks-industry-solutions/cv-playground](https://github.com/databricks-industry-solutions/cv-playground) that will show case other CV-related solutions on Databricks.   
# MAGIC > - Developed and last tested [`2025Dec`] using remote multiple `sgc_A10` and `AI_v4` by `yang.yang@databricks.com`
# MAGIC `@distributed(gpus=8, gpu_type='A10', remote=True)`
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---    

# COMMAND ----------

# DBTITLE 1,Setup Widgets for User to setup their own I/O path for the project
## replace with your specific catalog and schema etc. names in the widgets panel above.
dbutils.widgets.text("CATALOG_NAME", "yyang", "Catalog Name")
dbutils.widgets.text("SCHEMA_NAME", "computer_vision", "Schema Name")
dbutils.widgets.text("VOLUME_NAME", "projects", "Volume Name")

# COMMAND ----------

# DBTITLE 1,Pinned Dependencies
import serverless_gpu
%pip install -U mlflow>=3.0
%pip install threadpoolctl==3.1.0
%pip install ultralytics==8.3.204
%pip install nvidia-ml-py==13.580.82 # for later mlflow GPU monitoring
%pip install pyrsmi==0.2.0 # for later mlflow AMD GPU monitoring if you have AMD

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,check dependencies version
import importlib.metadata as meta

versions = {}
packages = [
    'ultralytics',
    'torch',
    'mlflow',
    'scikit-learn',
    'matplotlib',
    'nvidia-ml-py',
    'threadpoolctl'
]

for pkg in packages:
    try:
        versions[pkg] = meta.version(pkg)
    except Exception:
        try:
            mod = __import__(pkg.replace('-', '_'))
            versions[pkg] = getattr(mod, '__version__', 'Not found')
        except Exception:
            versions[pkg] = 'Not found'

for pkg, ver in versions.items():
    print(f"{pkg:15}: {ver}")


### check wrt pinned depdendencies
# ultralytics    : 8.3.200
# torch          : 2.6.0+cu124
# mlflow         : 2.21.3
# scikit-learn   : 1.7.2
# matplotlib     : 3.10.7
# nvidia-ml-py3  : 7.352.0
# threadpoolctl  : 3.6.0    

# COMMAND ----------

# DBTITLE 1,Load Library
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


from ultralytics import YOLO
from serverless_gpu import distributed
import mlflow


import os
from ultralytics import YOLO
import torch
import mlflow
import torch.distributed as dist
from ultralytics import settings
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.signature import ModelSignature
from ultralytics.utils import RANK, LOCAL_RANK

# COMMAND ----------

# DBTITLE 1,check local temp folder Ultralytics settings.json
# MAGIC %cat /tmp/Ultralytics/settings.json

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,PATH NAMES
#: these variables inherited from widgets panel on the top.
CATALOG_NAME = dbutils.widgets.get("CATALOG_NAME")
SCHEMA_NAME = dbutils.widgets.get("SCHEMA_NAME")
VOLUME_NAME = dbutils.widgets.get("VOLUME_NAME")

PROJECT_NAME = "NuInsSeg"

## Volumes path prefix
VOLUME_PATH = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/"
# volume
PROJECTS_DIR = f"{VOLUME_PATH}/{VOLUME_NAME}"
# folder under volume
PROJECT_PATH = f"{PROJECTS_DIR}/{PROJECT_NAME}"
# subfolder
YOLO_DATA_DIR = f"{PROJECT_PATH}/yolo_dataset" # can update this to change the path to your own data 

## local VM storage /tmp/
tmp_project_dir = f"/tmp/{PROJECT_NAME}"

## Get the current working directory
nb_context = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
current_path = f"/Workspace{nb_context}"
# print(f"Current path: {current_path}")
WS_PROJ_DIR = '/'.join(current_path.split('/')[:-1]) 

WORKSPACE_PATH = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
USER_WORKSPACE_PATH = f"/Users/{WORKSPACE_PATH}"


### we need to defined experiment_name when starting mflow...
project_name = "yolo_CellTypesNuclei_InstanceSeg_scg"
# experiment_name = f"{USER_WORKSPACE_PATH}/mlflow_experiments/yolo/{project_name}"
experiment_name = f"{WS_PROJ_DIR}/{project_name}"
# os.makedirs(experiment_name, exist_ok=True)  # won't work on serverless
mlflow.set_experiment(experiment_name)
print(f"Setting experiment name to be {experiment_name}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- CREATE CATALOG IF NOT EXISTS ${CATALOG_NAME};
# MAGIC CREATE SCHEMA IF NOT EXISTS ${CATALOG_NAME}.${SCHEMA_NAME};
# MAGIC CREATE VOLUME IF NOT EXISTS ${CATALOG_NAME}.${SCHEMA_NAME}.${VOLUME_NAME};

# COMMAND ----------

# DBTITLE 1,check paths
# List of paths to check/create (Volumes)
paths_to_check = [
    PROJECTS_DIR,
    PROJECT_PATH,    
    YOLO_DATA_DIR
]

def path_exists(path):
    try:
        dbutils.fs.ls(path)
        return True
    except Exception:
        return False

for path in paths_to_check:
    if not path_exists(f"{path}"):
        print(f"{path}", path_exists(f"{path}"))
        dbutils.fs.mkdirs(f"{path}")
    else:
        print(f"{path}", path_exists(f"{path}"))    

## Alternatively
# display(dbutils.fs.ls(f"{PROJECTS_DIR}"))     

# COMMAND ----------

PROJECT_PATH

# COMMAND ----------

# MAGIC %md
# MAGIC --------------------------

# COMMAND ----------

# DBTITLE 1,preprocessed DATA in YOLO format
# MAGIC %ls -lah {WS_PROJ_DIR}/datasets/ 

# COMMAND ----------

# DBTITLE 1,data.yaml specifying data paths
# MAGIC %cat data.yaml

# COMMAND ----------

# MAGIC %md
# MAGIC __Volume path for dataset is recommended__

# COMMAND ----------

# DBTITLE 1,(RUN Only Once)move .yaml and datasets folder to volume for better governance
# Copy data.yaml to UC volume
if not path_exists(f"{YOLO_DATA_DIR}/data.yaml"):
  dbutils.fs.cp(f"file:{WS_PROJ_DIR}/data.yaml", f"{YOLO_DATA_DIR}/data.yaml")

# Copy datasets folder to UC volume (recursive)
if not path_exists(f"{YOLO_DATA_DIR}/datasets"):
  dbutils.fs.cp(f"file:{WS_PROJ_DIR}/datasets", f"{YOLO_DATA_DIR}/datasets", recurse=True)

# COMMAND ----------

# DBTITLE 1,specify the data.yaml path under the volume
data_yaml_path = f"{YOLO_DATA_DIR}/data.yaml"

# COMMAND ----------

f"{YOLO_DATA_DIR}/datasets"

# COMMAND ----------

# MAGIC %md
# MAGIC > Note: **For better governance, we prefer to have datasets sit in the UC volume.**

# COMMAND ----------

# MAGIC %md 
# MAGIC # Multiple-GPU Training

# COMMAND ----------

# DBTITLE 1,Load related library
from ultralytics import YOLO
import torch
import mlflow
import torch.distributed as dist
from ultralytics import settings
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.signature import ModelSignature
from mlflow.models.signature import infer_signature

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Note: To infer the signature, we will need to load a model (off-the-shell), predict on a few images and get the output.**
# MAGIC
# MAGIC The format of the prediction output matters, not the quality at this time. We will not log anything specific.

# COMMAND ----------

# DBTITLE 1,load the best model
model = YOLO(f"yolo11n-seg.pt")

# COMMAND ----------

model.train(
                # task="detect",
                batch=8, # Batch size, with three modes: set as an integer (e.g., batch=16), auto mode for 60% GPU memory utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70). Note, with multi-GPU, only integer works. Others modes all throw errors.
                device=[-1], # need to be LOCAL_RANK, i.e., 0 for this case since we already init_process_group beforehand. RANK wont work. There is no need to specify [0,1] given for example if we have 2 GPUs per node. [0,1] with world_size of 4 or 2 beforehand will both fail. 
                data=data_yaml_path,
                epochs=2,
                # project=f'{tmp_project_location}', # local VM ephermal location
                # project=f'{volume_project_location}', # volume path still wont work
                #
                save=True,
                name="runs/segment/train_sgc", # workspace path relative to notebook to save run outputs
                project=WS_PROJ_DIR,
                exist_ok=True,
                #
                fliplr=1,
                flipud=1,
                perspective=0.001,
                degrees=.45
            )

# COMMAND ----------

# DBTITLE 1,input and output
example_image_path = [f"{WS_PROJ_DIR}/datasets/test/images/human_bladder_01.png", f"{WS_PROJ_DIR}/datasets/test/images/human_brain_9.png"]
predictions = model.predict(example_image_path)

# COMMAND ----------

predictions

# COMMAND ----------

# DBTITLE 1,to_pandas so easy to infer signature later
prediction = predictions[0]
pred_df = prediction.to_df().to_pandas()

# COMMAND ----------

pred_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manual ModelSignature Limitations
# MAGIC
# MAGIC **Key Limitation: Manual ModelSignature can only define flat schemas**
# MAGIC
# MAGIC When using MLflow's `ModelSignature` with manual `ColSpec` definitions, you are limited to **flat, primitive data types only**:
# MAGIC
# MAGIC ### Supported Types in Manual ColSpec:
# MAGIC - `boolean`, `integer`, `long`, `float`, `double`, `string`, `binary`, `datetime`
# MAGIC
# MAGIC ### **NOT Supported** in Manual ColSpec:
# MAGIC - ❌ Nested structures like `{"box": {"x1": 10.5, "y1": 15.2}}`
# MAGIC - ❌ Array types like `Array(double)` or `[1.0, 2.0, 3.0]`
# MAGIC - ❌ Complex objects with multiple levels of nesting
# MAGIC
# MAGIC ### Comparison:
# MAGIC
# MAGIC **`infer_signature()` (Automatic)**:
# MAGIC - ✅ Automatically detects nested structures from actual data
# MAGIC - ✅ Supports complex types like `Object` and `Array`
# MAGIC - ✅ Can handle DataFrames with dictionary columns
# MAGIC - ✅ Creates schemas like: `'box': {x1: double, x2: double, y1: double, y2: double}`
# MAGIC
# MAGIC **Manual `ModelSignature` with `ColSpec`**:
# MAGIC - ❌ Limited to flat, primitive columns only
# MAGIC - ❌ Cannot create nested `Object` or `Array` types
# MAGIC - ✅ Gives precise control over schema definition
# MAGIC - ✅ Better for production models with predictable, flat outputs
# MAGIC
# MAGIC ### Best Practice:
# MAGIC **For complex nested data**: Use `infer_signature()` with sample data
# MAGIC **For production models**: Flatten your data structure and use manual `ColSpec` for precise control
# MAGIC
# MAGIC ### Here:
# MAGIC **Here we will use `infer_signature()` since `pred_df` has nested structure.**
# MAGIC

# COMMAND ----------

signature = infer_signature(example_image_path, pred_df)

# COMMAND ----------

signature

# COMMAND ----------

# MAGIC %md
# MAGIC __since we will use `infer_signature`, here we skip the manual ModelSignature definition.__

# COMMAND ----------

# DBTITLE 1,(skip) Manual ModelSignature - No Exact Match
# # Create manual ModelSignature that exactly matches the inferred signature
# from mlflow.types.schema import Schema, ColSpec
# from mlflow.models.signature import ModelSignature
# from mlflow.types import DataType

# # Manual definition that exactly matches the inferred signature structure
# input_schema_exact = Schema([
#     ColSpec(DataType.string, "image_source")
# ])

# output_schema_exact = Schema([
#     ColSpec(DataType.string, "name"),
#     ColSpec(DataType.long, "class"),
#     ColSpec(DataType.double, "confidence"),
#     ColSpec(DataType.double, "x1"),
#     ColSpec(DataType.double, "x2"),
#     ColSpec(DataType.double, "y1"),
#     ColSpec(DataType.double, "y2"),
#     ColSpec(DataType.string, "segments_x"),
#     ColSpec(DataType.string, "segments_y")
# ])

# # Create the exact manual signature
# exact_manual_signature = ModelSignature(inputs=input_schema_exact, outputs=output_schema_exact)

# print("Original inferred signature:")
# print(signature)
# print("\n" + "="*60 + "\n")
# print("Manual signature (exact match):")
# print(exact_manual_signature)
# print("\n" + "="*60 + "\n")
# print("Verification - Are they identical?")
# print(f"Input schemas match: {signature.inputs == exact_manual_signature.inputs}")
# print(f"Output schemas match: {signature.outputs == exact_manual_signature.outputs}")
# print(f"Signatures are identical: {signature == exact_manual_signature}")

# COMMAND ----------

# DBTITLE 1,helper functions
def setup():
    """Initialize the distributed training process group"""
    # Check if we're in a distributed environment
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        # Fallback for single GPU
        rank = 0
        world_size = 1
        local_rank = 0

    # Initialize process group
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return rank, world_size, device
  
def cleanup():
    """Clean up the distributed training process group"""
    if dist.is_initialized():
        dist.destroy_process_group()

# COMMAND ----------

import serverless_gpu
from serverless_gpu import distributed

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup MLflow Experiment

# COMMAND ----------

# DBTITLE 1,Set MLflow Experiment Parameters
# We set the experiment details here
import os

os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"
print(f"MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING set to {os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING']}")

os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
print(f"MLFLOW_EXPERIMENT_NAME set to {os.environ['MLFLOW_EXPERIMENT_NAME']}")

experiment = mlflow.set_experiment(experiment_name)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Kick off Skyrun

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Data should be put under `/Volumes/` for better governance;
# MAGIC 2. `project` path in YOLO training argument should use local VM storage `/tmp/` (ephemeral) and later copy the folder back to `/Volumes/` after training finished.

# COMMAND ----------

# MAGIC %md
# MAGIC | Location | Supports seek/append? | Durable? | Multi‑node safe? | Notes |
# MAGIC |----------|------------------------|----------|------------------|-------|
# MAGIC | `/tmp` | **Yes** | No | No | Best for training I/O; ephemeral; fastest local SSD |
# MAGIC | `/dbfs` | Partial / unreliable | Yes | Yes | FUSE layer causes issues; not ideal for YOLO checkpoints |
# MAGIC | Workspace (Repos / Workspace files) | No | No | No | Not designed for training outputs; not durable |
# MAGIC | `/Volumes` | **No** | **Yes** | **Yes** | Best for final storage; object‑store backed; no seek/append support |

# COMMAND ----------

# DBTITLE 1,where we have data
data_yaml_path

# COMMAND ----------

# DBTITLE 1,where we used as training I/O path
tmp_project_dir

# COMMAND ----------

# DBTITLE 1,where we move artifacts from training I/O path to this final storage path
PROJECT_PATH

# COMMAND ----------

# DBTITLE 1,skyrun
settings.update({"mlflow":True}) # if you do want to autolog.
mlflow.autolog(disable = False)

print('data_yaml_path is:', data_yaml_path)

import logging
import shutil
logging.getLogger("mlflow").setLevel(logging.DEBUG)


@distributed(gpus=8, gpu_type='A10', remote=True)
#: -----------worker func: this function is visible to each GPU device.-------------------
def train_fn(world_size = None, parent_run_id = None):
    try:
        from ultralytics.utils import RANK, LOCAL_RANK

        # Setup distributed training
        rank, world_size, device = setup()

        print(f"Rank: {rank}, World Size: {world_size}, Device: {device}")
        print(f"Rank: {RANK}, World Size: {world_size}, Device: {LOCAL_RANK}")

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")


        ############################
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0" # use 1 for synchronization operation, debugging model prefers this.
        os.environ["NCCL_DEBUG"] = "INFO" # "WARN" # for more debugging info on the NCCL side.
        os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"
        os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
        # We set the experiment details here
        experiment = mlflow.set_experiment(experiment_name)
        print('data_yaml_path is:', data_yaml_path)
        
        #
        # with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run():
            model = YOLO(f"yolo11n-seg.pt")
            model.train(
                # task="detect",
                batch=8, # Batch size, with three modes: set as an integer (e.g., batch=16), auto mode for 60% GPU memory utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70). Note, with multi-GPU, only integer works. Others modes all throw errors.
                device=[LOCAL_RANK], # need to be LOCAL_RANK, i.e., 0 for this case since we already init_process_group beforehand. RANK wont work. There is no need to specify [0,1] given for example if we have 2 GPUs per node. [0,1] with world_size of 4 or 2 beforehand will both fail. 
                data=data_yaml_path,
                epochs=50,
                # project=WS_PROJ_DIR, # dont recommend since workspace shouldn't be used to store large data.
                # project=f'{volume_project_location}', # volume path still wont work due to only support atomic file operation
                project=f'{tmp_project_dir}', # local VM ephermal location
                #
                save=True,
                name="runs/segment/train_sgc", # workspace path relative to notebook to save run outputs
                exist_ok=True,
                #
                fliplr=1,
                flipud=1,
                perspective=0.001,
                degrees=.45
            )
            success = None
            if RANK in (0, -1):
                success = model.val()
                if success:
                    model.export() # ref: https://docs.ultralytics.com/modes/export/#introduction
            

        active_run_id = mlflow.last_active_run().info.run_id
        print("For YOLO autologging, active_run_id is: ", active_run_id)

        # after training is done.
        if not dist.is_initialized():
        # import torch.distributed as dist
            dist.init_process_group("nccl")

        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"------After training, we have: RANK:{global_rank=} -- LOCAL_RANK:{local_rank=} -- world_size: {world_size=}------")

        if global_rank == 0:
            with mlflow.start_run(run_id=active_run_id) as run:
                mlflow.log_artifact(data_yaml_path, "input_data_yaml")
                # mlflow.log_dict(data, "data.yaml")
                mlflow.log_params({"rank":global_rank})
                mlflow.pytorch.log_model(YOLO(str(model.trainer.best)), "model", signature=signature) # this succeeded
                #: TODO: we can log more stuff here
        
        return "finished" # can return any picklable object
    
    finally:
        # clean up
        cleanup()
        
        # copy back training artifacts to final storage under /Volumes/, it needs to be done on rank = 0
        if global_rank == 0:
            shutil.copytree(tmp_project_dir, PROJECT_PATH, dirs_exist_ok=True)


train_fn.distributed(world_size = None, parent_run_id = None) # now can program can run without specifying manually the parameters of world_size and parent_run_id. 

# COMMAND ----------

# MAGIC %md
# MAGIC ________

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Conclusions
# MAGIC
# MAGIC  ## 🎉 **Conclusions**  
# MAGIC  
# MAGIC  1. 🚀 **Distributed training** with YOLO and MLflow autologging was successfully set up using serverless GPUs.
# MAGIC  
# MAGIC  
# MAGIC  2. 💾 **Data storage best practices** were followed: ephemeral local storage (`/tmp`) for training I/O, and durable object storage (`/Volumes`) for final artifacts.
# MAGIC  
# MAGIC  
# MAGIC  3. 🧬 **Model signature** was inferred automatically using `infer_signature()`, which supports complex nested output structures.
# MAGIC  
# MAGIC  4. ⚠️ **Manual ModelSignature** is limited to flat schemas and was not used due to the nested nature of prediction outputs.
# MAGIC  
# MAGIC  5. 📦 **Training artifacts and model outputs** were logged to MLflow, enabling experiment tracking and reproducibility.
# MAGIC  
# MAGIC  6. ✅ The workflow is **ready for production deployment** and further evaluation using MLflow's experiment analysis tools.
