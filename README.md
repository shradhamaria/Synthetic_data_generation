# Synthetic Data Generation

Synthetic data generation package based on [BlenderProc](https://github.com/DLR-RM/BlenderProc).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Requirements

- BlenderProc 1
- Python
- OpenCV

## Installation

Install the Requirements..

Clone the Repository
> [BlenderProc](https://github.com/DLR-RM/BlenderProc)

> cd BlenderProc

Run the following command
> python scripts/download_cc_textures.py

## Instructions for Data

Download this repository inside the Blenderproc repository.

## Project Hierarchy
In order to run the following code, first set up the the hierarchy in a similar manner:

>     ├── BlenderProc
>       |── All the Blenderproc files/folders
>       |
>       ├── synthetic_data_generation
>       │   ├── generate_images.py
>       │   ├── config.yaml
>       │   ├── output
>       │   ├── object_files ├── 
>       │                      ├── object.obj
>       │                      ├── object.mtl
>       │                      ├── obj_scenenet.obj
>       │                      ├── obj_scenenet.mtl
>       │   ├── texture_library
>       

## Usage

    cd BlenderProc
    python rerun.py synthetic_data_generation/generate_images.py synthetic_data_generation/config.yaml synthetic_data_generation/object_files/object.obj synthetic_data_generation/object_files/scenenet.obj synthetic_data_generation/texture_library resources/cctextures resources/id_mappings/nyu_idset.csv synthetic_data_generation/texture_library/unknown synthetic_data_generation/output


**NOTE:** To customize the code, make sure to check config.yaml.
For example, if you want to generate 10 images, go to config.yaml and change the required parameter **"num_images"**.

**NOTE:** For multiple runs make sure to check rerun.py and change **"amount_of_runs"** parameter accordingly.

