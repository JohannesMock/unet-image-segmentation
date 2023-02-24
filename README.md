# UNET Image Segmentation

## Image Segmentation of Vehicle Driving Situations

&nbsp; &nbsp;  &nbsp; &nbsp; &nbsp;  &nbsp;  &nbsp; &nbsp;  &nbsp; &nbsp; <img src="https://github.com/JohannesMock/unet-image-segmentation/blob/main/output_video_weighted.gif" width="400" class='center'/>

## Setup virtual environment
```bash
$ cd ~
$ python3.9 -m venv <environment-name>
$ source <environment-name>/bin/activate
$ pip install -r <path-to-repo>/requirements.txt
```

## File Structure
### data/
* **Images** : Raw Image Data
* **Labels** : Labels as RGB Images
* **Labels_int** : Labels transfered as Integer Classes
* **Test** : One Test Driving Session, used to create an Video to visualize the Segmentation
* **David_Description.pdf** : Paper of the used Dataset

### presentation/
* **data_visualization** : Exemplaric Images and Histogram of the Occurrence of the Labeled Classes
* **evaluation** : Plots to illustrate the different Results of each Model

### src/
* ```transform_data.ipynb``` : Transfer RGB Labels of the Dataset into Integer Classes
* ```alexnet_transfer_learning.ipynb``` : Train Unet with Alexnet as Encoder
* ```resnet18_transfer_learning.ipynb``` : Train Unet with Resnet18 as Encoder
* ```shufflenet_transfer_learning.ipynb``` : Train Unet with Shufflenet as Encoder
* ```model_architecture.py``` : Decoder for the Resnet18
* ```evaluate_models.ipynb``` : Notebook to compare the Results of the different Models
* ```create_video.ipynb``` : Notebook to visualize the Segmentation Results as Video
* **models** : Already Trained Models to reload and use them without retraining


## About the Authors
Johannes Mock  
Simeon Grossmann  


## References
The used Dataset is provided by:
* https://mediatum.ub.tum.de/1596437


The Decoder used for the Resnet18 Model is taken from:
* https://github.com/jarvislabsai/siim-acr


## License
GNU GENERAL PUBLIC LICENSE Version 3 
