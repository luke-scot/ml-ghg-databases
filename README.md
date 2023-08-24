<pre><img width="195" alt="Refficiency logo" src= https://www.refficiency.org/wp-content/uploads/2018/05/Refficiency-04-1-e1474497018375.png>  <img width="400" alt="Cam logo" src= https://www.cam.ac.uk/sites/www.cam.ac.uk/files/inner-images/logo.jpg>   <img width="105" alt="AI4ER logo" src= https://avatars.githubusercontent.com/u/55584824?s=200&v=4>  </pre>

## Machine learning for gap-filling in greenhouse gas emissions databases
This repository contains 3 folders of notebooks:
- `data_formatting` - Notebooks for extracting each of the UNFCCC, ClimateTRACE and petrochemicals datasets from raw files. If you have downloaded the preprocessed files from https://doi.org/10.5281/zenodo.8279939 preprocessing is not necessary. `dataset_preparation.ipynb` takes the preprocessed data and creates ML ready datasets that can be used by the notebooks in the `model_run` folder.
- `model_run` - Notebooks for running each type of model on the datasets.
- `output_generation` - Notebooks for producing output figures and metrics seen in the publication.
