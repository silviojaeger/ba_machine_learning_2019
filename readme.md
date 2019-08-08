# Installation (Windows)
* ``` git clone https://github.com/silviojaeger/ba_machine_learning_2019 ```
* ``` cd .\ba_machine_learning_2019 ```
* ``` setup.ps1 ```
* Setup VSCode (Optional)
  * Open Folder ``` ba_machine_learning_2019 ```
  * Select Interpreter: 
    <kbd>command</kbd> + <kbd>shift</kbd> + <kbd>P</kbd>
    execute ``` Python: Select Interpreter ``` 
    select ``` .\env\Scripts\python.exe ```
  * Activate env (Optional for terminal commands): 
    ``` .\activate.ps1 ```

# Upgrade environment
* ``` activate.ps1 ```
* ``` update_requirements.ps1 ```
* ``` python -m pip install <package> ```
* Commit changes

# Create a new environment
* Copy following files to a new foder .\<newEnv>\:
  * ``` setup.ps1```
  * ``` update_requirements.ps1 ```
  * add ```.\<newEnv>\env``` to ```.gitignore ```
  * 
# Install graphviz (used to visualize the keras graph)
* Download and install the Windows package from here:
    https://graphviz.gitlab.io/_pages/Download/Download_windows.html

# Install CUDA library(used to train the models on your GPU)
* Update your graphics-card driver: 
    https://www.nvidia.de/Download/index.aspx?lang=de
* Install CUDA 10.0: 
 https://developer.nvidia.com/cuda-10.0-download-archive
* Download cudnn and safe it in ``` C:\tools``` (you have to create a free NVIDIA-account for this): 
 https://developer.nvidia.com/cudnn
* Add the following paths to the system %PATH%
    *``` C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin``` 
    *``` C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\libx64``` 
    *``` C:\tools\cuda\bin``` 
* to make it run on a system with cpu's that does'nt support AVX (like the NTB remote-computer) you have to install tensorflow the following:
    * Download this wheel file: https://drive.google.com/open?id=1dWlAqVqcCZmH3q3vefo8ohVhDxBTRp0H
    * Make sure tensorflow is not installed (```pip uninstall tensorflow```)
    * Install the tensorflow wheele: ```pip install /PATH-TO-MY-FILE/tensorflow-1.12.0-cp36-cp36m-win_amd64.whl```
* More details: https://www.tensorflow.org/install/gpu
