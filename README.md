### amul-mascot-girl-flux-t2i
Scripts for preprocessing , inferencing and finetuning the most popular [amul mascot girl](https://en.wikipedia.org/wiki/Amul_girl)  
- Used flux 1.1 dev model with lora (low rank adaption) for finetuning text to image generation

### preprocessing
https://github.com/sanjay7178/amul-mascot-girl-flux-t2i/blob/main/amul_mascot_girl_preprocess.ipynb
### dataset
https://huggingface.co/datasets/sanjay7178/amul-mascot-girl
### configuration
Install conda-forge or mamba-forge 
create virtual environment 
```bash 
conda create -n amul python=3.10 
conda activate amul     # change current env to amul
```
Linux:
```bash 
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python3 -m venv venv
source venv/bin/activate
# .\venv\Scripts\activate on windows
# install torch first
pip3 install torch
pip3 install -r requirements.txt
```
Windows:
```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python -m venv venv
.\venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
### finetuning
Before fine tuning change dataset path and some required vram hyperparameters according to your system requirements from `ostris-config/config.yml`

```bash
python run.py ostris-config.config.yml
```
### inference (gradio demo)
https://github.com/sanjay7178/amul-mascot-girl-flux-t2i/tree/main/lora-gradio-demo

### results
###### Loss Plot
<a href="url"><img src="https://github.com/user-attachments/assets/59177225-78cc-4963-9698-798aa5fdadfc" align="left" height="400" width="1000" ></a>

### benchmarks 
coming soon ..


