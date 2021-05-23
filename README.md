# zono-reach-net
Neural network approach for generating zonotope reachable sets

## Setup
(Developed and tested on Windows with GPU support)
1. Make sure anaconda is setup. Open anaconda prompt
2. Create conda environment: `conda env create -f environment.yml`
3. Activate newly created environment: `conda activate reach-net`
4. Install cvxpy: https://www.cvxpy.org/install/
5. Install cvxpylayers: `pip install cvxpylayers`
6. Install (latest) pytorch with CUDA support: `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`

### Jupyter notebooks in VSCode
1. Open new anaconda prompt and activate environment
2. Run `code` to open VSCode 
3. In bottom left corner of VSCode, select interpreter `'reach-net': conda`
4. In upper right corner of VSCode, select kernel `'reach-net': conda`
  a. May need to go to Preferences > Search "path" and select "python" and add path to anaconda envs to "Venv Path"
