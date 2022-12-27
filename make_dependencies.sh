# git clone https://github.com/tensorflow/tensor2tensor.git
# Initialize submodule(s)
git submodule update --init --recursive

# First install tensor2tensor
cd tensor2tensor

# Resolve any dependencies for tensor2tensor
# pip install -r requirements.txt

# Install tensor2tensor
python setup.py install

# Now install MinAtar
cd ..
cd MinAtar

# Resolve any other dependencies for MinAtar
# pip install -r requirements.txt

# Install MinAtar
python setup.py install

# Now install Q-space
cd ..

# Resolve any other dependencies for Q-space
pip install -r requirements.txt

# Install Q-space
python setup.py install