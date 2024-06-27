# Create conda env
#conda create -n lm_decoder python=3.9 conda-forge::gcc
#conda activate lm_decoder

# Clone repository
git clone https://github.com/fwillett/speechBCI.git

# Install NeuralDecoder
cd speechBCI/NeuralDecoder
python -m pip install -e .

# Install LanguageModelDecoder
cd ../LanguageModelDecoder/srilm-1.7.3
export SRILM=$PWD
make -j8 MAKE_PIC=yes World
make -j8 cleanest
cd ../runtime/server/x86
mkdir build
cd build
cmake ..
make -j8
cd ..
python setup.py install

