# Clone repos
git clone https://github.com/cffan/neural_seq_decoder.git
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

# Install NeuralDecoder of neural_seq_decoder
cd ../../../../neural_seq_decoder
python -m pip install -e .