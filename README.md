# Voice Activity Detection system 

![Demo](./data/demo.gif)

Visual Voice Activity Detection (VAD) system based on PyTorch. The system is based on the output of [3DI](http://github.com/computational-psychiatry/3DI).

The method takes the canonical landmarks produced by 3DI and detects moments where the person in the video is speaking and moments where the person is silent.

## Quickstart

You can immediately test the method by running

```bash
python demo.py
```

## Installation
It is advised that you create and run a virtual environment as follows
```bash
python3 -m venv env
source ./env/bin/activate
```

The required pip packages can be installed by running

```bash
pip install -r requirements.txt
```
## Testing
The code can be run on a single video through the script ```VAD.py``` which has three required parameters as follows

```
python VAD.py --file_lmks=#LMKS_FILE# --file_video_in=#INPUT_VIDEO# --file_video_out=#OUTPUT_VIDEO#
```

As seen, the code requires three arguments: The canonical landmarks file produced by 3DI (see [here](https://github.com/computational-psychiatry/3DI#output-formats)), the input video path and the output video path. Example command is:
```bash
python VAD.py --file_lmks=./data/test/input/CNN1.canonical_lmks --file_video_in=./data/test/input/CNN1.mp4 --file_video_out=./data/test/CNN1_output.mp4
```

There is also a script to run two videos  from a dyadic interaction (`VAD_dyadic.py`) and merge the produced results (e.g., see video on top of this README file). To run a demo of `VAD_dyadic.py`, you can run
```bash
python demo.py
```

## Training
To perform training, you first need to download the pre-processed training data from this link: (https://sariyanidi.com/wp-content/uploads/2023/11/VAD_train_data.zip). This dataset contains canonicalized landmarks files obtaind by processing the (publicly available) Visual Voice Activity Detection dataset released by [Guy et al. (INRIA).](https://team.inria.fr/perception/research/vvad/).

Then, you need to unzip the data into the `data` folder so that the following directories are created and populated:
```bash
./data/VAD_train_data/raw/pos_class
./data/VAD_train_data/raw/neg_class
```

Then, you need to (once) run the following script to prepare the data for training:
```bash
python create_training_data.py
```

Finally, you can simply run the `train.py` script as
```bash
python train.py
```

Your trained model should be saved to the directory `./models/checkpoints`.


## Acknowledgments
We thank Guy et al. from [INRIA](https://team.inria.fr/perception/research/vvad/) for collecting and publicly releasing the Visual Voice Activity Detection dataset.
