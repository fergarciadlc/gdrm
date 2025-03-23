# gdrm

gdrum, short for GAN drum, is a bar-level drum sequencer. Given a parameters like BPM and genre, the system will create a bar in 4-4 that matches your needs. You can increase or decrease variability per bar, force a dramatic change, and even explore mixing genres.

## Installing:
To install the system, clone the repository, and create your python environment in python=3.10. Once in your environment, install project dependencies with `pip install -r requirements.txt`.

## Dataset:
downloaded the [groove dataset](https://magenta.tensorflow.org/datasets/groove#download) and place it in the root of this repository.

This dataset contains many drumming performances in MIDI, with accompanying tempo and genre.

## Training:
#### **Preprocessing:**
You need to first pre-process the dataset with the following command. This shouldn't take too long.
```bash
python src/preprocess.py
```
This step iterates through the dataset, splitting and quantising the performances into separate bars, and save them as numpy arrays in a separate folder.

#### **Training:**
Once the dataset is preprocessed, we can begin the training. You can do so like this:
```bash
python src/train.py
```


If you'd like to resume training from some checkpoint, you can run instead:
```bash
python src/train.py --resume_epoch <epoch_num>
```
This will automatically find the checkpoints it needs and load them in.

#### **Monitoring progress:**
You can monitor the progress of the training using tensorboard. You can fire it up using this command:
```bash
tensorboard --logdir=./runs/gdrm_experiment
```


## Playing the sequencer:


## The GAN:
The discriminator is trained as a critic, using the wasserstein distance to evaluate error metrics.
