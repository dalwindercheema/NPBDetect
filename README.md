# NPBDetect: A Neural network model to detect bioactivities from Antismash GBK files

NPBDetect is available in:
<table>
<tr>
<td><img src="https://edent.github.io/SuperTinyIcons/images/svg/powershell.svg" width="100" title="CLI"></td>
<td><img src="https://edent.github.io/SuperTinyIcons/images/svg/colaboratory.svg" width="100" title="Google Colab"></td>
<td><img src="https://edent.github.io/SuperTinyIcons/images/svg/docker.svg" width="100" title="Docker"></td>
</tr>
</table>

----

## Contents ##

* [Usage](#usage)
* [CLI](#cli-installation)
* [Google Colab](#google-colab)
* [Docker](#docker)

----


## Usage ##

1. Detect biosynthetic gene clusters (BGCs) with antiSMASH 7. The program generate a GBK file which NPBDetect uses to predict bioactivites. antiSMASH 7 is available online or could be setup locally.

   To use the online service, use this link
   
   ```https://antismash.secondarymetabolites.org```

   To setup antiSMASH 7 locally, please download the code `https://antismash.secondarymetabolites.org/#!/download` and read instructions to install it `http://docs.antismash.secondarymetabolites.org/install/`

   To generate GBK file with information about BGCs, we have using the following command:
   
   ```antismash```
   
2. NPBDetect is also available online and installed locally on a user's system. The online vesion uses Google Colaboratory while local version could be setup through conda environment and docker. 


## CLI installation ##

1. Setup conda environment

```conda create -n npbdetect python=3.10```

2. Install packages with pip

```
conda activate npbdetect
pip3 install pandas
pip3 install scikit-learn
pip3 install biopython
pip3 install torch torchaudio torchvision torchtext torchdata
```
3. Clone repository
   
```git clone git@github.com:dalwindercheema/NPBDetect.git```

5. Validate installation and test samples
   
```
python NPBDetect.py \
predict \
-v 1 \
--gbk test/BGC0000004.gbk \
--pred HC \
--out_dir outs/
```

----

----

## Google Colab ##
The online version with Google colab is available at:

*[NPBDetect@google-colab](https://colab.research.google.com/drive/12zXjqk1DFX8Ouv0rYSAWSoaOheluBys9#scrollTo=MiUQ7HHR2rC8)*

----

## Docker ##

The docker image is available at:

Instructions for docker are 

```
docker run --rm -it \
--volume <INPUT/OUTPUT_DIR>:/data \
npbdetect \
python NPBDetect.py \
predict \
--gbk /data/BGC0000004.gbk \
--pred HC \
--out_dir /data/test1 \
```

----

If you use the code, please cite the paper:

Hemant Goyat, Dalwinder Singh Sunaina Paliyal, Shrikant Mantri , *[Predicting biological activity from biosynthetic gene clusters using neural networks](https://www.biorxiv.org/content/10.1101/2024.06.20.599829v1.full.pdf))*, 2024.
