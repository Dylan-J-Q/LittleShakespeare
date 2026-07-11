# LittleShakespeare - Pushed to its limits

[CI badge] [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.12.10](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-31213/)

## Demo
Prompt: To be, or not to be:

holy mon.

SISABELLA:
So with were what Ifear.

ANGELO:
Ha good ct me finness that the wdyself.

ANGELO:
Whad to mustingman?

ISABELO:
SABELLA:
I hat.

ISABELLA:
So strum, my lords it dection.

ISABELLA:
Hay, sir.

ISABELLA:
Yought son
Thout ailkillain, you to your grave you your prant.

ISABELLA:
O, my lord.

ISABELLA:
I'll not your your good my lor,
ABELLA:
Ifear meancan it is your gray you your shand
That t.

ISABELLA:
ABELLA:
How?

ISABELLA:
O, Iness your your your it, thenerttainst good osonours to the father,
Witherrongumost be how,
Aladyryety,
To have you make it aba ed, as your hand
What you ves de't.

ISABELLA:
SABELLA:
O, that, by your your sa scuch so you dis'ty lord.

ISABELLA:
How it. Hay, sir; and band
ABELLA:
O, sir, th hall you leser blook liveven spper we thinke
Hows of the t's de donden swhat you leseech you
To have may father sper sheak you,
Nor it. Thanceit ink your your have sa mintter give your your sper
As ince to bbtrich your for your y,
Whent what I ds dench ark for your your your desscrany litlong as you ve.

LABELO:
ABELO:
GELLA:
How cantain,
As you hatold a genten, ter for for you.

ANGELO:
SABELO:
GELO:
Yous I much are a day.

ISABELLA:
Oratess, I'ter prove! it is your so lest
By you are you have madodisdy
BELO:
As could you not you speavengaintend
That you your fore you comen you,
Have you sotopartenty dogoo ttrand, I have sotwor your prave.

ISABELO:
I have no more to you me, sir, sir.

ANGELO:
ISABELLA:
We you your your hat.

ISABELLA:
ABELO:
We pitippivoy.

ISABELLA:
ABELO:
Who you your gention.

ANGELO:
We your your by you se; I have you your your gray.

ANGELLA:
O, sir, werviva wor to--morter you your your it will I
Wither your your your your your your gent.

ISABELO:
I know you
Whater to good you your it not be preving.

ISABELLA:
I'll you your headenator your sudence?

ISABELLA:
Ory you your dices; and you have not have ave, you.

ANGELO:
Thance to your sewere come, madeep

## Why This Project
This project is an attempt to push a small transformer trained on shakespeares works to it's limit. The project constraints are hardware based, with all the training being completed on a system with a Ryzen 7800x3d CPU, NVIDIA 4070 super GPU (12GB of VRAM) and 32GB of RAM. The goal is to improve generation accuracy and token generation speed as much as possible within these constraints.

## Architecture
The architecture of the transformer currently consists of:
1. Embedding Layer
2. Sinusoidal Positional Encoding
3. All the Transformer Blocks 
4. Layer Norm
5. Output layer (linear layer)

The transformer blocks consist of:
1. Layer Norm 1
2. Multi Head Attention
3. Residual Connection
4. Layer Norm 2
5. Feed Forward Network
6. Residual Connection

## Repo Structure


## Installation
Use the requirements.txt to install all the dependencies. You may need to visit pytorch's website to download a compatible version with your system.

## Usage


## Benchmarks


## Testing & CI


## Roadmap


## License
MIT license as seen in [License](/LICENSE)