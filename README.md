# Sound Texture Synthesis Using Optimal Transport

Anonymous repo for DAFx 2026 submission.

## Overview

This repository contains audio examples and code for the paper *Sound Texture Synthesis Using Optimal Transport*. We provide synthesized outputs from three methods across ten textures, alongside the reference recordings.

## Listening Guide

For each texture, four audio files are provided:

- the original reference recording
- synthesis using Antognini et al. (ICASSP 2019)
- synthesis using Caracalla & Roebel (ICASSP 2020)
- synthesis using our proposed OT-based method

## Textures

`applause` · `bees` · `birds` · `crowd` · `fire` · `insects` · `rain` · `sink` · `static` · `wind`

## What to Listen For

- Preservation of temporal structure and fine details
- Fidelity to the spectral statistics of the original texture
- Diversity and avoidance of exact repetitions from the reference

## Acknowledgements

We would like to thank Dr. Antognini and Dr. Roebel for allowing us to use their reference textures and generated results for our quantitative comparison.

## Notes

- All audio is sampled at 16 kHz
- Authors and institutional affiliation withheld for double-blind review

The snippets, PANN model weights, and textures used for FAD and Diversity tests are given at the following google drive link (https://drive.google.com/file/d/1N_QTBDJD4DcFJq0D9nDd91ShDf0mnhBn/view?usp=sharing). There are approximately > 4000 snippets, so this would not fit in github.
