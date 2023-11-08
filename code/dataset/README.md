# NSFW DATASET:

1. https://github.com/EBazarov/nsfw_data_source_urls/tree/master

Requested access for the PEDA 376K dataset from the authors of [PEDA 376K: A Novel Dataset for Deep-Learning
Based Porn-Detectors](https://ieeexplore.ieee.org/document/9206701)
Request was denied due to copyright issues.


## Action Plan:
<strike>
1. Download the images from above mentioned repository and clean the images(few of these are expired links so would have to handle that)
</strike>

2. Train initial model


## Image Classification Approximation from GCP results.

1. Adult: [(VERY_LIKELY,UNLIKELY,UNLIKELY,UNLIKELY,UNLIKELY),(VERY_LIKELY,VERY_LIKELY,UNLIKELY,UNLIKELY,VERY_LIKELY),(VERY_LIKELY,UNLIKELY,UNLIKELY,UNLIKELY,VERY_LIKELY)] all other possibilities other than Likely 
2. Medical: [(Anything_but_likely,VERY_LIKELY,Anything_but_likely,Anything_but_likely,Anything_but_likely)]
3. spoofed: [(Anything_but_likely,Anything_but_likely,VERY_LIKELY,Anything_but_likely,Anything_but_likely)]
4. violence: [(Anything_but_likely,Anything_but_likely,Anything_but_likely,VERY_LIKELY,Anything_but_likely)]
5. racy: [(Anything_but_likely,VERY_LIKELY/LIKELY/POSSIBLE,VERY_LIKELY,Anything_but_likely,VERY_LIKELY)]