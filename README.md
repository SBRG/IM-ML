# IM-ML
A machine learning workflow to predict gene regulon membership based on promoter sequence features, focusing on top-down regulons derived from an Independent Component Analysis (ICA) of the PRECISE E. coli RNAseq database.<br>
## What is Independent Component Analysis?
To learn about ICA, how ICA components are computed, and what they can tell you, please visit https://imodulondb.org/about.html
## Workflow outline
1. Generate SigmaFactor PSSMs<br>
2. Feature Matrix Generation (This generates a ~200MB file necessary for machine learning)<br>
3. Feature Engineering<br>
4. Machine learing: model training and hyperparameter optimization<br>
5. ArcA Direct Repeats motifs to improve model performance<br>
## Dependencies
The workflow depends on:<br>
1. bitome: https://github.com/SBRG/bitome<br>
2. pymodulon: https://github.com/SBRG/pymodulon<br>
3. DNAshapeR:https://github.com/TsuPeiChiu/DNAshapeR<br>
4. scikit-learn: https://scikit-learn.org/stable/ <br>
5. seaborn statistical data visualization:https://seaborn.pydata.org/index.html<br>
        
Recommended package versions are:<br>
  Python==3.8<br>
  seaborn==0.12.2<br>
  numpy==1.24.3<br>
  matplotlib==3.7.1<br>
 ## Citation
Qiu, S., Lamoureux, C., Akbari, A., Palsson, B. O., &amp; Zielinski, D. C. (2022). Quantitative sequence basis for the E. coli transcriptional regulatory network. https://doi.org/10.1101/2022.02.20.481200
