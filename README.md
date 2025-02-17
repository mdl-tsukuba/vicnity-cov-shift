
# Harnessing the Power of Vicinity-Informed Analysis for Classification under Covariate Shift

This repository contains the experimental code associated with the paper [Harnessing the Power of Vicinity-Informed Analysis for Classification under Covariate Shift](https://arxiv.org/abs/2405.16906). The main script, `newsim-sample-excess.py`, reproduces the results discussed in the paper.

To generate the figures shown in Figure 1, run the following commands:
```sh
python newsim-sample-excess.py --alpha 0.5 --tau 1 --fontsize 20 --tick_fontsize 20
python newsim-sample-excess.py --alpha 0.25 --tau 1 --fontsize 20 --tick_fontsize 20
python newsim-sample-excess.py --alpha 0.5 --tau 2 --fontsize 20 --tick_fontsize 20
python newsim-sample-excess.py --alpha 0.25 --tau 2 --fontsize 20 --tick_fontsize 20
```

