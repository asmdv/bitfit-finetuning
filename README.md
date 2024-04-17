# Fine-Tuning with BitFit

This repo contains the code for fine-tuning BERT tiny model using [BitFit](https://aclanthology.org/2022.acl-short.1/).

|  Bitfit  | # of Params      | Accuracy |
| -------- | ---------------- | -------- | 
| Off      | 4,386,178 (100%) | 0.86636  |
| On       | 3,330 (0.08%)    | 0.7392   |

In my experiment, conventional hyperparameter tuning outperformed BitFit on the BERT-tiny model by approximately 12% in accuracy. However, BitFit operates with only 0.08% of the parameters—a reduction of nearly 1,000-fold. According to the findings of Zaken et al. (2020), BitFit achieves comparable or even superior performance to full tuning on the GLUE benchmark, particularly in low-data scenarios. As data volume increases, however, full tuning generally proves more effective. In this case, the superior accuracy of full tuning can likely be attributed to the use of the BERT-tiny model, as opposed to the larger models analyzed by Zaken et al., as well as the substantial size of the IMDB dataset, which favors full tuning. Ultimately, BitFit stands out as an efficient method for quickly fine-tuning pretrained models, requiring far fewer resources with only a modest trade-off in accuracy. This highlights the intriguing role of bias parameters in language models, demonstrating how bias tuning can effectively facilitate new task learning in pretrained models.