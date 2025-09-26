# Large Language Models for Legal Interpretation

This repository contains materials for two works.

* **Large Language Models for Legal Interpretation? Don't Take Their Word for It**
* **Not Ready for the Bench: LLM Legal Interpretation Is Unstable And Uncalibrated to Human Judgments** 

It contains jupyter notebook for the Red-teaming study from the following article.
## Large Language Models for Legal Interpretation? Don't Take Their Word for It

The `colab_vllm.ipynb` notebook accompanies the below article:


>  Waldon, Brandon and Schneider, Nathan and Wilcox, Ethan and Zeldes, Amir and Tobia, Kevin, Large Language Models for Legal Interpretation? Don't Take Their Word for It (February 03, 2025). Georgetown Law Journal, Vol. 114 (forthcoming), Available at SSRN: https://ssrn.com/abstract=5123124



## Not Ready for the Bench: LLM Legal Interpretation Is Unstable And Uncalibrated to Human Judgments

Most of the files in the repository correspond to the experiments from the following article:
> Purushothama, A., Min, J., Waldon, B., & Schneider, N. (2025). Not ready for the bench: LLM legal interpretation is unstable and out of step with human judgments (No. arXiv:2510.25356). arXiv. https://doi.org/10.48550/arXiv.2510.25356


Our code represents inference and analysis pipeline for the experiments
`main.py` is the main script for the inference, it uses vLLM and huggingface transformers to collect model judgments.
* `models.py` contains the code for inference and judgment extraction.
* `prompts.py` contains code from generating the prompts for each of the question variants from the source data.
* `analysis-robustness.ipynb` contains the analysis codee for robustness (Section 4.2)
* `correlation-analysis.Rmd` contains the R code for the human correlation analysis (Section 4.3)


`runs/runs-42_07_16` contains the model outputs for each of the models with labels for the variants.

*Warning* : 
A confusing naming convention we have is that we use `_probs` but they actually have logprobs and are exponentiated in the analysis code.
Soon in the future we will change this to `_logprobs` to avoid confusion.

### Replication 

Requirements are catalogued in `requirements.txt`.
The data used for the paper are available in the `data/` folder.