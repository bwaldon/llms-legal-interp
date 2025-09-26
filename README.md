# Large Language Models for Legal Interpretation

This repository contains materials for two works.

* **Large Language Models for Legal Interpretation? Don't Take Their Word for It**
* **Not Ready for the Bench: LLM Legal Interpretation Is Unstable And Uncalibrated to Human Judgments** 

It contains jupyter notebook for the Red-teaming study from the following article.
## Large Language Models for Legal Interpretation? Don't Take Their Word for It

The `colab_vllm.ipynb` notebook accompanies the below article:


>  Waldon, Brandon and Schneider, Nathan and Wilcox, Ethan and Zeldes, Amir and Tobia, Kevin, Large Language Models for Legal Interpretation? Don't Take Their Word for It (February 03, 2025). Georgetown Law Journal, Vol. 114 (forthcoming), Available at SSRN: https://ssrn.com/abstract=5123124



## Not Ready for the Bench: LLM Legal Interpretation Is Unstable And Uncalibrated to Human Judgments

`main.py` is the main script for the experiments, it uses vLLM and huggingface transformers to collect model judgments.
* `models.py` contains the code for inference and judgment extraction.
* `prompts.py` contains code from generating the prompts for each of the question variants from the source data.
* `analysis.ipynb` contains the analysis codee for robustness (Section 4.2)
* `correlation-analysis.Rmd` contains the R code for the human correlation analysis (Section 4.3)

### Replication 

Requirements are catalogued in `requirements.txt`.
The data used for the paper are available in the `data/` folder.