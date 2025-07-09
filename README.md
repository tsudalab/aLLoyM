# aLLoyM

## Reproduce similar research
View [src/make_QA](https://github.com/tsudalab/aLLoyM/tree/main/src/make_QA) to make similar datasets, [src/mistral](https://github.com/tsudalab/aLLoyM/tree/main/src/mistral) to finetune and generate on Mistral (LLM), [src/score](https://github.com/tsudalab/aLLoyM/tree/main/src/score) to score them. There are README files in each of the directories.

## To add more phase diagram data
Add your .dat data under dataset/CPDDB_data and run run_all.sh
Can only take in phase names from phase_list in config.py

## Run all
chmod +x run_all.sh
./run_all.sh
