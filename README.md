# Project for my Master's essay based on the Black-Litterman Model - LLM project written by Youngbin Lee, Yejin Kim and Juhyeong Kim (https://github.com/youngandbin/LLM-BLM)
 - The main improvement to add is in the input data fed to LLMs. Instead of only using historical price data, we can enhance the input by including yearly and quarterly financial statements.
 
 - High level list of changes made to the original project:
    - Changed usage of yfinance to instead use dataset from McGill-Fiam Hackathon 2025 for returns data.
       - This dataset is monthly instead of daily. Therefore, the portfolio is rebalanced monthly instead of bi-weekly like originally.
    - Added yearly and quarterly financial statements data as input to LLMs.
    - Extended the time period of analysis from January 2015 to June 2025.
    - Used different versions of LLM models (gpt-4o-mini was ran on OpenAI servers and the others locally).
    - Added a new step in the pipeline: summarize_text_reports.py to summarize financial statements through GPT-4o-mini.
    - Changed name of some files to better reflect their purpose.
    - Reduced the number of requests to LLMs from 30 to 5 per stock per month to reduce costs and speed up the process. (We input more data so responses take longer to generate).
       - The dispersion of response is therefore computed on only 5 responses instead of 30.

 - Instructions to run the project from scratch:
    - Have ollama running locally with the 3 models (gemma3, qwen and llama) loaded and ready to receive requests.
    - Make sure you have an OpenAI API key
    - Run the full dataprep.ipynb notebook to prepare the returns data from the McGill-FIAM Hackathon 2025 dataset. (Assuming you have the datasets in the correct folders).
    - Run summarize_text_reports.py to generate summarized financial statements for each stock and each month. (Assuming you have the financial statements data in the correct folder).
    - Make sure you have csv files containing the SP500 constituents per year in a folder. In our case it is in the folder sp500-master/sp500-constituents and were extracted from the github repo: https://github.com/fja05680/sp500 
    - Run run.py for each LLM model to get their stock return predictions.
    - Run baselines.py to generate benchmark portfolios.
    - Run blacklitterman_weights.py for each LLM model to generate the Black-Litterman portfolio weights.
    - Run returns_from_weights.py for each LLM model to compute the returns from the Black-Litterman portfolio weights.
    - Run the full pf_perf_evaluation.ipynb notebook to generate performance metrics and plots.