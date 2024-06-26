# YouTube-Video-Summarizer
- This tutorial will walk you through creating a YouTube video summarizer application using cutting-edge serverless technologies. You’ll learn how to extract summaries from YouTube URLs by harnessing the power of the Llama-3 8B model.
## Architecture
<img width="1339" alt="image" src="https://github.com/inferless/YouTube-Video-Summarizer/assets/150957746/b375a194-67c1-4c5c-b064-f1247c4c5b4c">

---
## Prerequisites
- **Git**. You would need git installed on your system if you wish to customize the repo after forking.
- **Python>=3.8**. You would need Python to customize the code in the app.py according to your needs.
- **Curl**. You would need Curl if you want to make API calls from the terminal itself.

  ---
## Quick Start
Here is a quick start to help you get up and running with this template on Inferless.

### Fork the Repository
Get started by forking the repository. You can do this by clicking on the fork button in the top right corner of the repository page.

This will create a copy of the repository in your own GitHub account, allowing you to make changes and customize it according to your needs.

### Create a Custom Runtime in Inferless
To access the custom runtime window in Inferless, simply navigate to the sidebar and click on the Create new Runtime button. A pop-up will appear.

Next, provide a suitable name for your custom runtime and proceed by uploading the inferless-runtime.yaml file given above. Finally, ensure you save your changes by clicking on the save button.

### Import the Model in Inferless
Log in to your inferless account, select the workspace you want the model to be imported into and click the Add Model button.

Select the PyTorch as framework and choose **Repo(custom code)** as your model source and select your provider, and use the forked repo URL as the **Model URL**.

Enter all the required details to Import your model. Refer [this link](https://docs.inferless.com/integrations/github-custom-code) for more information on model import.

---
## Customizing the Code
Open the `app.py` file. This contains the main code for inference. It has three main functions, initialize, infer and finalize.

**Initialize** -  This function is executed during the cold start and is used to initialize the model. If you have any custom configurations or settings that need to be applied during the initialization, make sure to add them in this function.

**Infer** - This function is where the inference happens. The argument to this function `inputs`, is a dictionary containing all the input parameters. The keys are the same as the name given in inputs. Refer to [input](#input) for more.

```python
    def infer(self, inputs):
        youtube_url = inputs["youtube_url"]
        transcript = self.get_transcription(youtube_url)
        trimmed_text = self.tokenizer.decode((self.tokenizer.encode(transcript)[:7900]),skip_special_tokens=True)
        format_text = self.format_template(trimmed_text)
        text = self.tokenizer.apply_chat_template(format_text,tokenize=False,add_generation_prompt=True)
        result = self.llm.generate(text, self.sampling_params)
        result_output = [output.outputs[0].text for output in result]
        
        return {"generated_summary": result_output[0]}
```

**Finalize** - This function is used to perform any cleanup activity for example you can unload the model from the gpu by setting `self.pipe = None`.

For more information refer to the [Inferless docs](https://docs.inferless.com/).
