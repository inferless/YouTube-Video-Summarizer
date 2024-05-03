from youtube_transcript_api import YouTubeTranscriptApi
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class InferlessPythonModel:
    def initialize(self):
        model_id = "NousResearch/Hermes-2-Pro-Llama-3-8B"
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)
        self.llm = LLM(model=model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def get_transcription(self,youtube_url):
        video_id = youtube_url.split("=")[1]
        transcript_text=YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]
        return transcript
        
    def format_template(self,transcript):
        messages = [
        {
        "role": "system",
        "content": 
        """You are a Senior NYT Reporter tasked with summarizing a youtube video.
            "You will be provided with a youtube video transcript.",
            "Carefully read the transcript a prepare thorough report of key facts and details.",
            "Provide as many details and facts as possible in the summary.",
            "Your report will be used to generate a final New York Times worthy report.",
            "Give the section relevant titles and provide details/facts/processes in each section."
            "REMEMBER: you are writing for the New York Times, so the quality of the report is important.",
            "Make sure your report is properly formatted and follows the <report_format> provided below.
             <report_format>
        ### Overview
        {give an overview of the video}

        ### Section 1
        {provide details/facts/processes in this section}

        ... more sections as necessary...

        ### Takeaways
        {provide key takeaways from the video}
        </report_format>
        """
        },
        {"role": "user", "content":transcript}]
        
        return messages
        
    def infer(self, inputs):
        youtube_url = inputs["youtube_url"]
        transcript = self.get_transcription(youtube_url)
        trimmed_text = self.tokenizer.decode((self.tokenizer.encode(transcript)[:7900]),skip_special_tokens=True)
        format_text = self.format_template(trimmed_text)
        print("---"*100,format_text)
        text = self.tokenizer.apply_chat_template(format_text,tokenize=False,add_generation_prompt=True)
        print("---"*100,text)
        result = self.llm.generate(text, self.sampling_params)
        
        result_output = [output.outputs[0].text for output in result]
        print("---"*100,result_output[0])
        
        return {"generated_summary": result_output[0]}

    def finalize(self):
        # Finalize resources if needed
        self.llm = None
