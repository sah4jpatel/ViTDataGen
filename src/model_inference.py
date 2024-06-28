from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

class ModelInference:
    def __init__(self, model_id='microsoft/Florence-2-large'):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def run_inference(self, image_path, classes):
        image = Image.open(image_path)
        task_prompt = '<OPEN_VOCABULARY_DETECTION> this image is captured onboard a blimp in an indoor arena. The evenet acts like a game that resembles blimp quiddich with balloons to capture (green and purple) and target goals to put the captured balloons through. The goals are circles, squares, or triangles that are either redish orange or yellow. However, a problem with this event is incomsistent lightning conditions and background noise with the colors of the arena and other blimps (red or blue). So there are be perceived colors in slightly different shades as a result.'
        text_input = " Look for the following in the image and classify them as: orange/red hollow circle, orange/red hollow triangle, orange/red hollow square, yellow hollow circle, yellow hollow triangle, yellow hollow square, light green balloon, light purple balloon, blue blimp, or red blimp."
        prompt = task_prompt + text_input
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )
        
        return self._convert_to_od_format(parsed_answer['<OPEN_VOCABULARY_DETECTION>'])

    @staticmethod
    def _convert_to_od_format(data):
        bboxes = data.get('bboxes', [])
        labels = data.get('bboxes_labels', [])
        return {'bboxes': bboxes, 'labels': labels}
