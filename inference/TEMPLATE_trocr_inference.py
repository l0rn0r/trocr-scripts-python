""" Module to inference a dataset with line images with a TrOCR model """

from typing import Tuple
from glob import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tqdm import tqdm

################# CONFIG #################
# define the paths
IMAGE_FOLDER = './img_def/'

# define the model and processor
TROCR_MODEL = 'microsoft/trocr-large-handwritten'
# some TrOCR models do not have a processor, so it's defined manually
TROCR_PROCESSOR = 'microsoft/trocr-large-handwritten'

# define the name of the dataset or model
NAME_INF = 'name_of_the_dataset_or_model'

# define the parameters for the image handling
DO_RESIZE = False
ASPECT_RATIO_RESIZE = False # not really reliable yet
################# CONFIG #################

# get the proper device to work
DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'

# if cuda is not available, check if mps is available
if DEVICE_NAME != 'cuda':
    DEVICE_NAME = 'mps' if torch.backends.mps.is_available() else 'cpu'

DEVICE = torch.device(DEVICE_NAME)

# functions
# custom collate function: stack images and file names
def custom_collate_fn(batch: list) -> dict:
    """
    Custom collate function to stack images and file name
    batch: list of dictionaries with keys 'pixel_values' and 'file_name'
    """
    pixel_values = [item['pixel_values'] for item in batch]
    file_names = [item['file_name'] for item in batch]

    # stack action
    pixel_values = torch.stack(pixel_values)

    return {'pixel_values': pixel_values, 'file_names': file_names}

# define TrOCR dataset class
class TrOCRInferenceDataset(Dataset):
    """
    TrOCR inference dataset class
    root_dir: root directory with the images
    file_names: list with the file names
    processor: TrOCRProcessor
    max_target_length: maximum target length (default: 256)
    """
    def __init__(
            self,
            file_names: list,
            processor: TrOCRProcessor,
            max_target_length: int=256,
            **kwargs,
            ):
        self.file_names = file_names
        self.processor = processor
        self.max_target_length = max_target_length
        self.do_resize = kwargs.get('do_resize', False)
        self.aspect_ratio_resize = kwargs.get('aspect_ratio_resize', False)
        self.image_size = kwargs.get('image_size', (384, 384))

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> dict:
        # get file name + text
        file_name = self.file_names[idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(file_name).convert('RGB')

        if self.do_resize \
            and self.aspect_ratio_resize \
                and (image.size[0] < self.image_size[0] or image.size[1] < self.image_size[1]):
            image = self.resize_with_aspect_ratio(image=image, target_size=self.image_size)
        elif self.do_resize:
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)

        pixel_values = self.processor(image, return_tensors='pt').pixel_values

        encoding = {'pixel_values': pixel_values.squeeze(), 'file_name': file_name}
        return encoding

    # used, if images are not in the right format
    def resize_with_aspect_ratio(
            self,
            image: Image,
            target_size: Tuple[int, int] = (384, 384),
            ) -> Image:
        """
        Resize the image while maintaining the aspect ratio
        image: PIL image
        target_size: target size (default: (384, 384))
        """
        image.thumbnail(target_size, Image.Resampling.LANCZOS)

        # create a new image with a white background and paste the resized image into it
        padded_image = ImageOps.pad(
            image,
            target_size,
            method=Image.Resampling.LANCZOS,
            color=(1, 1, 1)
            )
        return padded_image

# load model function
def load_model_processor(
        model: str,
        processor: str = None
    ) -> Tuple[VisionEncoderDecoderModel, TrOCRProcessor]:
    """
    Load the model and processor
    model: model name, has to be a TrOCR model (local or from Hugging Face)
    processor: processor name, has to be a TrOCR processor (local or from Hugging Face)
    """
    if processor:
        processor_loaded = TrOCRProcessor.from_pretrained(processor)
    else:
        processor_loaded = TrOCRProcessor.from_pretrained(model)

    model_loaded = VisionEncoderDecoderModel.from_pretrained(model)
    model_loaded.to(DEVICE)

    return model_loaded, processor_loaded

# run the inference
def run_batch_inference(
        inference_dataloader: DataLoader,
        model: VisionEncoderDecoderModel,
        processor: TrOCRProcessor,
        max_new_tokens: int=100
    ) -> float:
    """
    Run the inference
    inference_dataloader: DataLoader
    model: VisionEncoderDecoderModel
    processor: TrOCRProcessor
    max_new_tokens: maximum number of new tokens to generate (default: 100)
    """
    infered_txt = []

    # infere with the model
    print('Running inference...')

    for batch in tqdm(inference_dataloader):
        # predict using generate
        pixel_values = batch['pixel_values'].to(DEVICE)
        outputs = model.generate(pixel_values, max_new_tokens=max_new_tokens)

        # decode
        pred_str = processor.batch_decode(outputs, skip_special_tokens=True)
        # sometimes the text is not decoded properly
        # pred_str = [bytes(t, 'utf-8').decode('unicode_escape') for t in pred_str]

        file_names = batch['file_names']
        line = [f'{os.path.basename(file_name)}\t{pred}' for file_name, pred in zip(file_names, pred_str)]
        infered_txt.extend(line)

    return infered_txt

def inference(
        file_names: list,
        model: VisionEncoderDecoderModel,
        processor: TrOCRProcessor,
        **kwargs,
        ) -> float:
    """
    Run the evaluation
    file_names: list with the file names
    model: VisionEncoderDecoderModel
    processor: TrOCRProcessor
    name: name of the dataset or model
    image_folder: folder with the images of the lines
    do_resize: resize the images (default: False)
    aspect_ratio_resize: resize the images while maintaining the aspect ratio (default: False)
    max_new_tokens: maximum number of new tokens to generate (default: 100)
    """
    max_new_tokens = kwargs.get('max_new_tokens', 100)
    image_folder = kwargs.get('image_folder', './')
    do_resize = kwargs.get('do_resize', False)
    aspect_ratio_resize = kwargs.get('aspect_ratio_resize', False)

    inference_dataset = TrOCRInferenceDataset(
        root_dir=image_folder,
        file_names=file_names,
        processor=processor,
        do_resize=do_resize,
        aspect_ratio_resize=aspect_ratio_resize,
    )

    print('Number of lines to infere:', len(inference_dataset))

    inference_dataloader = DataLoader(
        inference_dataset,
        collate_fn=custom_collate_fn,
        batch_size=8,
        shuffle=False,
    )

    list_infered = run_batch_inference(
        inference_dataloader,
        model,
        processor,
        max_new_tokens=max_new_tokens,
    )

    return list_infered

def main():
    """
    Main function
    """
    # load the model and processor
    model, processor = load_model_processor(TROCR_MODEL, TROCR_PROCESSOR)

    # get the image lines
    image_files = []
    for ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tif', 'tiff']:
        image_files.extend(glob(f'{IMAGE_FOLDER}/*.{ext}'))
        image_files.extend(glob(f'{IMAGE_FOLDER}/*.{ext.upper()}'))

    # run the inference
    result = inference(
        image_files,
        model,
        processor,
        image_folder=IMAGE_FOLDER,
        do_resize=DO_RESIZE,
        aspect_ratio_resize=ASPECT_RATIO_RESIZE,
        max_new_tokens=100,
    )

    # save the results
    with open(f'{NAME_INF}_inference.txt', 'w', encoding='utf-8') as f:
        for line in result:
            f.write(f'{line}\n')

if __name__ == '__main__':
    print('Starting inference...')
    main()
