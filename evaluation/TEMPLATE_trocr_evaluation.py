""" Module to evaluate a TrOCR model on a dataset with line images """

from typing import Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from evaluate import load
from tqdm import tqdm

################# CONFIG #################
# define the paths
GT_FILE_TEST = './gt.txt'
IMAGE_FOLDER = './img_def/'

# define the model and processor
TROCR_MODEL = 'microsoft/trocr-large-handwritten'
# some TrOCR models do not have a processor, so it's defined manually
TROCR_PROCESSOR = 'microsoft/trocr-large-handwritten'

# define the name of the dataset or model
NAME_EVAL = 'name_of_the_dataset_or_model'

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
# custom collate function: stack images and pad labels
def custom_collate_fn(batch: list) -> dict:
    """
    Custom collate function to stack images and pad labels
    batch: list of dictionaries with keys 'pixel_values' and 'labels'
    """
    pixel_values = [item['pixel_values'] for item in batch]
    labels = [item['labels'].clone().detach() for item in batch]

    # stack action
    pixel_values = torch.stack(pixel_values)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {'pixel_values': pixel_values, 'labels': labels}

# define TrOCR dataset class
class TrOCRDataset(Dataset):
    """
    TrOCR dataset class
    root_dir: root directory with the images
    df: dataframe with columns 'file_name' and 'text'
    processor: TrOCRProcessor
    max_target_length: maximum target length (default: 256)
    """
    def __init__(
            self,
            root_dir: str,
            df: pd.DataFrame,
            processor: TrOCRProcessor,
            max_target_length: int=256,
            **kwargs,
            ):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        self.do_resize = kwargs.get('do_resize', False)
        self.aspect_ratio_resize = kwargs.get('aspect_ratio_resize', False)
        self.image_size = kwargs.get('image_size', (384, 384))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        # get file name + text
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert('RGB')

        if self.do_resize \
            and self.aspect_ratio_resize \
                and (image.size[0] < self.image_size[0] or image.size[1] < self.image_size[1]):
            image = self.resize_with_aspect_ratio(image=image, target_size=self.image_size)
        elif self.do_resize:
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)

        pixel_values = self.processor(image, return_tensors='pt').pixel_values

        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding='max_length',
                                          max_length=self.max_target_length).input_ids

        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label
                if label != self.processor.tokenizer.pad_token_id
                else -100 for label in labels]

        encoding = {'pixel_values': pixel_values.squeeze(), 'labels': torch.tensor(labels)}
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

# run the evaluation
def run_batch_evaluation(
        test_dataloader: DataLoader,
        model: VisionEncoderDecoderModel,
        processor: TrOCRProcessor,
        name: str,
        max_new_tokens: int=100
    ) -> float:
    """
    Run the evaluation
    test_dataloader: DataLoader
    model: VisionEncoderDecoderModel
    processor: TrOCRProcessor
    max_new_tokens: maximum number of new tokens to generate (default: 100)
    """
    cer_metric = load('cer')
    gt_txt = []
    predicted_txt = []

    # evaluate the model
    print("Running evaluation...")

    for batch in tqdm(test_dataloader):
        # predict using generate
        pixel_values = batch["pixel_values"].to(DEVICE)
        outputs = model.generate(pixel_values, max_new_tokens=max_new_tokens)

        # decode
        pred_str = processor.batch_decode(outputs, skip_special_tokens=True)
        pred_str = [bytes(t, "utf-8").decode("unicode_escape") for t in pred_str]
        labels = batch["labels"]
        labels[labels == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels, skip_special_tokens=True)
        label_str = [bytes(t, "utf-8").decode("unicode_escape") for t in label_str]
        cer_metric.add_batch(predictions=pred_str, references=label_str)
        gt_txt.extend(label_str)
        predicted_txt.extend(pred_str)

    with open(f'./gt_{name}.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(gt_txt))

    with open(f'./hypothesis_{name}.txt', 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(predicted_txt))

    return cer_metric.compute()

def evaluation(
        dataframe: pd.DataFrame,
        model: VisionEncoderDecoderModel,
        processor: TrOCRProcessor,
        name: str,
        **kwargs,
        ) -> float:
    """
    Run the evaluation
    dataframe: dataframe with columns 'file_name' and 'text'
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

    test_dataset = TrOCRDataset(
        root_dir=image_folder,
        df=dataframe,
        processor=processor,
        do_resize=do_resize,
        aspect_ratio_resize=aspect_ratio_resize,
    )

    print('Number of test lines:', len(test_dataset))

    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=custom_collate_fn,
        batch_size=8,
        shuffle=False,
    )

    cer_metric = run_batch_evaluation(
        test_dataloader,
        model,
        processor,
        name,
        max_new_tokens=max_new_tokens,
    )

    return cer_metric

def main():
    """
    Main function
    """
    # load the model and processor
    model, processor = load_model_processor(TROCR_MODEL, TROCR_PROCESSOR)

    # load the dataframe
    df = pd.read_csv(GT_FILE_TEST, sep='\t', names=['file_name', 'text'])

    # run the evaluation
    cer = evaluation(
        df,
        model,
        processor,
        name=NAME_EVAL,
        image_folder=IMAGE_FOLDER,
        do_resize=DO_RESIZE,
        aspect_ratio_resize=ASPECT_RATIO_RESIZE,
        max_new_tokens=100,
    )

    print(f"CER: {cer}")


if __name__ == '__main__':
    print('Starting evaluation...')
    main()
