import random

import torch
from torchvision import transforms
import random
from torch.utils.data import Dataset

from PIL import Image
from transformers import BertTokenizer

from os.path import join
import json

CLASS_NAME = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets',
              'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli',
              'caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry',
              'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
              'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts',
              'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips',
              'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
              'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon',
              'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus',
              'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons',
              'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes',
              'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich',
              'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad',
              'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak',
              'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']
TEXT_MAX_LENGTH = 50
MIN_FREQ = 3
NUMBER_OF_SAMPLES_PER_CLASS = None


class Datapool(Dataset):
    def __init__(self, all_ids, mode):
        self.all_ids = all_ids
        self.mode = mode
        assert len(list(set(self.all_ids))) == len(self.all_ids), "dataset has duplicated ids"
        self.unlabeled_ids = self.all_ids.copy()
        self.labeled_ids = []
        self.sample_ids = self.all_ids.copy()  # default for val and test datapool

    def initialize(self, query_budget: int):
        # query_budget is the number of labels been queried each round
        # random initialization for first batch of labels
        self.labeled_ids = self.unlabeled_ids[:query_budget]
        self.unlabeled_ids = [id for id in self.all_ids if id not in self.labeled_ids]

    def query_for_label(self, queried_ids: list):
        # queried_ids are generated from query strategy
        self.labeled_ids += queried_ids
        self.unlabeled_ids = [id for id in self.all_ids if id not in self.labeled_ids]
        assert len(self.labeled_ids) + len(self.unlabeled_ids) == len(self.all_ids)

    def query(self):
        # prepare unlabeled data index for label querying
        self.mode = "query"
        print("dataset for querying")
        self.sample_ids = self.unlabeled_ids

    def train(self):
        # prepare labeled queried data index for model training
        self.mode = "train"
        print("dataset for training")
        self.sample_ids = self.labeled_ids

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        pass


class Food101(Datapool):
    def __init__(self,
                 mode="train",
                 dataset_root_dir=r"",
                 ):
        self.dataset_root_dir = dataset_root_dir
        self.mode = mode
        assert self.mode in ["train", "dev", "test"]
        with open(join(dataset_root_dir, f"{mode}.json")) as file:
            data_list = json.load(file)
            self.data = {x["id"]: x for x in data_list}

        self.all_ids = list(self.data.keys())
        random.Random(0).shuffle(self.all_ids)

        super(Food101, self).__init__(self.all_ids, self.mode)
        color_distort_strength = 0.5
        color_jitter = transforms.ColorJitter(
            brightness=0.8 * color_distort_strength,
            contrast=0.8 * color_distort_strength,
            saturation=0.8 * color_distort_strength,
            hue=0.2 * color_distort_strength
        )
        gaussian_kernel_size = 21
        self.train_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=gaussian_kernel_size),
            transforms.ToTensor(),
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

        self.sentence_max_len = TEXT_MAX_LENGTH
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def load_bert_tokens(self, sample_id):
        text_tokens = ' '.join(self.data[sample_id]["text_tokens"])
        text_input = self.tokenizer(
            text_tokens,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.sentence_max_len,
            truncation=True,
            return_tensors='pt'
        )
        for k, v in text_input.items():
            text_input[k] = v.squeeze(0)
        return text_input

    def load_image(self, sample_id):
        image_path = join(self.dataset_root_dir, self.data[sample_id]["img_path"])
        with open(image_path, "rb") as f:
            image = Image.open(f)
            image = image.convert("RGB")
        if self.mode == "train":
            image = self.train_transform(image)
        else:
            image = self.val_transform(image)

        return image

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        text_input = self.load_bert_tokens(sample_id)
        class_name = self.data[sample_id]["label"]
        image = self.load_image(sample_id)
        label = torch.tensor(CLASS_NAME.index(class_name), dtype=torch.long)
        return text_input, image, label

