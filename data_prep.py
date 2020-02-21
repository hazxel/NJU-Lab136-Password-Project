from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import json
import os
import torch
import logging
import random
from torch.nn import utils as nn_utils
from config import *


class Password:
    all_letters = string.ascii_letters + string.digits + ".,?/;:'+-*!@#$%^&"
    n_letters = len(all_letters) + 1 # Plus EOS marker
    default_txt_file_path = './data/rockyou.txt'
    default_json_file_path = './data/rockyou_cleaned.json'
    
    def __init__(self, txt_file_path = default_txt_file_path, json_file_path = default_json_file_path, update_all = False):
        logging.info("Initializing passwords...")
        
        self.txt_file_path = txt_file_path
        self.json_file_path = json_file_path
        
        if os.path.isfile(json_file_path) and not(update_all):
            logging.debug("Loading from existing json file...")
            with open(json_file_path, 'r', encoding = 'utf-8') as pas:
                self.passwords_string = json.load(pas)
        else:
            logging.debug("Reading passwords from file...")
            logging.debug("Converting unicode to ASCII and ignoring spaces...")

            self.passwords_string = Password.readPass(self.txt_file_path)

            logging.debug("Deleting empty passwords...")
            self.passwords_string = Password.deleteEmpty(self.passwords_string)
        
            logging.debug("Saving to json file...")
            with open(json_file_path, 'w',encoding='utf-8') as pas:
                json.dump(self.passwords_string, pas, ensure_ascii = True)
            
        #logging.debug("Converting string to tensor...")
        #self.passwords_tensor = Password.passToTensor(self.passwords_string)
        
        self.string_gen = self.inf_string_gen()
        logging.info("Done initializing passwords.")
    
    def getPasswords(self):
        return self.passwords_string
    
    def passIter(self):
        return iter(self.passwords_string)
    
    def inf_string_gen(self):
        while True:
            random.shuffle(self.passwords_string)
            for i in range(0, len(self.passwords_string) - BATCH_SIZE + 1, BATCH_SIZE):
                yield self.passwords_string[i : i + BATCH_SIZE]
            logging.info("Current epoch done.")
            
    def inf_tensor_pack_gen(self, seq_len):
        while True:
            string_batch = self.string_gen.__next__()
            tensor_in = torch.zeros(BATCH_SIZE, seq_len, CHARMAP_LEN)
            pwd_len = []
            for i in range(len(string_batch)):
                l = len(string_batch[i]) if len(string_batch[i]) <= seq_len else seq_len
                pwd_len.append(l)
                for j in range(l):
                    tensor_in[i][j][Password.letterToIndex(string_batch[i][j])] = 1
            yield nn_utils.rnn.pack_padded_sequence(tensor_in, pwd_len, batch_first=True, enforce_sorted=False)        

    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    @staticmethod
    def unicodeToAscii(s):
        s = ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in Password.all_letters
        )
        s = s.replace(' ','')
        return s

    @staticmethod
    def readPass(fp = default_txt_file_path):
        assert os.path.isfile(fp)
        passwords = open(fp, encoding='utf-8', errors='ignore').read().strip().split('\n')
        return [Password.unicodeToAscii(password) for password in passwords ]
    
    @staticmethod
    def deleteEmpty(passwords):
        while("" in passwords):
            passwords.remove("")
        return passwords

    @staticmethod
    def letterToIndex(letter):
        return Password.all_letters.find(letter)

    @staticmethod
    def letterToTensor(letter):
        tensor = torch.zeros(1, 1, Password.n_letters)
        tensor[0][0][Password.letterToIndex(letter)] = 1
        return tensor
    
    @staticmethod
    def passwordToPretrainTensor(password):
        tensor = torch.LongTensor(1, len(password), CHARMAP_LEN).zero_().to(device)
        for i, letter in enumerate(password):
            tensor[0][i][Password.letterToIndex(letter)] = 1
        return tensor
    
    @staticmethod
    def passwordToInputTensor(password):
        tensor = torch.LongTensor(1, len(password)).zero_().to(device)
        for i, letter in enumerate(password):
            tensor[0][i] = Password.letterToIndex(letter)
        return tensor
    
    # Target Tensor is not one-hot tensor
    @staticmethod
    def passwordToTargetTensor(password):
        target = [Password.all_letters.find(password[i]) for i in range(1, len(password))]
        target.append(Password.n_letters - 1)
        return torch.LongTensor(target)
    