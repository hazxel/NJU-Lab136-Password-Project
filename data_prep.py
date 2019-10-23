from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import json
import os
import torch
import logging



class Password:
    all_letters = string.ascii_letters + string.digits + ".,?/;:'+-*!@#$%^&"
    n_letters = len(all_letters) + 1 # Plus EOS marker
    default_txt_file_path = 'data/rockyou.txt'
    default_json_file_path = 'data/rockyou_cleaned.json'
    pointer = 0
    
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
            self.passwords_string = Password.deleteSpace(self.passwords_string)
        
            logging.debug("Saving to json file...")
            with open(json_file_path, 'w',encoding='utf-8') as pas:
                json.dump(self.passwords_string, pas, ensure_ascii = True)
            
        logging.info("Done initializing passwords.")
        #logging.debug("Converting string to tensor...")
        #self.passwords_tensor = Password.passToTensor(self.passwords_string)
    
    def getPasswords(self):
        return self.passwords_string
    
    def next(self):
        self.pointer = self.pointer + 1
        return self.passwords_string[self.pointer - 1]
    
    def passIter(self):
        return iter(self.passwords_string)

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
    def deleteSpace(passwords):
        while("" in passwords):
            passwords.remove("")
        return passwords

    @staticmethod
    def letterToIndex(letter):
        return Password.all_letters.find(letter)

    @staticmethod
    def passToTensor(passwords):
        tensors = []
        for password in passwords:
            tensor = torch.zeros(len(password) + 1, 1, Password.n_letters)
            tensor[-1][0][-1] = 1
            for i, letter in enumerate(password):
                tensor[i][0][Password.letterToIndex(letter)] = 1
            tensors.append(tensor)
        return tensors
    
    @staticmethod
    def passwordToInputTensor(password):
        tensor = torch.zeros(len(password), 1, Password.n_letters)
        for i, letter in enumerate(password):
            tensor[i][0][Password.letterToIndex(letter)] = 1
        return tensor
    
    # Target Tensor is not one-hot tensor
    @staticmethod
    def passwordToTargetTensor(password):
        target = [Password.all_letters.find(password[i]) for i in range(1, len(password))]
        target.append(Password.n_letters - 1)
        return torch.LongTensor(target)