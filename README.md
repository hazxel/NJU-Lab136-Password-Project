# NJU-Lab136-Password-Project

Datasets are available here:
[Hashes.org](https://hashes.org/public.php) &
[skullsecurity.org](http://downloads.skullsecurity.org/passwords/)

Python package requirement:

    pythoch
    json
    unicodedata
    ...


To modify configuration, touch file
    
    config.py


To train the model, run:

    python improved_train.py

To generate passwords, run:

    python generate.py 

To evaluate the generated passwords, run:

    python evaluate.py 
