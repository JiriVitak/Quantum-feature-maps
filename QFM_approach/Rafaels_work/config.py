import os

class Config:
    def __init__(self):
        self.QXToken = os.getenv('QXToken', '')  # Minha
        self.SIMULATION = os.getenv('SIMULATION', 'False')


        