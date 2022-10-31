
class Config:
    def __init__(self):
        self.REDOWNLOAD = False  # If False; will assume data is downloaded already
                                # If True; will delete everything in the data folder and redownload

        self.data_url = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/njdjkbxdpn-1.zip'
        self.extract_path = './data/vechicles_raw'

        self.augmentation_fac = 5
