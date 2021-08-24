# Setups #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option("expand_frame_repr", False)

# Authenticate and create the PyDrive client.
# Please follow the steps as instructed when you run the following commands.

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Load the same with information provided above in the link.
fileid = '13zSR_W20MWknPr2cbXS6Ck-GmLufpNPT'
filename = 'diabetic_data_initial.csv'
downloaded = drive.CreateFile({'id':fileid})
downloaded.GetContentFile(filename)
data = pd.read_csv(filename)

# Check the data loaded.
print(data.head())
