import splitfolders
import logging

input_folder = 'root_image_dir/'

output_folder = 'train_and_validation_dir'

logging.basicConfig(
    filename= 'logs/train_test_split_logs.log',
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s' 
)

splitfolders.ratio(
        input_folder, output = output_folder,
        seed = 42,  ratio = (0.7, 0.2 , 0.1),
        group_prefix = None
        )
print('Build Successful! ')
logging.info('Successfully created train, validation and test sets for the classes!! ')