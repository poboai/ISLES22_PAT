import SimpleITK
import numpy as np
import json
import os
from pathlib import Path
import glob
DEFAULT_INPUT_PATH = Path("/input")
DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH = Path("/output/images/")
DEFAULT_ALGORITHM_OUTPUT_FILE_PATH = Path("/output/results.json")

#nnunet
from nnunet.inference.predict import predict_from_folder
from nnunet.inference.ensemble_predictions import merge
from batchgenerators.utilities.file_and_folder_operations import *

# Preprocessing Code
import ants

# todo change with your team-name
class PAT():
    def __init__(self,
                 input_path: Path = DEFAULT_INPUT_PATH,
                 output_path: Path = DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH):

        self.debug = True  # False for running the docker!
        if self.debug:
            self._input_path = Path('test')
            self._output_path = Path('test/output')
            self._algorithm_output_path = self._output_path / 'stroke-lesion-segmentation'
            self._output_file = self._output_path / 'results.json'
            self._case_results = []

        else:
            self._input_path = input_path
            self._output_path = output_path
            self._algorithm_output_path = self._output_path / 'stroke-lesion-segmentation'
            self._output_file = DEFAULT_ALGORITHM_OUTPUT_FILE_PATH
            self._case_results = []

    def predict(self, input_data):
        """
        Input   input_data, dict.
                The dictionary contains 3 images and 3 json files.
                keys:  'dwi_image' , 'adc_image', 'flair_image', 'dwi_json', 'adc_json', 'flair_json'

        Output  prediction, array.
                Binary mask encoding the lesion segmentation (0 background, 1 foreground).
        """
        # Get all image inputs.
        dwi_image, adc_image, flair_image = input_data['dwi_image'],\
                                            input_data['adc_image'],\
                                            input_data['flair_image']
        # Get all json inputs.
        dwi_json, adc_json, flair_json = input_data['dwi_json'],\
                                         input_data['adc_json'],\
                                         input_data['flair_json']

        ################################################################################################################
        #################################### Beginning of your prediction method. ######################################
        # todo replace with your best model here!

        ## Step0. Save in User Folder
        print('\nStep0. Save in User Folder\n')
        sdir = 'usr_folder'
        if not os.path.exists(sdir):os.mkdir(sdir)
        step0_dir = 'usr_folder/step0'
        if not os.path.exists(step0_dir):os.mkdir(step0_dir)
        # adc, dwi, flair
        SimpleITK.WriteImage(adc_image, '%s/sample_0000.nii.gz'%(step0_dir))  
        SimpleITK.WriteImage(dwi_image, '%s/sample_0001.nii.gz'%(step0_dir))
        SimpleITK.WriteImage(flair_image, '%s/sample_0002.nii.gz'%(step0_dir))  
        print(SimpleITK.GetArrayFromImage(dwi_image).shape)#(73, 112, 112)
        print(SimpleITK.GetArrayFromImage(adc_image).shape) #(73, 112, 112)
        print(SimpleITK.GetArrayFromImage(flair_image).shape)#(352, 352, 281)
        
        ## Step1. Preprocessing
        print('\nStep1. Preprocessing\n')
        step1_dir = 'usr_folder/step1'
        if not os.path.exists(step1_dir):os.mkdir(step1_dir)
        # Registration
        img_fix = ants.image_read('%s/sample_0000.nii.gz'%(step0_dir))
        img_move = ants.image_read('%s/sample_0002.nii.gz'%(step0_dir))
        mytx_syn = ants.registration(fixed=img_fix, moving=img_move, type_of_transform="SyN")
        warped_moving_syn = mytx_syn['warpedmovout']
        # Savedata
        SimpleITK.WriteImage(adc_image, '%s/sample_0000.nii.gz'%(step1_dir))  
        SimpleITK.WriteImage(dwi_image, '%s/sample_0001.nii.gz'%(step1_dir))
        ants.image_write(warped_moving_syn, '%s/sample_0002.nii.gz'%(step1_dir), ri=True)
        # Printout Output
        print(SimpleITK.GetArrayFromImage(dwi_image).shape)#(73, 112, 112)
        print(SimpleITK.GetArrayFromImage(adc_image).shape) #(73, 112, 112)
        print(warped_moving_syn.shape)#(73, 112, 112)
        
        ## Step2. nnUNet
        print('\nStep2. nnUNet\n')
        step2_dir = 'usr_folder/step2'
        if not os.path.exists(step2_dir):os.mkdir(step2_dir)
        step2_2d_dir = 'usr_folder/step2_2d'
        if not os.path.exists(step2_2d_dir):os.mkdir(step2_2d_dir)
        step2_3d_dir = 'usr_folder/step2_3d'
        if not os.path.exists(step2_3d_dir):os.mkdir(step2_3d_dir)
        # 2d Network Run
        model_folder_name = 'nnUNet_model/2d'
        predict_from_folder(model_folder_name, step1_dir, step2_2d_dir, None, True, 6,
                            2, None, 0, 1, True,overwrite_existing=True, mode="normal", overwrite_all_in_gpu=None,
                            mixed_precision=True,step_size=0.5, checkpoint_name='model_final_checkpoint')

        # 3d Network Run
        model_folder_name = 'nnUNet_model/3d_fullres'
        predict_from_folder(model_folder_name, step1_dir, step2_3d_dir, None, True, 6, #save_npz True
                            2, None, 0, 1, True,overwrite_existing=True, mode="normal", overwrite_all_in_gpu=None,
                            mixed_precision=True,step_size=0.5, checkpoint_name='model_final_checkpoint')
        
        # Merge 2d, 3d Network 
        merge([step2_2d_dir, step2_3d_dir],step2_dir,2,True,None)

        # Step3. Load Results
        print('\nStep3. Load Results\n')
        step2_data_dir = glob.glob('%s/*.nii.gz'%(step2_dir))
        final_output = SimpleITK.ReadImage(step2_data_dir)
        
        prediction = np.squeeze(SimpleITK.GetArrayFromImage(final_output))##(189, 233, 197)
        
        #################################### End of your prediction method. ############################################
        ################################################################################################################
        
        return prediction.astype(int)

    def process_isles_case(self, input_data, input_filename):
        # Get origin, spacing and direction from the DWI image.
        origin, spacing, direction = input_data['dwi_image'].GetOrigin(),\
                                     input_data['dwi_image'].GetSpacing(),\
                                     input_data['dwi_image'].GetDirection()

        # Segment images.
        prediction = self.predict(input_data) # function you need to update!

        # Build the itk object.
        output_image = SimpleITK.GetImageFromArray(prediction)
        output_image.SetOrigin(origin), output_image.SetSpacing(spacing), output_image.SetDirection(direction)

        # Write segmentation to output location.
        if not self._algorithm_output_path.exists():
            os.makedirs(str(self._algorithm_output_path))
        output_image_path = self._algorithm_output_path / input_filename
        SimpleITK.WriteImage(output_image, str(output_image_path))

        # Write segmentation file to json.
        if output_image_path.exists():
            json_result = {"outputs": [dict(type="Image", slug="stroke-lesion-segmentation",
                                                 filename=str(output_image_path.name))],
                           "inputs": [dict(type="Image", slug="dwi-brain-mri",
                                           filename=input_filename)]}

            self._case_results.append(json_result)
            self.save()


    def load_isles_case(self):
        """ Loads the 6 inputs of ISLES22 (3 MR images, 3 metadata json files accompanying each MR modality).
        Note: Cases missing the metadata will still have a json file, though their fields will be empty. """

        # Get MR data paths.
        dwi_image_path = self.get_file_path(slug='dwi-brain-mri', filetype='image')
        adc_image_path = self.get_file_path(slug='adc-brain-mri', filetype='image')
        flair_image_path = self.get_file_path(slug='flair-brain-mri', filetype='image')

        # Get MR metadata paths.
        dwi_json_path = self.get_file_path(slug='dwi-mri-acquisition-parameters', filetype='json')
        adc_json_path = self.get_file_path(slug='adc-mri-parameters', filetype='json')
        flair_json_path = self.get_file_path(slug='flair-mri-acquisition-parameters', filetype='json')

        input_data = {'dwi_image': SimpleITK.ReadImage(str(dwi_image_path)), 'dwi_json': json.load(open(dwi_json_path)),
                      'adc_image': SimpleITK.ReadImage(str(adc_image_path)), 'adc_json': json.load(open(adc_json_path)),
                      'flair_image': SimpleITK.ReadImage(str(flair_image_path)), 'flair_json': json.load(open(flair_json_path))}

        # Set input information.
        input_filename = str(dwi_image_path).split('/')[-1]
        return input_data, input_filename

    def get_file_path(self, slug, filetype='image'):
        """ Gets the path for each MR image/json file."""

        if filetype == 'image':
            file_list = list((self._input_path / "images" / slug).glob("*.nii.gz"))# *.mha"))
        elif filetype == 'json':
            file_list = list(self._input_path.glob("*{}.json".format(slug)))

        # Check that there is a single file to load.
        if len(file_list) != 1:
            print('Loading error')
        else:
            return file_list[0]

    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results, f)

    def process(self):
        input_data, input_filename = self.load_isles_case()
        self.process_isles_case(input_data, input_filename)


if __name__ == "__main__":
    # todo change with your team-name
    PAT().process()
